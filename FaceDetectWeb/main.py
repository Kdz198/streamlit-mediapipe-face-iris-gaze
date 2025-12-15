import cv2
import mediapipe as mp
import numpy as np
import math
import streamlit as st
import json
import av
import queue
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- 1. CẤU HÌNH ---
st.set_page_config(layout="wide", page_title="Inblue Visual Test")
st.markdown("""
<style>
    div.stButton > button:first-child { width: 100%; font-weight: bold; }
    .css-1y4p8pa { padding-top: 0rem; }
</style>
""", unsafe_allow_html=True)

st.title("Face Detect Visual Pipeline (WebRTC Stable)")

# --- 2. SESSION STATE (ĐÃ FIX LỖI) ---
# XÓA dòng khởi tạo u_file đi, để file_uploader tự lo
# Chỉ khởi tạo giá trị mặc định cho các sliders
defaults = {
    'h_left': 10, 'h_right': 7, 'h_up': 12, 'h_down': 10,
    'd_tol': 0.25,
    'i_left': 0.42, 'i_right': 0.58, 'i_up': 0.38, 'i_down': 0.60
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v


# --- 3. CALLBACK NẠP FILE ---
def load_config_callback():
    # Khi file được upload, hàm này sẽ chạy
    # Lúc này st.session_state.u_file đã tự động có dữ liệu do widget tạo ra
    if st.session_state.u_file is not None:
        try:
            data = json.load(st.session_state.u_file)
            st.session_state.h_left = data['HEAD_THRESHOLDS']['YAW_LEFT']
            st.session_state.h_right = data['HEAD_THRESHOLDS']['YAW_RIGHT']
            st.session_state.h_up = data['HEAD_THRESHOLDS']['PITCH_UP']
            st.session_state.h_down = data['HEAD_THRESHOLDS']['PITCH_DOWN']
            st.session_state.i_left = data['IRIS_THRESHOLDS']['LEFT']
            st.session_state.i_right = data['IRIS_THRESHOLDS']['RIGHT']
            st.session_state.i_up = data['IRIS_THRESHOLDS']['UP']
            st.session_state.i_down = data['IRIS_THRESHOLDS']['DOWN']
            st.session_state.d_tol = data['DISTANCE_TOLERANCE']
            st.toast("Nạp file thành công!", icon="✅")
        except:
            st.error("File lỗi hoặc sai định dạng")


# --- 4. VIDEO PROCESSOR CLASS ---
result_queue = queue.Queue()


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.is_calibrated = False
        self.do_calibration_trigger = False
        self.base_vals = {'pitch': 0, 'yaw': 0, 'width': 0}
        self.thresh = defaults.copy()

    def get_gaze_ratios(self, eye_points, iris_center, img_w, img_h):
        p_left = np.array([eye_points[0].x * img_w, eye_points[0].y * img_h])
        p_right = np.array([eye_points[1].x * img_w, eye_points[1].y * img_h])
        p_top = np.array([eye_points[2].x * img_w, eye_points[2].y * img_h])
        p_bottom = np.array([eye_points[3].x * img_w, eye_points[3].y * img_h])
        p_iris = np.array([iris_center.x * img_w, iris_center.y * img_h])
        width = np.linalg.norm(p_right - p_left)
        dist_x = np.linalg.norm(p_iris - p_left)
        rx = dist_x / width if width != 0 else 0.5
        height = np.linalg.norm(p_bottom - p_top)
        dist_y = np.linalg.norm(p_iris - p_top)
        ry = dist_y / height if height != 0 else 0.5
        return rx, ry

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img_h, img_w, _ = img.shape

            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            status_vn = "..."
            status_color = (200, 200, 200)
            debug_text = ""

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh_points = face_landmarks.landmark
                    face_2d = []
                    face_3d = []

                    lm_left = mesh_points[234];
                    lm_right = mesh_points[454]
                    lx, ly = int(lm_left.x * img_w), int(lm_left.y * img_h)
                    rx, ry = int(lm_right.x * img_w), int(lm_right.y * img_h)
                    curr_width = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

                    key_indices = [1, 199, 33, 263, 61, 291]
                    for idx in key_indices:
                        lm = mesh_points[idx]
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y]);
                        face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    curr_pitch = angles[0] * 360
                    curr_yaw = angles[1] * 360

                    L_PTS = [mesh_points[33], mesh_points[133], mesh_points[159], mesh_points[145]]
                    L_IRIS = mesh_points[468]
                    R_PTS = [mesh_points[362], mesh_points[263], mesh_points[386], mesh_points[374]]
                    R_IRIS = mesh_points[473]
                    rx_l, ry_l = self.get_gaze_ratios(L_PTS, L_IRIS, img_w, img_h)
                    rx_r, ry_r = self.get_gaze_ratios(R_PTS, R_IRIS, img_w, img_h)
                    avg_rx = (rx_l + rx_r) / 2
                    avg_ry = (ry_l + ry_r) / 2

                    if self.do_calibration_trigger:
                        self.base_vals = {'pitch': curr_pitch, 'yaw': curr_yaw, 'width': curr_width}
                        self.is_calibrated = True
                        self.do_calibration_trigger = False
                        try:
                            result_queue.put_nowait({"msg": "calibrated"})
                        except:
                            pass

                    if not self.is_calibrated:
                        cv2.ellipse(img, (img_w // 2, img_h // 2), (120, 180), 0, 0, 360, (0, 255, 255), 2)
                        cv2.putText(img, "BAM START", (img_w // 2 - 80, img_h // 2 + 230), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)
                        status_vn = "CHO BAM START..."
                    else:
                        cv2.ellipse(img, (img_w // 2, img_h // 2), (120, 180), 0, 0, 360, (0, 255, 0), 1)
                        base = self.base_vals
                        d_yaw = curr_yaw - base['yaw']
                        d_pitch = curr_pitch - base['pitch']
                        ratio_dist = curr_width / base['width'] if base['width'] > 0 else 1

                        is_turning = False
                        debug_text = f"Y:{int(d_yaw)} P:{int(d_pitch)} X:{avg_rx:.2f}"
                        t = self.thresh

                        if d_yaw < -t['h_left']:
                            status_vn = "QUAY TRAI";
                            status_color = (0, 165, 255);
                            is_turning = True
                        elif d_yaw > t['h_right']:
                            status_vn = "QUAY PHAI";
                            status_color = (0, 165, 255);
                            is_turning = True
                        elif d_pitch < -t['h_down']:
                            status_vn = "CUI DAU";
                            status_color = (0, 165, 255);
                            is_turning = True
                        elif d_pitch > t['h_up']:
                            status_vn = "NGUOC LEN";
                            status_color = (0, 165, 255);
                            is_turning = True

                        if not is_turning:
                            if ratio_dist > (1 + t['d_tol']):
                                status_vn = "QUA GAN";
                                status_color = (0, 0, 255);
                                is_turning = True
                            elif ratio_dist < (1 - t['d_tol']):
                                status_vn = "QUA XA";
                                status_color = (0, 0, 255);
                                is_turning = True

                        if not is_turning:
                            status_vn = "BINH THUONG";
                            status_color = (0, 255, 0)
                            if avg_rx < t['i_left']:
                                status_vn = "LIEC TRAI";
                                status_color = (255, 0, 255)
                            elif avg_rx > t['i_right']:
                                status_vn = "LIEC PHAI";
                                status_color = (255, 0, 255)
                            elif avg_ry < t['i_up']:
                                status_vn = "NHIN LEN";
                                status_color = (255, 255, 0)
                            elif avg_ry > t['i_down']:
                                status_vn = "NHIN XUONG";
                                status_color = (255, 0, 255)

                        try:
                            result_queue.put_nowait({"debug": debug_text})
                        except:
                            pass

                    cv2.putText(img, status_vn, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
                    if self.is_calibrated:
                        cv2.putText(img, debug_text, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            return frame


# --- 5. GIAO DIỆN (MAIN THREAD) ---
st.sidebar.title("⚙️ Bảng Điều Khiển")
st.sidebar.subheader("1. Thao tác")

start_btn = st.sidebar.button("🚀 BẮT ĐẦU (START)", type="primary")
reset_btn = st.sidebar.button("🔄 CHỈNH LẠI TỪ ĐẦU")

st.sidebar.markdown("---")
# Đã fix lỗi file_uploader tại đây
st.sidebar.file_uploader("📂 Nạp Cấu Hình", type=["json"], key="u_file", on_change=load_config_callback)

col_left, col_center, col_right = st.columns([1, 3, 1])
with col_left:
    st.info("👤 Head Pose (Đầu)")
    st.caption("Nguyên tắc: Số càng NHỎ = Càng NHẠY")

    h_left = st.slider("Quay Trái", 0, 45, key="h_left",
                       help="Kéo nhỏ xuống (sang trái) để NHẠY hơn (Dễ bắt lỗi quay đầu hơn).")

    h_right = st.slider("Quay Phải", 0, 45, key="h_right",
                        help="Kéo nhỏ xuống (sang trái) để NHẠY hơn.")

    h_up = st.slider("Ngước Lên", 0, 45, key="h_up",
                     help="Kéo nhỏ xuống (sang trái) để NHẠY hơn.")

    h_down = st.slider("Cúi Xuống", 0, 45, key="h_down",
                       help="Kéo nhỏ xuống (sang trái) để NHẠY hơn.")

    st.markdown("---")
    d_tol = st.slider("Dung sai Cự ly (%)", 0.1, 0.5, key="d_tol",
                      help="Số càng nhỏ = Càng khắt khe về khoảng cách ngồi.")

with col_right:
    st.info("👁️ Eye Gaze (Mắt)")
    st.caption("Chỉnh độ nhạy bắt liếc mắt")

    # Nhóm < (Nhỏ hơn): Kéo LÊN để nhạy
    st.markdown("**Nhóm 1: Liếc Trái / Lên** (Kéo LỚN = Nhạy)")
    i_left = st.slider("Liếc Trái (<)", 0.30, 0.50, key="i_left",
                       help="Kéo TĂNG LÊN để nhạy hơn.")

    i_up = st.slider("Liếc Lên (<)", 0.20, 0.50, key="i_up",
                     help="Kéo TĂNG LÊN để nhạy hơn.")

    st.markdown("---")

    # Nhóm > (Lớn hơn): Kéo XUỐNG để nhạy
    st.markdown("**Nhóm 2: Liếc Phải / Xuống** (Kéo NHỎ = Nhạy)")
    i_right = st.slider("Liếc Phải (>)", 0.50, 0.80, key="i_right",
                        help="Kéo GIẢM XUỐNG (sang trái) để nhạy hơn.")

    i_down = st.slider("Liếc Xuống (>)", 0.40, 0.80, key="i_down",
                       help="Kéo GIẢM XUỐNG (sang trái) để nhạy hơn.")
curr_conf = {
    "HEAD_THRESHOLDS": {"YAW_LEFT": h_left, "YAW_RIGHT": h_right, "PITCH_UP": h_up, "PITCH_DOWN": h_down},
    "IRIS_THRESHOLDS": {"LEFT": i_left, "RIGHT": i_right, "UP": i_up, "DOWN": i_down},
    "DISTANCE_TOLERANCE": d_tol
}
st.sidebar.download_button("💾 LƯU CẤU HÌNH", json.dumps(curr_conf, indent=4), "config.json", "application/json")

# --- 6. WEBRTC STREAMER ---
with col_center:
    ctx = webrtc_streamer(
        key="visual-test",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,

        # Cấu hình WebRTC với Metered TURN (Đã tối ưu cho Cloudflare Tunnel)
        rtc_configuration={
            "iceServers": [
                # 1. STUN Server (Dò đường cơ bản)
                {
                    "urls": ["stun:stun.relay.metered.ca:80"]
                },

                # 2. TURN UDP Port 80 (Nhanh nhất nếu mạng thoáng)
                {
                    "urls": ["turn:global.relay.metered.ca:80"],
                    "username": "ac617824b367dfbee3468f18",
                    "credential": "Ix60ZpgQZcJSUeLh"
                },

                # 3. TURN TCP Port 80 (QUAN TRỌNG: Dùng để vượt Firewall chặn UDP)
                {
                    "urls": ["turn:global.relay.metered.ca:80?transport=tcp"],
                    "username": "ac617824b367dfbee3468f18",
                    "credential": "Ix60ZpgQZcJSUeLh"
                },

                # 4. TURNS Port 443 (VIP NHẤT: Giả dạng HTTPS để chui qua mọi loại Firewall/Proxy)
                {
                    "urls": ["turns:global.relay.metered.ca:443?transport=tcp"],
                    "username": "ac617824b367dfbee3468f18",
                    "credential": "Ix60ZpgQZcJSUeLh"
                }
            ]
        }
    )

    metrics_ph = st.empty()

    if ctx.video_processor:
        ctx.video_processor.thresh = {
            'h_left': h_left, 'h_right': h_right, 'h_up': h_up, 'h_down': h_down,
            'd_tol': d_tol,
            'i_left': i_left, 'i_right': i_right, 'i_up': i_up, 'i_down': i_down
        }
        if start_btn: ctx.video_processor.do_calibration_trigger = True
        if reset_btn: ctx.video_processor.is_calibrated = False

        try:
            while True:
                msg = result_queue.get_nowait()
                if "msg" in msg and msg["msg"] == "calibrated":
                    st.toast("Đã ghi nhận vị trí chuẩn!", icon="📸")
                if "debug" in msg:
                    metrics_ph.markdown(f"**Thông số thực tế:** `{msg['debug']}`")
        except queue.Empty:
            pass
