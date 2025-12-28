import whisper
import warnings
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import re

# --- CẤU HÌNH ---
DURATION = 30  # Test 20s cho tập trung
FS = 44100
FILENAME = "test_smart_logic.wav"
warnings.filterwarnings("ignore")


def record_audio(duration, filename):
    print(f"\n🎤 ĐANG THU ÂM TRONG {duration} GIÂY...")
    print("👉 KỊCH BẢN (Cố gắng đọc đúng nhịp):")
    print("1. 'Tên tôi là Nam.' (Nói liền mạch - AI phải tha)")
    print("2. 'Tên tôi... (nghỉ)... là... ờ... Nam.' (Ngập ngừng - AI phải bắt)")
    print("-" * 60)

    device_id = 0  # Hoặc đổi lại mic của bạn
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, device=device_id)
    sd.wait()

    max_val = np.max(np.abs(recording))
    if max_val > 0: recording = recording / max_val * 0.9
    write(filename, FS, recording)
    print(f"✅ Đã lưu file.")


def analyze_smart_logic(audio_path):
    print(f"⏳ Đang tải Model 'MEDIUM' (Với Prompt 'bẩn')...")

    model = whisper.load_model("medium")

    # --- THAY ĐỔI QUYẾT ĐỊNH Ở ĐÂY ---
    # Thay vì ra lệnh, ta đưa ví dụ cụ thể chứa đầy từ đệm.
    # Model sẽ nhìn vào đây và hiểu: "À, phong cách của bài này là phải ghi cả tiếng ậm ừ".
    dirty_prompt = "Dạ thưa... ờ... anh chị, em... à... tên là... ừm... Nguyễn Văn A. Em... ờ... xin phép... à... trình bày."

    result = model.transcribe(
        audio_path,
        language="vi",
        initial_prompt=dirty_prompt,  # Dùng prompt bẩn
        condition_on_previous_text=False,
        word_timestamps=True,
        # Các tham số giúp model nhạy hơn với tiếng động lạ
        beam_size=5,
        best_of=5,
        temperature=0.2
    )

    print("\n" + "=" * 20 + " PHÂN TÍCH THÔNG MINH " + "=" * 20)

    # Mở rộng từ điển để bắt dính hơn
    hard_fillers = ["ờ", "à", "ừm", "um", "hmm", "ha", "hả", "ho", "ui", "uh"]
    soft_fillers = ["thì", "là", "mà", "kiểu", "cái", "rồi", "vậy"]

    detected_errors = []

    print("--- TRANSCRIPT CHI TIẾT ---")

    previous_end_time = 0.0

    for segment in result['segments']:
        line_buffer = ""
        for word_info in segment['words']:
            raw_word = word_info['word']
            # Làm sạch nhẹ nhàng hơn để không mất dấu vết
            clean_word = re.sub(r'[^\w]', '', raw_word).strip().lower()

            start = word_info['start']
            end = word_info['end']
            silence_gap = start - previous_end_time

            is_error = False
            error_type = ""

            # LOGIC 1: Bắt Hard Filler (Ờ, À, Ừm)
            if clean_word in hard_fillers:
                is_error = True
                error_type = "HARD"

            # LOGIC 2: Bắt Soft Filler (Thì, Là + Ngập ngừng)
            # Giảm threshold xuống 0.3s để nhạy hơn
            elif clean_word in soft_fillers and silence_gap > 0.3:
                is_error = True
                error_type = f"SOFT(gap={silence_gap:.2f}s)"

            if is_error:
                line_buffer += f" [❌{raw_word}] "
                detected_errors.append(f"'{raw_word}' ({error_type})")
            else:
                line_buffer += f"{raw_word} "

            previous_end_time = end

        print(line_buffer)

    print("\n" + "=" * 20 + " TỔNG KẾT " + "=" * 20)
    print(f"📝 Full Text: \n{result['text']}")
    print("-" * 50)
    print(f"📊 Số lỗi phát hiện: {len(detected_errors)}")
    if detected_errors:
        print(f"🔍 Chi tiết: {detected_errors}")
    else:
        print("⚠️ Vẫn không bắt được? -> Hãy thử nói 'Ờ' và 'Ừm' to hơn và kéo dài hơn.")


if __name__ == "__main__":
    record_audio(DURATION, FILENAME)
    analyze_smart_logic(FILENAME)