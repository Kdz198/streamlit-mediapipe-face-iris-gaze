# Base Image: Python 3.10 Slim
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống (FIX: Thay libgl1-mesa-glx bằng libgl1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt dependencies
COPY requirements.txt .
# Cài đặt dependencies VÀ XÓA NGAY các thư viện GPU thừa thãi
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /usr/local/lib/python3.10/site-packages/nvidia \
    && rm -rf /usr/local/lib/python3.10/site-packages/torch \
    && rm -rf /usr/local/lib/python3.10/site-packages/triton \
    && rm -rf /usr/local/lib/python3.10/site-packages/jax \
    && rm -rf /usr/local/lib/python3.10/site-packages/jaxlib

# Copy toàn bộ source code
COPY . .

# Expose port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entrypoint
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]