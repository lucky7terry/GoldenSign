# GoldenSign AI 서버 이미지.
# TF 2.17 은 Python 3.13 을 지원하지 않으므로 3.12 로 고정한다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# MediaPipe / OpenCV 가 import 시 찾는 공유 라이브러리.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts/download_mediapipe_models.py ./scripts/

EXPOSE 8000

# models/ 는 호스트에서 볼륨으로 붙인다(인식 모델 .keras 는 저장소에 없다).
# MediaPipe .task 파일은 없을 때만 받는다 - 두 번째 기동부터는 건너뛴다.
# --proxy-headers: Caddy 뒤에서 X-Forwarded-Proto 를 믿어야 ws_url 이
# wss:// 로 나간다. 없으면 앱이 ws:// 로 붙으려다 실패한다.
CMD ["sh", "-c", "python scripts/download_mediapipe_models.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'"]
