# Golden Sign AI Server

## 실행 환경

- Python 3.13
- FastAPI
- Uvicorn

## 설치

pip install -r requirements.txt

## 실행

python -m uvicorn app.main:app --reload

## Swagger

http://127.0.0.1:8000/docs

## 구현 API

GET /health

POST /v1/sessions

## 현재 상태

- Mock Model 사용
- 실제 MediaPipe 미연동
- 실제 AI Model 미연동