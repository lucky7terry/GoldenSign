# GoldenSign 

## 프로젝트 소개
응급 상황의 농인 환자가 수어로 증상을 전하면, 의료진이 착용한 스마트글래스(Mentra Live)가 이를 인식해 텍스트와 음성으로 전달하는 시스템입니다.

## 동작 흐름
1. 글래스 버튼을 길게 눌러 카메라 스트리밍 시작 (WebRTC)
2. 수어 시작과 끝에 버튼을 짧게 눌러 단어 구간 지정
3. 서버가 MediaPipe로 좌표를 추출하고 모델로 단어 인식
4. 앱 화면에 단어 표시, 글래스 스피커로 음성 재생

## 디렉터리 구성
- `miniapp/` — Mentra Miniapp SDK 0.3.0 기반 온글래스 앱. background(AI 서버 연결)와 ui(React) 두 엔트리로 나뉜다.
- `server/` — FastAPI + MediaPipe AI 서버. REST(`/health`, `/v1/sessions`)와 WebSocket 세션 엔드포인트를 제공한다.
- `docs/` — API 계약, WebSocket 계약.
- `fixtures/` — 문서에서 참조하는 응답 예시 JSON.
- `infra/` — 배포 설정(Docker Compose + Caddy HTTPS). 사용법은 `infra/README.md`.
