# GoldenSign 

프로젝트 소개 및 테스트용 리드미 파일입니다.

## 디렉터리 구성

- `miniapp/` — Mentra Miniapp SDK 0.3.0 기반 온글래스 앱. background(AI 서버 연결)와 ui(React) 두 엔트리로 나뉜다.
- `server/` — FastAPI + MediaPipe AI 서버. REST(`/health`, `/v1/sessions`)와 WebSocket 세션 엔드포인트를 제공한다.
- `docs/` — API 계약, WebSocket 계약.
- `fixtures/` — 문서에서 참조하는 응답 예시 JSON.
- `infra/` — 배포·인프라 설정 자리. 현재는 `.gitkeep`만 있는 빈 디렉터리다.
