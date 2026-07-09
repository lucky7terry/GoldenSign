# Week 1 실행 가이드

이 문서는 **예원(FastAPI 서버 담당)이 Golden Sign MiniApp 코드를 로컬에서 실행하고 동작을 확인**하기 위한 가이드다.

Mentra Developer Console 앱 관리, ngrok Public URL 등록, 스마트 글래스 실기 테스트는 태리가 담당하므로 이 문서에서 다루지 않는다.

## 사전 요구사항

- Node.js 20+ (`node -v`)
- Bun (`bun -v`)

폰의 Mentra 앱이나 스마트 글래스 실기는 이 실행 가이드에서 필요 없다.

## 실행 절차

```bash
git pull origin main
cd mentra
bun install
cp .env.example .env
```

`.env`를 편집한다:

| 변수 | 값 |
|---|---|
| `PACKAGE_NAME` | 태리가 공유해준 값 그대로 사용 |
| `MENTRAOS_API_KEY` | 태리가 공유해준 값. 없으면 임의 문자열(예: `local-dev-only`) 넣어도 로컬 서버 기동은 됨 |
| `PORT` | `3000` (기본값) |
| `AI_SERVER_URL` | `http://localhost:8000` (기본값 유지) |
| `MOCK_AI_SERVER` | `true` (기본값 유지) |
| `HEALTH_CHECK_TIMEOUT_MS` | `5000` (기본값) |

서버 실행:

```bash
bun run dev
```

## 확인할 것

- 터미널에 서버가 3000 포트에 뜨는 로그가 나온다
- TypeScript 컴파일 에러가 없다 (`bunx tsc --noEmit`)
- `mentra/src/server/CameraApp.ts`의 `onSession` 안에 다음 흐름이 구현되어 있다:
  1. 초기 상태 표시 (`AI 서버 상태 확인 중...`)
  2. `AIServerClient.checkHealth()`
  3. 성공 시 `AIServerClient.createSession(userId)`
  4. 성공 시 Ready 상태 표시
  5. 실패 시 오류 메시지 표시 (무한 로딩 없음)

**주의**: `onSession`은 Mentra 앱에서 MiniApp을 실행해야 호출된다. 예원 로컬 환경에서는 자동 호출되지 않으므로, 이 흐름의 실제 동작 확인은 태리의 실기 테스트에 의존한다. 예원은 코드 리뷰와 서버 기동 확인까지가 이 가이드의 범위다.

## FastAPI 서버 완성 후 통신 테스트

예원이 FastAPI 서버를 완성한 뒤 MiniApp과 통신을 검증하려면:

1. FastAPI 서버를 로컬에서 실행 (예: `http://localhost:8000`)
2. `mentra/.env`에서 `MOCK_AI_SERVER=false`로 변경
3. `AI_SERVER_URL`은 로컬 FastAPI 주소 그대로
4. 통신 검증은 두 가지 방법:
   - **직접 요청 (권장)**: `curl http://localhost:8000/health` 등으로 FastAPI가 계약대로 응답하는지 확인. 계약은 [`api-contract.md`](./api-contract.md) 참고.
   - **태리의 실기 테스트에 붙이기**: 태리에게 `.env` 상태를 공유하고 실기 테스트 요청. 태리가 로그로 결과 공유.

## mock 모드 ↔ 실서버 전환 요약

- `MOCK_AI_SERVER=true`: 실제 HTTP 호출 없이 mock 응답. 서버 없이 MiniApp 코드만 확인할 때.
- `MOCK_AI_SERVER=false` + `AI_SERVER_URL=<실제 서버>`: 실제 FastAPI로 요청.

전환할 때 `.env`만 바꾸면 되고 코드 변경은 없다.

## 문제 해결

| 증상 | 확인 |
|---|---|
| `Missing required env: ...` 에러 | `.env`가 `mentra/` 안에 있고 필수 변수가 채워졌는지 |
| `bun install` 실패 | Bun 버전 확인 (`bun -v`), Node.js 20+ 설치 확인 |
| TypeScript 컴파일 에러 | `bunx tsc --noEmit`로 재현. 코드가 최신인지(`git pull`) 확인 후 태리에게 공유 |
| `MOCK_AI_SERVER=false`인데 timeout | `AI_SERVER_URL`이 실제로 접근 가능한지, FastAPI가 뜨는지 확인 |

## 참고

- API 계약: [`api-contract.md`](./api-contract.md)
- 예시 응답: [`../fixtures/health-response.example.json`](../fixtures/health-response.example.json), [`../fixtures/session-response.example.json`](../fixtures/session-response.example.json)
