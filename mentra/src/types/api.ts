// Golden Sign AI Server API 타입 정의.
// docs/api-contract.md와 반드시 일치해야 함.
// 서버가 이 형태로 응답한다는 계약이고, 실제 서버 구현 시에도 이 형태를 유지해야 함.

// ----- GET /health -----
export interface HealthResponse {
  status: "ok" | string;
  api: "ready" | string;
  model: {
    loaded: boolean;
    mode: string; // "mock" | "production" 등
    version: string; // 예: "mock-0.1"
  };
  time: string; // ISO 8601
}

// ----- POST /v1/sessions -----
export interface CreateSessionRequest {
  client: "mentra";
  user_id: string;
}

export interface SessionResponse {
  session_id: string;
  status: "created" | "active" | "stopped" | string;
  schema_version: string; // 예: "dev-0.1"
  ws_url: string | null;
}

// ----- GET /v1/sessions/{id} -----
// Week 1에서는 호출하지 않음. 타입만 정의.
export type GetSessionResponse = SessionResponse;

// ----- POST /v1/sessions/{id}/stop -----
// Week 1에서는 호출하지 않음. 함수 틀만 준비.
export interface StopSessionResponse {
  session_id: string;
  status: "stopped" | string;
}

// ----- 클라이언트 공통 결과 타입 -----
// AIServerClient의 각 메서드는 Result 형태를 반환해서
// 호출부(PR4)가 성공/실패를 분기하기 쉽게 함.
export type AIServerResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: AIServerError };

export interface AIServerError {
  kind: "timeout" | "network" | "parse" | "http";
  message: string;      // 로그용 상세 메시지
  userMessage: string;  // 사용자 화면용 짧은 메시지 (stack trace 없음)
  status?: number;      // HTTP 상태 (kind === "http"일 때만)
}
