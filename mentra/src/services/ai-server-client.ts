import { env } from "../config/env";
import type {
  AIServerError,
  AIServerResult,
  CreateSessionRequest,
  HealthResponse,
  SessionResponse,
  StopSessionResponse,
} from "../types/api";

// AI 서버 통신 클라이언트.
// MOCK_AI_SERVER=true 이면 실제 HTTP 호출 없이 fixtures와 동일한 형태를 반환.
// MOCK_AI_SERVER=false 이면 AI_SERVER_URL로 실제 요청.
//
// 서버가 없을 때 무한 대기를 방지하기 위해 fetch에 AbortController timeout을 건다.
// 에러는 종류(timeout / network / parse / http)를 구분해 로그로 남긴다.
// 사용자에게 보여줄 짧은 메시지는 userMessage 필드로 함께 반환한다.

const USER_MSG_CONNECTION_FAILED =
  "AI 서버 연결 실패. 서버 주소 또는 실행 상태를 확인하세요.";
const USER_MSG_TIMEOUT =
  "AI 서버 응답이 지연됩니다. 잠시 후 다시 시도하세요.";
const USER_MSG_BAD_RESPONSE =
  "AI 서버 응답 형식이 올바르지 않습니다.";
const USER_MSG_HTTP_ERROR =
  "AI 서버에서 오류를 반환했습니다.";

// ----- mock 응답 (fixtures와 동일 형태 유지) -----

function mockHealth(): HealthResponse {
  return {
    status: "ok",
    api: "ready",
    model: {
      loaded: true,
      mode: "mock",
      version: "mock-0.1",
    },
    time: new Date().toISOString(),
  };
}

function mockSession(): SessionResponse {
  return {
    session_id: "mock-" + Math.random().toString(36).slice(2, 10),
    status: "created",
    schema_version: "dev-0.1",
    ws_url: null,
  };
}

// ----- 공통 fetch 래퍼 -----

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

function classifyError(e: unknown): AIServerError {
  // AbortController timeout
  if (e instanceof Error && e.name === "AbortError") {
    const err: AIServerError = {
      kind: "timeout",
      message: `timeout after ${env.healthCheckTimeoutMs}ms`,
      userMessage: USER_MSG_TIMEOUT,
    };
    console.error("[AIServerClient] timeout:", err.message);
    return err;
  }
  // JSON 파싱 실패
  if (e instanceof SyntaxError) {
    const err: AIServerError = {
      kind: "parse",
      message: `JSON parse failed: ${e.message}`,
      userMessage: USER_MSG_BAD_RESPONSE,
    };
    console.error("[AIServerClient] parse error:", err.message);
    return err;
  }
  // 네트워크/기타
  const msg = e instanceof Error ? e.message : String(e);
  const err: AIServerError = {
    kind: "network",
    message: `network error: ${msg}`,
    userMessage: USER_MSG_CONNECTION_FAILED,
  };
  console.error("[AIServerClient] network error:", err.message);
  return err;
}

function httpError(res: Response): AIServerError {
  const err: AIServerError = {
    kind: "http",
    message: `HTTP ${res.status} ${res.statusText}`,
    userMessage: USER_MSG_HTTP_ERROR,
    status: res.status,
  };
  console.error("[AIServerClient] http error:", err.message);
  return err;
}

// ----- 공개 API -----

export const AIServerClient = {
  async checkHealth(): Promise<AIServerResult<HealthResponse>> {
    if (env.mockAiServer) {
      return { ok: true, data: mockHealth() };
    }
    try {
      const res = await fetchWithTimeout(
        `${env.aiServerUrl}/health`,
        { method: "GET", headers: { "Accept": "application/json" } },
        env.healthCheckTimeoutMs,
      );
      if (!res.ok) return { ok: false, error: httpError(res) };
      const data = (await res.json()) as HealthResponse;
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: classifyError(e) };
    }
  },

  async createSession(
    userId: string,
  ): Promise<AIServerResult<SessionResponse>> {
    if (env.mockAiServer) {
      return { ok: true, data: mockSession() };
    }
    try {
      const body: CreateSessionRequest = { client: "mentra", user_id: userId };
      const res = await fetchWithTimeout(
        `${env.aiServerUrl}/v1/sessions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
          },
          body: JSON.stringify(body),
        },
        env.requestTimeoutMs,
      );
      if (!res.ok) return { ok: false, error: httpError(res) };
      const data = (await res.json()) as SessionResponse;
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: classifyError(e) };
    }
  },

  // Week 1에서는 호출하지 않음. 함수 틀만 유지.
  // 실제 사용은 이후 주차에 추가.
  async stopSession(
    sessionId: string,
  ): Promise<AIServerResult<StopSessionResponse>> {
    if (env.mockAiServer) {
      return {
        ok: true,
        data: { session_id: sessionId, status: "stopped" },
      };
    }
    try {
      const res = await fetchWithTimeout(
        `${env.aiServerUrl}/v1/sessions/${encodeURIComponent(sessionId)}/stop`,
        { method: "POST", headers: { "Accept": "application/json" } },
        env.requestTimeoutMs,
      );
      if (!res.ok) return { ok: false, error: httpError(res) };
      const data = (await res.json()) as StopSessionResponse;
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: classifyError(e) };
    }
  },
};

export type AIServerClientType = typeof AIServerClient;
