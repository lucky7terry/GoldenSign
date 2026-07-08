// 환경변수 로더. 필수값이 없으면 즉시 에러를 던진다.
// 값이 잘못됐을 때 앱이 조용히 이상 동작하는 걸 막기 위함.

function required(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") {
    throw new Error(`Missing required env: ${name}`);
  }
  return v;
}

function parseBool(name: string, defaultValue: boolean): boolean {
  const v = process.env[name];
  if (v === undefined) return defaultValue;
  const lower = v.trim().toLowerCase();
  if (lower === "true" || lower === "1") return true;
  if (lower === "false" || lower === "0") return false;
  throw new Error(`Invalid boolean for ${name}: ${v}`);
}

function parseNumber(name: string, defaultValue: number): number {
  const v = process.env[name];
  if (v === undefined) return defaultValue;
  const n = Number(v);
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid number for ${name}: ${v}`);
  }
  return n;
}

// mock 여부를 먼저 파싱해야 AI_SERVER_URL 검증을 조건부로 적용할 수 있다.
const mockAiServer = parseBool("MOCK_AI_SERVER", true);

// mock 모드에서는 실제 AI 서버로 요청을 보내지 않으므로 AI_SERVER_URL이 없어도 된다.
// mock이 아닐 때만 기존처럼 필수값으로 검증한다.
const aiServerUrl = mockAiServer ? (process.env.AI_SERVER_URL ?? "") : required("AI_SERVER_URL");

export const env = {
  packageName: required("PACKAGE_NAME"),
  mentraApiKey: required("MENTRAOS_API_KEY"),
  port: parseNumber("PORT", 3000),
  // mockAiServer가 true이면 사용되지 않는다(빈 문자열일 수 있음). false일 때만 유효한 URL이 보장된다.
  aiServerUrl,
  mockAiServer,
  healthCheckTimeoutMs: parseNumber("HEALTH_CHECK_TIMEOUT_MS", 5000),
} as const;

console.log(
  `[env] mode=${mockAiServer ? "mock" : "live"} ` +
    `aiServerUrl=${aiServerUrl || "(none)"} port=${env.port}`,
);

export type Env = typeof env;
