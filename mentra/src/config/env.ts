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

export const env = {
  packageName: required("PACKAGE_NAME"),
  mentraApiKey: required("MENTRAOS_API_KEY"),
  port: parseNumber("PORT", 3000),
  aiServerUrl: required("AI_SERVER_URL"),
  mockAiServer: parseBool("MOCK_AI_SERVER", true),
  healthCheckTimeoutMs: parseNumber("HEALTH_CHECK_TIMEOUT_MS", 5000),
} as const;

export type Env = typeof env;
