import type { Context } from "hono";
import { sessions } from "../manager/SessionManager";

/** GET /health */
export function getHealth(c: Context) {
  return c.json({ status: "ok", timestamp: new Date().toISOString() });
}

/**
 * GET /session-status?userId=...
 * onSession의 Golden Sign 흐름 결과(connecting/ready/failed)를 웹뷰에 노출한다.
 * 아직 glasses 세션이 시작되지 않아 User가 없으면 "connecting"으로 응답.
 */
export function getSessionStatus(c: Context) {
  const userId = c.req.query("userId");
  if (!userId) return c.json({ error: "userId is required" }, 400);

  const user = sessions.get(userId);
  const status = user?.sessionStatus ?? { state: "connecting", modelVersion: null };
  return c.json(status);
}
