/**
 * CameraApp — MentraOS AppServer for the Camera template.
 *
 * Handles the glasses lifecycle (onSession/onStop).
 * All per-user state is managed by the User class via SessionManager.
 */

import { AppServer, AppSession } from "@mentra/sdk";
import { sessions } from "./manager/SessionManager";
import { AIServerClient } from "../services/ai-server-client";

export interface CameraAppConfig {
  packageName: string;
  apiKey: string;
  port: number;
  cookieSecret?: string;
}

export class CameraApp extends AppServer {
  // MentraOS sessionId → AI 서버 session_id 매핑.
  // onStop 시 이 매핑을 조회해 AI 서버 세션을 정리한다(리소스 누수 방지).
  private readonly aiSessionIds = new Map<string, string>();

  constructor(config: CameraAppConfig) {
    super({
      packageName: config.packageName,
      apiKey: config.apiKey,
      port: config.port,
      cookieSecret: config.cookieSecret,
    });
  }

  /** Called when a user launches the app on their glasses */
  protected async onSession(
    session: AppSession,
    sessionId: string,
    userId: string,
  ): Promise<void> {
    // 템플릿 기본 카메라 세션 배선 (삭제 금지, 상태 흐름과 무관하게 항상 유지)
    console.log(`📸 Camera session started for ${userId}`);
    const user = sessions.getOrCreate(userId);
    user.setAppSession(session);

    // ----- Golden Sign 세션 상태 흐름 -----
    session.logger.info(`[GoldenSign] session start user=${userId}`);

    // 1) 초기 표시 (HUD + 웹뷰 상태)
    user.sessionStatus = { state: "connecting", modelVersion: null };
    const initialText =
      "Golden Sign\nMiniApp 실행됨\nAI 서버 상태 확인 중...";
    session.layouts.showTextWall(initialText);

    // 2) health 확인
    const health = await AIServerClient.checkHealth();
    if (!health.ok) {
      session.logger.warn(
        `[GoldenSign] health failed kind=${health.error.kind} msg=${health.error.message}`,
      );
      user.sessionStatus = { state: "failed", modelVersion: null };
      session.layouts.showTextWall(
        "Golden Sign\nAI 서버 연결 실패\n서버 주소 또는 실행 상태를 확인하세요.",
      );
      return;
    }
    session.logger.info(
      `[GoldenSign] health ok model=${health.data.model.version}`,
    );

    // 3) 세션 생성
    const created = await AIServerClient.createSession(userId);
    if (!created.ok) {
      session.logger.warn(
        `[GoldenSign] createSession failed kind=${created.error.kind} msg=${created.error.message}`,
      );
      user.sessionStatus = { state: "failed", modelVersion: null };
      session.layouts.showTextWall(
        "Golden Sign\nAI 서버 연결 실패\n서버 주소 또는 실행 상태를 확인하세요.",
      );
      return;
    }
    // stopSession 호출을 위해 AI 서버 session_id를 sessionId에 매핑해 저장.
    this.aiSessionIds.set(sessionId, created.data.session_id);
    session.logger.info(
      `[GoldenSign] session created id=${created.data.session_id}`,
    );

    // 4) Ready 표시 (HUD + 웹뷰 상태)
    user.sessionStatus = {
      state: "ready",
      modelVersion: health.data.model.version,
    };
    const readyText =
      `Golden Sign Ready\n` +
      `MiniApp Server: OK\n` +
      `AI Server: OK\n` +
      `Model: ${health.data.model.version} loaded`;
    session.layouts.showTextWall(readyText);
  }

  /** Called when a user closes the app or disconnects */
  protected async onStop(
    sessionId: string,
    userId: string,
    reason: string,
  ): Promise<void> {
    console.log(`👋 Camera session ended for ${userId}: ${reason}`);

    // AI 서버 세션 정리 (리소스 누수 방지).
    // mock 모드에서도 stopSession이 동일한 형태로 동작한다.
    const aiSessionId = this.aiSessionIds.get(sessionId);
    if (aiSessionId) {
      try {
        const stopped = await AIServerClient.stopSession(aiSessionId);
        if (stopped.ok) {
          console.log(
            `[GoldenSign] AI session stopped id=${aiSessionId} status=${stopped.data.status}`,
          );
        } else {
          console.error(
            `[GoldenSign] AI session stop failed id=${aiSessionId} kind=${stopped.error.kind} msg=${stopped.error.message}`,
          );
        }
      } catch (err) {
        // stopSession은 내부적으로 에러를 result로 반환하지만, 예기치 못한 예외에도 앱이 죽지 않도록 방어.
        console.error(
          `[GoldenSign] AI session stop threw id=${aiSessionId}:`,
          err,
        );
      } finally {
        // 성공/실패와 무관하게 매핑 제거.
        this.aiSessionIds.delete(sessionId);
      }
    }

    try {
      sessions.remove(userId);
      console.log(`Cleaned up session for ${userId}`);
    } catch (err) {
      console.error(`Error during session cleanup for ${userId}:`, err);
    }
  }
}
