import { AppSession } from "@mentra/sdk";
import { PhotoManager } from "../manager/PhotoManager";
import { TranscriptionManager } from "../manager/TranscriptionManager";
import { AudioManager } from "../manager/AudioManager";
import { InputManager } from "../manager/InputManager";
import { WebRTCStreamManager } from "../manager/WebRTCStreamManager";
import type { AIServerStreamClient } from "../../services/ai-server-stream-client";

/**
 * User — per-user state container.
 *
 * Composes all managers and holds the glasses AppSession.
 * Created when a user connects (glasses or webview) and
 * destroyed when the session is cleaned up.
 */
export class User {
  /** Active glasses connection, null when webview-only */
  appSession: AppSession | null = null;

  /** Photo capture, storage, and SSE broadcasting */
  photo: PhotoManager;

  /** Speech-to-text listener and SSE broadcasting */
  transcription: TranscriptionManager;

  /** Text-to-speech and audio control */
  audio: AudioManager;

  /** Button presses and touchpad gestures */
  input: InputManager;

  /** Managed WebRTC stream lifecycle */
  webrtcStream: WebRTCStreamManager;

  /** AI server WebSocket stream for frame transfer */
  aiStream: AIServerStreamClient | null = null;

  /**
   * Golden Sign 세션 상태. onSession의 health→createSession 흐름 결과를 담아
   * 웹뷰(/api/session-status)가 Ready/실패 문구를 렌더링하는 데 쓴다.
   * 초기값은 "connecting" (아직 흐름 미완료).
   */
  sessionStatus: { state: "connecting" | "ready" | "failed"; modelVersion: string | null } = {
    state: "connecting",
    modelVersion: null,
  };

  constructor(public readonly userId: string) {
    this.photo = new PhotoManager(this);
    this.transcription = new TranscriptionManager(this);
    this.audio = new AudioManager(this);
    this.input = new InputManager(this);
    this.webrtcStream = new WebRTCStreamManager(this);
  }

  /** Wire up a glasses connection — sets up all event listeners */
  setAppSession(session: AppSession): void {
    this.appSession = session;
    this.transcription.setup(session);
    this.input.setup(session);
    this.webrtcStream.setup(session);
    console.log(`📸 Camera ready for ${this.userId}`);
  }

  /** Disconnect glasses but keep user alive (photos, SSE clients stay) */
  clearAppSession(): void {
    this.webrtcStream.destroy();
    this.transcription.destroy();
    this.appSession = null;
  }

  /** Nuke everything — call on full disconnect */
  cleanup(): void {
    this.webrtcStream.destroy();
    this.aiStream?.stop("user_cleanup");
    this.aiStream = null;
    this.transcription.destroy();
    this.photo.destroy();
    this.appSession = null;
  }
}
