import WebSocket from "ws";
import type { StoredPhoto } from "../server/manager/PhotoManager";
import type { SessionResponse } from "../types/api";

const FRAME_SCHEMA_VERSION = "dev-0.2";
const WEBRTC_SCHEMA_VERSION = "dev-0.3";
const DEFAULT_CONNECTION_TIMEOUT_MS = 5000;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_DELAY_MS = 1000;

type StreamState = "idle" | "connecting" | "ready" | "closed" | "failed";

interface AIServerStreamClientOptions {
  session: SessionResponse;
  userId: string;
  onResult?: (text: string) => void;
  onStreamError?: (code: string, message?: string) => void;
  onReady?: () => void;
  connectionTimeoutMs?: number;
  maxRetries?: number;
  retryDelayMs?: number;
}

export class AIServerStreamClient {
  private socket: WebSocket | null = null;
  private state: StreamState = "idle";
  private frameIndex = 0;
  private shouldReconnect = true;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingConnectSettler: ((connected: boolean) => void) | null = null;

  constructor(private readonly options: AIServerStreamClientOptions) {}

  async connect(): Promise<boolean> {
    const wsUrl = this.options.session.ws_url;
    if (!wsUrl) {
      console.log("[AIServerStream] no ws_url returned; skipping stream");
      return false;
    }

    this.shouldReconnect = true;
    this.clearReconnectTimer();
    const maxRetries = this.options.maxRetries ?? DEFAULT_MAX_RETRIES;

    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      if (!this.shouldReconnect) {
        return false;
      }

      const connected = await this.connectOnce(wsUrl, attempt);
      if (connected || !this.shouldReconnect) {
        return connected;
      }

      if (attempt < maxRetries) {
        await this.waitBeforeRetry(attempt);
      }
    }

    this.state = "failed";
    console.error("[AIServerStream] all connection attempts failed");
    return false;
  }

  private connectOnce(wsUrl: string, attempt: number): Promise<boolean> {
    this.state = "connecting";

    return new Promise((resolve) => {
      let settled = false;
      const socket = new WebSocket(wsUrl);
      let timeout: ReturnType<typeof setTimeout>;
      const connectionTimeoutMs =
        this.options.connectionTimeoutMs ?? DEFAULT_CONNECTION_TIMEOUT_MS;
      this.socket = socket;

      const cleanup = () => {
        clearTimeout(timeout);
        socket.off("open", handleOpen);
        socket.off("error", handleError);
        if (this.pendingConnectSettler === settle) {
          this.pendingConnectSettler = null;
        }
      };

      const settle = (connected: boolean) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(connected);
      };

      const handleError = (error: Error) => {
        if (this.socket !== socket) {
          settle(false);
          return;
        }
        this.state = "failed";
        console.error(
          `[AIServerStream] connection attempt ${attempt + 1} failed:`,
          error.message,
        );
        socket.terminate();
        settle(false);
      };

      const handleOpen = () => {
        if (this.socket !== socket) {
          settle(false);
          return;
        }
        this.send({
          type: "hello",
          schema_version: FRAME_SCHEMA_VERSION,
          session_id: this.options.session.session_id,
          client_message_id: `hello-${Date.now()}`,
          client: "mentra",
          user_id: this.options.userId,
          capabilities: {
            frame_encoding: ["jpeg_base64"],
            max_frame_bytes: 262144,
          },
        });
        settle(true);
      };

      timeout = setTimeout(() => {
        if (this.socket !== socket) {
          settle(false);
          return;
        }
        this.state = "failed";
        console.error(
          `[AIServerStream] connection attempt ${attempt + 1} timed out`,
        );
        socket.terminate();
        settle(false);
      }, connectionTimeoutMs);

      this.pendingConnectSettler = settle;
      socket.once("error", handleError);
      socket.once("open", handleOpen);
      socket.on("message", (raw) => this.handleMessage(raw.toString()));
      socket.on("close", () => {
        if (this.socket !== socket) {
          return;
        }
        const wasReady = this.state === "ready";
        this.state = this.state === "failed" ? "failed" : "closed";
        this.socket = null;
        console.log("[AIServerStream] closed");

        if (wasReady && this.shouldReconnect) {
          this.scheduleReconnect();
        }
      });
      socket.on("error", (error) => {
        if (this.socket !== socket) {
          return;
        }
        this.state = "failed";
        console.error("[AIServerStream] error:", error.message);
      });
    });
  }

  private waitBeforeRetry(attempt: number): Promise<void> {
    const retryDelayMs = this.options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS;
    const delayMs = retryDelayMs * 2 ** attempt;
    console.log(`[AIServerStream] reconnecting in ${delayMs}ms`);
    return new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || !this.shouldReconnect) return;

    const retryDelayMs = this.options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS;
    console.log(`[AIServerStream] reconnecting in ${retryDelayMs}ms`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect();
    }, retryDelayMs);
  }

  private clearReconnectTimer(): void {
    if (!this.reconnectTimer) return;

    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  sendFrame(photo: StoredPhoto): boolean {
    if (
      !this.isReady() ||
      !this.socket ||
      this.socket.readyState !== WebSocket.OPEN
    ) {
      console.warn("[AIServerStream] frame skipped; socket is not ready");
      return false;
    }

    const frameIndex = ++this.frameIndex;
    this.send({
      type: "frame",
      schema_version: FRAME_SCHEMA_VERSION,
      session_id: this.options.session.session_id,
      client_message_id: `frame-${frameIndex}`,
      request_id: photo.requestId,
      frame_index: frameIndex,
      captured_at: photo.timestamp.toISOString(),
      image: {
        encoding: "jpeg_base64",
        mime_type: photo.mimeType,
        data: photo.buffer.toString("base64"),
      },
    });
    return true;
  }

  sendStreamStart(webrtcUrl: string, streamId: string): boolean {
    if (!webrtcUrl || !streamId) {
      console.warn("[AIServerStream] stream_start skipped; stream info is missing");
      return false;
    }
    if (!this.sendWhenReady("stream_start")) {
      return false;
    }

    this.send({
      type: "stream_start",
      schema_version: WEBRTC_SCHEMA_VERSION,
      session_id: this.options.session.session_id,
      client_message_id: `stream-start-${Date.now()}`,
      webrtc_url: webrtcUrl,
      stream_id: streamId,
    });
    return true;
  }

  sendStreamStop(streamId: string): boolean {
    if (!streamId) {
      console.warn("[AIServerStream] stream_stop skipped; stream_id is missing");
      return false;
    }
    if (!this.sendWhenReady("stream_stop")) {
      return false;
    }

    this.send({
      type: "stream_stop",
      schema_version: WEBRTC_SCHEMA_VERSION,
      session_id: this.options.session.session_id,
      client_message_id: `stream-stop-${Date.now()}`,
      stream_id: streamId,
    });
    return true;
  }

  stop(reason = "app_stopped"): void {
    this.shouldReconnect = false;
    this.clearReconnectTimer();
    this.pendingConnectSettler?.(false);

    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.socket?.terminate();
      this.socket = null;
      this.state = "closed";
      return;
    }

    this.send({
      type: "stop",
      schema_version: FRAME_SCHEMA_VERSION,
      session_id: this.options.session.session_id,
      client_message_id: `stop-${Date.now()}`,
      reason,
    });
    this.socket.close(1000);
  }

  isReady(): boolean {
    return this.state === "ready";
  }

  private send(payload: object): void {
    this.socket?.send(JSON.stringify(payload));
  }

  private sendWhenReady(messageType: string): boolean {
    if (
      !this.isReady() ||
      !this.socket ||
      this.socket.readyState !== WebSocket.OPEN
    ) {
      console.warn(`[AIServerStream] ${messageType} skipped; socket is not ready`);
      return false;
    }
    return true;
  }

  private handleMessage(raw: string): void {
    let message: unknown;
    try {
      message = JSON.parse(raw);
    } catch {
      console.error("[AIServerStream] invalid JSON from server");
      return;
    }

    if (!message || typeof message !== "object" || !("type" in message)) {
      console.error("[AIServerStream] invalid message from server:", raw);
      return;
    }

    const typedMessage = message as {
      type: string;
      result?: { text?: string };
      code?: string;
      message?: string;
      status?: string;
      stream_id?: string;
    };

    if (typedMessage.type === "ready") {
      this.state = "ready";
      console.log("[AIServerStream] ready");
      this.options.onReady?.();
      return;
    }
    if (typedMessage.type === "ack") {
      if (typedMessage.status === "stream_start_accepted") {
        console.log(
          `[AIServerStream] stream_start acknowledged stream=${typedMessage.stream_id}`,
        );
        return;
      }
      if (typedMessage.status === "stream_stop_accepted") {
        console.log(
          `[AIServerStream] stream_stop acknowledged stream=${typedMessage.stream_id}`,
        );
        return;
      }
      console.log("[AIServerStream] frame acknowledged");
      return;
    }
    if (typedMessage.type === "result") {
      const text = typedMessage.result?.text;
      console.log("[AIServerStream] result:", text ?? "(empty)");
      if (text) this.options.onResult?.(text);
      return;
    }
    if (typedMessage.type === "error") {
      console.error(
        `[AIServerStream] server error ${typedMessage.code}: ${typedMessage.message}`,
      );
      if (typedMessage.code === "stream_unavailable") {
        this.options.onStreamError?.(typedMessage.code, typedMessage.message);
      }
      return;
    }

    console.log("[AIServerStream] message:", raw);
  }
}
