import WebSocket from "ws";
import type { StoredPhoto } from "../server/manager/PhotoManager";
import type { SessionResponse } from "../types/api";

const SCHEMA_VERSION = "dev-0.2";

type StreamState = "idle" | "connecting" | "ready" | "closed" | "failed";

interface AIServerStreamClientOptions {
  session: SessionResponse;
  userId: string;
  onResult?: (text: string) => void;
}

export class AIServerStreamClient {
  private socket: WebSocket | null = null;
  private state: StreamState = "idle";
  private frameIndex = 0;

  constructor(private readonly options: AIServerStreamClientOptions) {}

  async connect(): Promise<boolean> {
    const wsUrl = this.options.session.ws_url;
    if (!wsUrl) {
      console.log("[AIServerStream] no ws_url returned; skipping stream");
      return false;
    }

    this.state = "connecting";

    return new Promise((resolve) => {
      const socket = new WebSocket(wsUrl);
      this.socket = socket;

      const fail = (error: Error) => {
        this.state = "failed";
        console.error("[AIServerStream] connection failed:", error.message);
        resolve(false);
      };

      socket.once("error", fail);
      socket.once("open", () => {
        socket.off("error", fail);
        this.send({
          type: "hello",
          schema_version: SCHEMA_VERSION,
          session_id: this.options.session.session_id,
          client_message_id: `hello-${Date.now()}`,
          client: "mentra",
          user_id: this.options.userId,
          capabilities: {
            frame_encoding: ["jpeg_base64"],
            max_frame_bytes: 262144,
          },
        });
        resolve(true);
      });

      socket.on("message", (raw) => this.handleMessage(raw.toString()));
      socket.on("close", () => {
        this.state = this.state === "failed" ? "failed" : "closed";
        this.socket = null;
        console.log("[AIServerStream] closed");
      });
      socket.on("error", (error) => {
        this.state = "failed";
        console.error("[AIServerStream] error:", error.message);
      });
    });
  }

  sendFrame(photo: StoredPhoto): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn("[AIServerStream] frame skipped; socket is not open");
      return false;
    }

    const frameIndex = ++this.frameIndex;
    this.send({
      type: "frame",
      schema_version: SCHEMA_VERSION,
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

  stop(reason = "app_stopped"): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.socket = null;
      this.state = "closed";
      return;
    }

    this.send({
      type: "stop",
      schema_version: SCHEMA_VERSION,
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
    };

    if (typedMessage.type === "ready") {
      this.state = "ready";
      console.log("[AIServerStream] ready");
      return;
    }
    if (typedMessage.type === "ack") {
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
      return;
    }

    console.log("[AIServerStream] message:", raw);
  }
}
