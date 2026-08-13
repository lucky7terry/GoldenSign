import type { AppSession, ManagedStreamStatus } from "@mentra/sdk";
import type { User } from "../session/User";

type WebRTCStreamState = "idle" | "starting" | "active" | "stopping" | "stopped" | "error";

interface ActiveManagedStream {
  streamId: string;
  webrtcUrl: string;
  streamStartSent: boolean;
}

export class WebRTCStreamManager {
  private state: WebRTCStreamState = "idle";
  private activeStream: ActiveManagedStream | null = null;
  private cleanupStatusListener: (() => void) | null = null;
  private sentStreamStartIds = new Set<string>();

  constructor(private readonly user: User) {}

  setup(session: AppSession): void {
    this.cleanupStatusListener?.();
    this.cleanupStatusListener = session.camera.onManagedStreamStatus((status) => {
      this.handleManagedStreamStatus(status);
    });
  }

  async toggle(): Promise<void> {
    if (this.state === "active" || this.state === "starting") {
      await this.stop("button_toggle");
      return;
    }

    await this.start();
  }

  isActive(): boolean {
    return this.state === "active" || this.state === "starting";
  }

  async start(): Promise<void> {
    const session = this.user.appSession;
    if (!session) {
      console.warn("[WebRTCStream] start skipped; no active glasses session");
      return;
    }
    if (!this.user.aiStream?.isReady()) {
      console.warn("[WebRTCStream] start skipped; AI WebSocket is not ready");
      this.showStatus("Golden Sign\nAI stream is not ready");
      return;
    }

    this.state = "starting";
    this.showStatus("Golden Sign\nStarting WebRTC stream...");

    try {
      const stream = await session.camera.startManagedStream({
        quality: "720p",
        enableWebRTC: true,
        video: { frameRate: 12 },
      });

      if (!stream.webrtcUrl) {
        this.state = "error";
        this.showStatus("Golden Sign\nWebRTC URL was not returned");
        console.error("[WebRTCStream] managed stream did not include webrtcUrl");
        return;
      }

      this.activeStream = {
        streamId: stream.streamId,
        webrtcUrl: stream.webrtcUrl,
        streamStartSent: this.sentStreamStartIds.has(stream.streamId),
      };
      this.sendStreamStartIfReady("startManagedStream");
    } catch (error) {
      this.state = "error";
      console.error("[WebRTCStream] failed to start managed stream:", error);
      this.showStatus("Golden Sign\nWebRTC stream start failed");
    }
  }

  async stop(reason = "app_stopped"): Promise<void> {
    const session = this.user.appSession;
    const streamId = this.activeStream?.streamId;

    if (!streamId && !session?.camera.isManagedStreamActive()) {
      this.state = "stopped";
      return;
    }

    if (streamId) {
      this.user.aiStream?.sendStreamStop(streamId);
    }

    this.state = "stopping";
    try {
      await session?.camera.stopManagedStream();
    } catch (error) {
      console.error(`[WebRTCStream] failed to stop managed stream (${reason}):`, error);
    } finally {
      if (streamId) {
        this.sentStreamStartIds.delete(streamId);
      }
      this.activeStream = null;
      this.state = "stopped";
      this.showStatus("Golden Sign\nWebRTC stream stopped");
    }
  }

  destroy(): void {
    this.cleanupStatusListener?.();
    this.cleanupStatusListener = null;
    void this.stop("cleanup");
  }

  private handleManagedStreamStatus(status: ManagedStreamStatus): void {
    console.log(
      `[WebRTCStream] status=${status.status} stream=${status.streamId ?? "(none)"}`,
    );

    if (status.status === "active" && status.webrtcUrl && status.streamId) {
      this.state = "active";
      this.activeStream = {
        streamId: status.streamId,
        webrtcUrl: status.webrtcUrl,
        streamStartSent: this.sentStreamStartIds.has(status.streamId),
      };
      this.sendStreamStartIfReady("managed_stream_status");
      return;
    }

    if (status.status === "stopped") {
      this.activeStream = null;
      this.state = "stopped";
      return;
    }

    if (status.status === "error") {
      const streamId = status.streamId ?? this.activeStream?.streamId;
      if (streamId) {
        this.user.aiStream?.sendStreamStop(streamId);
      }
      this.activeStream = null;
      this.state = "error";
      this.showStatus(`Golden Sign\nWebRTC stream error\n${status.message ?? ""}`);
    }
  }

  private sendStreamStartIfReady(source: string): void {
    if (!this.activeStream || this.activeStream.streamStartSent) {
      return;
    }

    const sent = this.user.aiStream?.sendStreamStart(
      this.activeStream.webrtcUrl,
      this.activeStream.streamId,
    );
    if (!sent) {
      return;
    }

    this.activeStream.streamStartSent = true;
    this.sentStreamStartIds.add(this.activeStream.streamId);
    this.state = "active";
    this.showStatus("Golden Sign\nWebRTC stream connected");
    console.log(
      `[WebRTCStream] stream_start sent from ${source} stream=${this.activeStream.streamId}`,
    );
  }

  private showStatus(text: string): void {
    try {
      this.user.appSession?.layouts.showTextWall(text);
    } catch (error) {
      console.error("[WebRTCStream] failed to show status:", error);
    }
  }
}
