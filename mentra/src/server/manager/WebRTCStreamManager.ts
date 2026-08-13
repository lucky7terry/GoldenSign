import type { AppSession, ManagedStreamStatus } from "@mentra/sdk";
import type { User } from "../session/User";

type WebRTCStreamState =
  | "idle"
  | "starting"
  | "active"
  | "stopping"
  | "stopped"
  | "error";

interface ActiveManagedStream {
  streamId: string;
  webrtcUrl: string;
  streamStartSent: boolean;
}

export class WebRTCStreamManager {
  private state: WebRTCStreamState = "idle";
  private activeStream: ActiveManagedStream | null = null;
  private cleanupStatusListener: (() => void) | null = null;
  private operation = Promise.resolve();
  private stopRequested = false;
  private destroyed = false;
  private sentStreamStartIds = new Set<string>();
  private sentStreamStopIds = new Set<string>();

  constructor(private readonly user: User) {}

  setup(session: AppSession): void {
    this.destroyed = false;
    this.cleanupStatusListener?.();
    this.cleanupStatusListener = session.camera.onManagedStreamStatus((status) => {
      this.handleManagedStreamStatus(status);
    });
  }

  async toggle(): Promise<void> {
    if (this.state === "starting") {
      this.stopRequested = true;
      this.state = "stopping";
      this.showStatus("Golden Sign\nStopping WebRTC stream...");
      return;
    }

    if (this.state === "active" || this.state === "stopping") {
      await this.stop("button_toggle");
      return;
    }

    await this.start();
  }

  isActive(): boolean {
    return this.state === "active";
  }

  start(): Promise<void> {
    return this.enqueue(() => this.startInternal());
  }

  stop(reason = "app_stopped"): Promise<void> {
    this.stopRequested = true;
    return this.enqueue(() => this.stopInternal(reason));
  }

  destroy(): void {
    this.destroyed = true;
    this.stopRequested = true;
    this.cleanupStatusListener?.();
    this.cleanupStatusListener = null;
    void this.enqueue(() => this.stopInternal("cleanup", false));
  }

  handleAiStreamError(code: string, message?: string): void {
    if (code !== "stream_unavailable") {
      return;
    }

    this.showStatus(
      `Golden Sign\nWebRTC stream unavailable\n${message ?? "Please try again."}`,
    );
    this.stopRequested = true;
    void this.enqueue(() => this.stopInternal("stream_unavailable", false));
  }

  private enqueue(operation: () => Promise<void>): Promise<void> {
    const next = this.operation.then(operation, operation);
    this.operation = next.catch(() => undefined);
    return next;
  }

  private async startInternal(): Promise<void> {
    const session = this.user.appSession;
    if (this.destroyed) {
      return;
    }
    if (!session) {
      console.warn("[WebRTCStream] start skipped; no active glasses session");
      return;
    }
    if (!this.user.aiStream?.isReady()) {
      console.warn("[WebRTCStream] start skipped; AI WebSocket is not ready");
      this.showStatus("Golden Sign\nAI stream is not ready");
      return;
    }

    this.stopRequested = false;
    this.state = "starting";
    this.showStatus("Golden Sign\nStarting WebRTC stream...");

    try {
      const stream = await session.camera.startManagedStream({
        quality: "720p",
        enableWebRTC: true,
        video: { frameRate: 12 },
      });

      if (this.destroyed || this.stopRequested) {
        await this.stopManagedStream(session, "start_cancelled");
        this.clearStream(stream.streamId);
        this.state = "stopped";
        return;
      }

      if (!stream.webrtcUrl) {
        this.state = "error";
        this.showStatus("Golden Sign\nWebRTC URL was not returned");
        console.error("[WebRTCStream] managed stream did not include webrtcUrl");
        return;
      }

      this.setActiveStream(stream.streamId, stream.webrtcUrl);
    } catch (error) {
      if (this.destroyed || this.stopRequested) {
        this.state = "stopped";
        return;
      }

      this.state = "error";
      console.error("[WebRTCStream] failed to start managed stream:", error);
      this.showStatus("Golden Sign\nWebRTC stream start failed");
    }
  }

  private async stopInternal(reason: string, showStoppedStatus = true): Promise<void> {
    const session = this.user.appSession;
    const streamId = this.getCurrentStreamId();

    this.sendStreamStopOnce(streamId);

    if (!streamId && !session?.camera.isManagedStreamActive()) {
      this.stopRequested = false;
      this.state = "stopped";
      return;
    }

    this.state = "stopping";
    await this.stopManagedStream(session, reason);
    this.clearStream(streamId);
    this.stopRequested = false;
    this.state = "stopped";

    if (showStoppedStatus) {
      this.showStatus("Golden Sign\nWebRTC stream stopped");
    }
  }

  private handleManagedStreamStatus(status: ManagedStreamStatus): void {
    console.log(
      `[WebRTCStream] status=${status.status} stream=${status.streamId ?? "(none)"}`,
    );

    if (status.status === "active" && status.webrtcUrl && status.streamId) {
      if (this.destroyed || this.stopRequested) {
        this.sendStreamStopOnce(status.streamId);
        void this.enqueue(() =>
          this.stopInternal("active_after_stop_requested", false),
        );
        return;
      }

      this.state = "active";
      this.setActiveStream(status.streamId, status.webrtcUrl);
      this.sendStreamStartIfReady("managed_stream_status");
      return;
    }

    if (status.status === "stopped") {
      const streamId = status.streamId ?? this.activeStream?.streamId;
      this.sendStreamStopOnce(streamId);
      this.clearStream(streamId);
      this.stopRequested = false;
      this.state = "stopped";
      return;
    }

    if (status.status === "error") {
      const streamId = status.streamId ?? this.activeStream?.streamId;
      this.sendStreamStopOnce(streamId);
      this.clearStream(streamId);
      this.stopRequested = false;
      this.state = "error";
      this.showStatus(`Golden Sign\nWebRTC stream error\n${status.message ?? ""}`);
    }
  }

  private setActiveStream(streamId: string, webrtcUrl: string): void {
    this.activeStream = {
      streamId,
      webrtcUrl,
      streamStartSent: this.sentStreamStartIds.has(streamId),
    };
  }

  private sendStreamStartIfReady(source: string): void {
    if (
      this.state !== "active" ||
      !this.activeStream ||
      this.activeStream.streamStartSent
    ) {
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
    this.sentStreamStopIds.delete(this.activeStream.streamId);
    this.showStatus("Golden Sign\nWebRTC stream connected");
    console.log(
      `[WebRTCStream] stream_start sent from ${source} stream=${this.activeStream.streamId}`,
    );
  }

  private sendStreamStopOnce(streamId?: string): void {
    if (!streamId || this.sentStreamStopIds.has(streamId)) {
      return;
    }

    const sent = this.user.aiStream?.sendStreamStop(streamId);
    if (sent) {
      this.sentStreamStopIds.add(streamId);
      this.sentStreamStartIds.delete(streamId);
    }
  }

  private getCurrentStreamId(): string | undefined {
    return this.activeStream?.streamId;
  }

  private async stopManagedStream(
    session: AppSession | null | undefined,
    reason: string,
  ): Promise<void> {
    try {
      await session?.camera.stopManagedStream();
    } catch (error) {
      console.error(`[WebRTCStream] failed to stop managed stream (${reason}):`, error);
    }
  }

  private clearStream(streamId?: string): void {
    if (streamId) {
      this.sentStreamStartIds.delete(streamId);
    }
    this.activeStream = null;
  }

  private showStatus(text: string): void {
    try {
      this.user.appSession?.layouts.showTextWall(text);
    } catch (error) {
      console.error("[WebRTCStream] failed to show status:", error);
    }
  }
}
