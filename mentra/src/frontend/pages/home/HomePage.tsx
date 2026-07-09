import { useState, useEffect, useCallback, useRef } from "react";
import { Camera, Zap, Terminal } from "lucide-react";
import {
  Badge,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "../../components/ui";
import { PhotoStream, type Photo } from "./components/PhotoStream";
import { AudioControls } from "./components/AudioControls";
import {
  TranscriptionFeed,
  type Transcription,
} from "./components/TranscriptionFeed";
import { SystemLogs, type Log } from "./components/SystemLogs";

interface HomePageProps {
  userId: string;
}

interface SessionStatus {
  state: "connecting" | "ready" | "failed";
  modelVersion: string | null;
}

export default function HomePage({ userId }: HomePageProps) {
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [logs, setLogs] = useState<Log[]>([]);
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>({
    state: "connecting",
    modelVersion: null,
  });
  const logIdCounter = useRef(Date.now());

  const addLog = useCallback((message: string) => {
    setLogs((prev) =>
      [
        {
          id: logIdCounter.current++,
          message,
          time: new Date().toLocaleTimeString(),
        },
        ...prev,
      ].slice(0, 20),
    );
  }, []);

  // Connect to SSE photo stream
  useEffect(() => {
    let eventSource: EventSource | null = null;

    const connect = () => {
      try {
        eventSource = new EventSource(
          `/api/photo-stream?userId=${encodeURIComponent(userId)}`,
        );

        eventSource.onopen = () => addLog("Connected to photo stream");

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "connected") return;

            setPhotos((prev) => {
              if (prev.some((p) => p.requestId === data.requestId)) return prev;
              addLog(
                `Photo captured at ${new Date(data.timestamp).toLocaleTimeString()}`,
              );
              return [
                {
                  id: data.requestId,
                  requestId: data.requestId,
                  url: data.dataUrl,
                  timestamp: new Date(data.timestamp).toLocaleTimeString(),
                },
                ...prev,
              ].slice(0, 6);
            });
          } catch {}
        };

        eventSource.onerror = () => {
          addLog("Photo stream disconnected, reconnecting...");
          eventSource?.close();
          setTimeout(connect, 3000);
        };
      } catch {
        addLog("Failed to connect to photo stream");
      }
    };

    connect();
    return () => eventSource?.close();
  }, [addLog, userId]);

  // Connect to SSE transcription stream
  useEffect(() => {
    let eventSource: EventSource | null = null;
    let idCounter = Date.now();

    const connect = () => {
      try {
        eventSource = new EventSource(
          `/api/transcription-stream?userId=${encodeURIComponent(userId)}`,
        );

        eventSource.onopen = () => addLog("Connected to transcription stream");

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "connected") return;

            setTranscriptions((prev) => {
              const entry = {
                id: idCounter++,
                text: data.text,
                time: new Date(data.timestamp).toLocaleTimeString(),
                isFinal: data.isFinal,
              };

              if (data.isFinal) {
                if (prev.length > 0 && !prev[0].isFinal) {
                  const updated = [...prev];
                  updated[0] = { ...updated[0], ...entry, id: updated[0].id };
                  return updated.slice(0, 10);
                }
                return [entry, ...prev].slice(0, 10);
              } else {
                if (prev.length === 0 || prev[0].isFinal) {
                  return [entry, ...prev].slice(0, 10);
                }
                const updated = [...prev];
                updated[0] = { ...updated[0], ...entry, id: updated[0].id };
                return updated;
              }
            });
          } catch {}
        };

        eventSource.onerror = () => {
          addLog("Transcription stream disconnected, reconnecting...");
          eventSource?.close();
          setTimeout(connect, 3000);
        };
      } catch {
        addLog("Failed to connect to transcription stream");
      }
    };

    connect();
    return () => eventSource?.close();
  }, [addLog, userId]);

  // Golden Sign 세션 상태 폴링.
  // onSession(glasses)에서 흐름이 끝나면 /api/session-status가 ready/failed를 반환한다.
  // 아직 connecting이면 짧게 재조회하고, ready/failed로 확정되면 폴링을 멈춘다 (무한 로딩 방지).
  useEffect(() => {
    if (!userId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const res = await fetch(
          `/api/session-status?userId=${encodeURIComponent(userId)}`,
        );
        const data = (await res.json()) as SessionStatus;
        if (stopped) return;
        setSessionStatus(data);
        if (data.state === "connecting") {
          timer = setTimeout(poll, 2000);
        }
      } catch {
        // 조회 실패는 표시 상태를 바꾸지 않고 다음 tick에서 재시도
        if (!stopped) timer = setTimeout(poll, 2000);
      }
    };

    poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [userId]);

  const statusLines =
    sessionStatus.state === "ready"
      ? [
          "Golden Sign Ready",
          "MiniApp Server: OK",
          "AI Server: OK",
          `Model: ${sessionStatus.modelVersion ?? ""} loaded`,
        ]
      : sessionStatus.state === "failed"
        ? [
            "Golden Sign",
            "AI 서버 연결 실패",
            "서버 주소 또는 실행 상태를 확인하세요.",
          ]
        : ["Golden Sign", "MiniApp 실행됨", "AI 서버 상태 확인 중..."];

  const statusClass =
    sessionStatus.state === "ready"
      ? "border-green-500/40 bg-green-500/10 text-foreground"
      : sessionStatus.state === "failed"
        ? "border-destructive/40 bg-destructive/10 text-destructive"
        : "border-muted bg-muted/30 text-muted-foreground";

  return (
    <div className="max-w-5xl mx-auto p-4 md:p-6 space-y-4">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
            <Camera className="w-4 h-4 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">Camera Example App</h1>
            <p className="text-xs text-muted-foreground">MentraOS</p>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-xs mt-2">
          {userId && userId.length > 20 ? `${userId.substring(0, 20)}...` : userId}
        </Badge>
      </div>

      {/* Golden Sign 세션 상태 (health → createSession 흐름 결과) */}
      <div className={`rounded-xl border p-4 text-sm ${statusClass}`}>
        {statusLines.map((line, i) => (
          <div key={i} className={i === 0 ? "font-semibold" : ""}>
            {line}
          </div>
        ))}
      </div>

      {/* Photo Stream */}
      <PhotoStream photos={photos} />

      {/* Audio Controls */}
      <AudioControls userId={userId} onLog={addLog} />

      {/* Transcriptions & Logs */}
      <Tabs defaultValue="transcriptions">
        <TabsList className="w-full">
          <TabsTrigger value="transcriptions" className="flex-1">
            <Zap className="w-3.5 h-3.5" />
            Transcriptions
          </TabsTrigger>
          <TabsTrigger value="logs" className="flex-1">
            <Terminal className="w-3.5 h-3.5" />
            System Logs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="transcriptions">
          <TranscriptionFeed transcriptions={transcriptions} />
        </TabsContent>

        <TabsContent value="logs">
          <SystemLogs logs={logs} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
