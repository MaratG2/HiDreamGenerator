import type { GenerationRequest, RuntimeSettings } from "../types";

interface ErrorResponse {
  error?: string;
}

interface StartResponse extends ErrorResponse {
  job_id?: string;
}

export type JobEvent =
  | { type: "progress"; step: number; total: number; preview?: string }
  | { type: "done"; image: string; width: number; height: number; seed: number; mock_mode: boolean; engine?: string }
  | { type: "error"; message: string };

async function parseResponse<T extends ErrorResponse>(response: Response): Promise<T> {
  let payload: T;
  try {
    payload = (await response.json()) as T;
  } catch {
    throw new Error(`The backend returned an unreadable response (${response.status}). Check that the image server is running.`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

export async function startGeneration(request: GenerationRequest, signal?: AbortSignal): Promise<string> {
  const response = await fetch("/api/generate/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  const payload = await parseResponse<StartResponse>(response);
  if (!payload.job_id) throw new Error("The backend did not return a job id.");
  return payload.job_id;
}

export async function refinePrompt(prompt: string): Promise<string> {
  const response = await fetch("/api/refine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const payload = await parseResponse<{ prompt?: string; error?: string }>(response);
  if (!payload.prompt) throw new Error("The refiner returned an empty prompt.");
  return payload.prompt;
}

export async function getRuntimeSettings(): Promise<RuntimeSettings> {
  const response = await fetch("/api/runtime");
  return parseResponse<RuntimeSettings & ErrorResponse>(response);
}

export async function updateRuntimeSettings(mockMode: boolean): Promise<RuntimeSettings> {
  const response = await fetch("/api/runtime", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mock_mode: mockMode }),
  });
  return parseResponse<RuntimeSettings & ErrorResponse>(response);
}

export function subscribeToJob(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onTransportError: () => void,
): () => void {
  const stream = new EventSource(`/api/generate/stream/${jobId}`);
  let finished = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  const close = () => {
    finished = true;
    clearTimeout(reconnectTimer);
    stream.close();
  };
  stream.onopen = () => { clearTimeout(reconnectTimer); reconnectTimer = undefined; };

  stream.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data) as JobEvent;
      if (!["progress", "done", "error"].includes(event.type)) throw new Error("Unknown event");
      if (event.type === "done" && (typeof event.image !== "string" || !event.width || !event.height)) {
        throw new Error("Missing image metadata");
      }
      if (event.type === "done" || event.type === "error") close();
      onEvent(event);
    } catch {
      close();
      onTransportError();
    }
  };
  stream.onerror = () => {
    if (finished) return;
    if (stream.readyState === EventSource.CLOSED) {
      close();
      onTransportError();
    } else if (!reconnectTimer) {
      // EventSource reconnects with Last-Event-ID; the backend replays missed events.
      reconnectTimer = setTimeout(() => { close(); onTransportError(); }, 60_000);
    }
  };
  return () => {
    close();
  };
}
