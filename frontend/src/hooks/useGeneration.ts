import { useCallback, useEffect, useRef, useState } from "react";

import { startGeneration, subscribeToJob } from "../api/hidream";
import type { BatchItemState, GenerationRequest, GenerationPhase, GenerationResult } from "../types";

export interface GenerationState {
  phase: GenerationPhase;
  currentIndex: number;
  totalImages: number;
  items: BatchItemState[];
  error: string | null;
}

const initialState: GenerationState = {
  phase: "idle",
  currentIndex: 0,
  totalImages: 0,
  items: [],
  error: null,
};

function updateItem(
  items: BatchItemState[],
  index: number,
  update: Partial<BatchItemState>,
): BatchItemState[] {
  return items.map((item) => item.index === index ? { ...item, ...update } : item);
}

export function useGeneration() {
  const [state, setState] = useState<GenerationState>(initialState);
  const closeStreamRef = useRef<null | (() => void)>(null);
  const runIdRef = useRef(0);
  const busyRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const rejectRef = useRef<((reason: Error) => void) | null>(null);

  useEffect(() => () => {
    runIdRef.current += 1;
    abortRef.current?.abort();
    closeStreamRef.current?.();
    rejectRef.current?.(new Error("Generation view closed"));
  }, []);

  const reset = useCallback(() => {
    runIdRef.current += 1;
    closeStreamRef.current?.();
    closeStreamRef.current = null;
    abortRef.current?.abort();
    rejectRef.current?.(new Error("Generation view reset"));
    rejectRef.current = null;
    busyRef.current = false;
    setState(initialState);
  }, []);

  const generate = useCallback(async (
    request: GenerationRequest,
    batchSize: number,
    onDone: (result: GenerationResult, index: number) => void,
  ) => {
    if (busyRef.current) return;
    busyRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    runIdRef.current += 1;
    const runId = runIdRef.current;
    closeStreamRef.current?.();
    const items: BatchItemState[] = Array.from({ length: batchSize }, (_, index) => ({
      index,
      seed: request.seed + index,
      phase: index === 0 ? "starting" : "queued",
      step: 0,
      total: request.num_inference_steps,
      previewUrl: null,
      resultUrl: null,
      error: null,
    }));
    setState({ phase: "starting", currentIndex: 0, totalImages: batchSize, items, error: null });

    try {
      for (let index = 0; index < batchSize; index += 1) {
        if (runId !== runIdRef.current) return;
        const itemRequest = { ...request, seed: request.seed + index };
        setState((current) => ({
          ...current,
          phase: "starting",
          currentIndex: index,
          items: updateItem(current.items, index, {
            phase: "starting",
            step: 0,
            previewUrl: null,
            error: null,
          }),
        }));

        const jobId = await startGeneration(itemRequest, controller.signal);
        if (runId !== runIdRef.current) return;
        const result = await new Promise<GenerationResult>((resolve, reject) => {
          rejectRef.current = reject;
          let settled = false;
          const finish = (callback: () => void) => {
            if (settled) return;
            settled = true;
            callback();
          };
          closeStreamRef.current = subscribeToJob(
            jobId,
            (event) => {
              if (runId !== runIdRef.current) return;
              if (event.type === "progress") {
                setState((current) => ({
                  ...current,
                  phase: "generating",
                  currentIndex: index,
                  items: updateItem(current.items, index, {
                    phase: "generating",
                    step: event.step,
                    total: event.total,
                    previewUrl: event.preview
                      ? `data:image/jpeg;base64,${event.preview}`
                      : current.items[index]?.previewUrl ?? null,
                  }),
                }));
              } else if (event.type === "done") {
                finish(() => resolve({ imageUrl: `data:image/png;base64,${event.image}`, width: event.width,
                  height: event.height, seed: event.seed, mock_mode: event.mock_mode, engine: event.engine }));
              } else {
                finish(() => reject(new Error(event.message)));
              }
            },
            () => finish(() => reject(new Error("The generation stream disconnected unexpectedly."))),
          );
        });

        if (runId !== runIdRef.current) return;
        setState((current) => ({
          ...current,
          phase: index === batchSize - 1 ? "complete" : "starting",
          currentIndex: Math.min(index + 1, batchSize - 1),
          items: updateItem(
            updateItem(current.items, index, {
              phase: "complete",
              step: current.items[index]?.total ?? itemRequest.num_inference_steps,
              previewUrl: null,
              resultUrl: result.imageUrl,
              width: result.width, height: result.height, mockMode: result.mock_mode, engine: result.engine,
            }),
            index + 1,
            index + 1 < batchSize ? { phase: "starting" } : {},
          ),
        }));
        rejectRef.current = null;
        onDone(result, index);
      }
    } catch (error) {
      if (runId !== runIdRef.current) return;
      const message = error instanceof Error ? error.message : "Generation could not start.";
      setState((current) => ({
        ...current,
        phase: "error",
        error: message,
        items: current.items.map((item) => item.phase === "complete" ? item : {
          ...item, phase: item.index === current.currentIndex ? "error" : "skipped",
          error: item.index === current.currentIndex ? message : "Not started because an earlier image failed.",
        }),
      }));
    } finally {
      if (runId === runIdRef.current) {
        busyRef.current = false;
        rejectRef.current = null;
        closeStreamRef.current = null;
      }
    }
  }, []);

  return { state, generate, reset };
}
