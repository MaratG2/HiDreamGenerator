export type StudioMode = "t2i" | "edit";
export type EditMethod = "flow_match" | "flash";
export type AspectId = "1:1" | "4:3" | "3:2" | "16:9" | "2:1" | "3:4" | "2:3" | "9:16" | "1:2";
export type ResolutionId = "1k" | "1.5k" | "2k";
export type QualityId = "low" | "medium" | "high" | "best";
export type GenerationPhase = "idle" | "starting" | "generating" | "complete" | "error";
export type BatchItemPhase = "queued" | "starting" | "generating" | "complete" | "error" | "skipped";

export interface BatchItemState {
  index: number;
  seed: number;
  phase: BatchItemPhase;
  step: number;
  total: number;
  previewUrl: string | null;
  resultUrl: string | null;
  error: string | null;
  width?: number;
  height?: number;
  mockMode?: boolean;
}

export interface AspectPreset {
  id: AspectId;
  x: number;
  y: number;
  orientation: "square" | "landscape" | "portrait";
}

export interface ResolutionPreset {
  id: ResolutionId;
  label: string;
  longSide: number;
}

export interface QualityPreset {
  id: QualityId;
  label: string;
  steps: number;
}

export interface RuntimeSettings {
  mock_mode: boolean;
  model_loaded: boolean;
  model_type: string;
  attention_backend: string;
  checkpoint?: { id: string; quantization: string; variant: string | null } | null;
}

export interface SourceImage {
  name: string;
  dataUrl: string;
  base64: string;
  width: number;
  height: number;
}

export interface GenerationRequest {
  mode: StudioMode;
  prompt: string;
  width: number;
  height: number;
  seed: number;
  refs_b64: string[];
  keep_original_aspect: boolean;
  editing_scheduler?: EditMethod;
  num_inference_steps: number;
}

export interface HistoryItem {
  id: string;
  batchId: string;
  batchIndex: number;
  imageUrl: string;
  prompt: string;
  mode: StudioMode;
  seed: number;
  width: number;
  height: number;
  createdAt: number;
  mockMode: boolean;
}

export interface GenerationResult {
  imageUrl: string;
  width: number;
  height: number;
  seed: number;
  mock_mode: boolean;
}
