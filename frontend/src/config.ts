import type { AspectPreset, QualityPreset, ResolutionPreset } from "./types";

export const ASPECTS: readonly AspectPreset[] = [
  { id: "1:1", x: 1, y: 1, orientation: "square" },
  { id: "4:3", x: 4, y: 3, orientation: "landscape" },
  { id: "3:2", x: 3, y: 2, orientation: "landscape" },
  { id: "16:9", x: 16, y: 9, orientation: "landscape" },
  { id: "2:1", x: 2, y: 1, orientation: "landscape" },
  { id: "3:4", x: 3, y: 4, orientation: "portrait" },
  { id: "2:3", x: 2, y: 3, orientation: "portrait" },
  { id: "9:16", x: 9, y: 16, orientation: "portrait" },
  { id: "1:2", x: 1, y: 2, orientation: "portrait" },
];

export const RESOLUTIONS: readonly ResolutionPreset[] = [
  { id: "1k", label: "1K", longSide: 1024 },
  { id: "1.5k", label: "1.5K", longSide: 1536 },
  { id: "2k", label: "2K", longSide: 2048 },
];

export const QUALITIES: readonly QualityPreset[] = [
  { id: "low", label: "Low", steps: 12 },
  { id: "medium", label: "Medium", steps: 20 },
  { id: "high", label: "High", steps: 32 },
  { id: "best", label: "Best", steps: 50 },
];

export const DEV_QUALITIES: readonly QualityPreset[] = QUALITIES.map((preset) =>
  preset.id === "high" ? { ...preset, steps: 28 } : preset,
);

export function sourceDimensions(width: number, height: number, longSide: number) {
  const scale = longSide / Math.max(width, height);
  return { width: Math.round(width * scale / 64) * 64, height: Math.round(height * scale / 64) * 64 };
}

export function dimensionsFor(aspect: AspectPreset, resolution: ResolutionPreset) {
  const round64 = (value: number) => Math.max(512, Math.round(value / 64) * 64);
  if (aspect.x >= aspect.y) {
    return { width: resolution.longSide, height: round64(resolution.longSide * aspect.y / aspect.x) };
  }
  return { width: round64(resolution.longSide * aspect.x / aspect.y), height: resolution.longSide };
}
