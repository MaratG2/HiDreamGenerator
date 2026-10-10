import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import {
  Gauge,
  ImageUp,
  Layers3,
  Lightbulb,
  RectangleHorizontal,
  RectangleVertical,
  Sparkles,
  Square,
  TestTube2,
  Trash2,
  WandSparkles,
} from "lucide-react";

import type {
  AspectPreset,
  EditMethod,
  QualityPreset,
  ResolutionPreset,
  SourceImage,
  StudioMode,
} from "../types";
import { ModeTabs } from "./ModeTabs";

interface ControlPanelProps {
  mode: StudioMode;
  prompt: string;
  sourceImage: SourceImage | null;
  aspect: AspectPreset;
  aspects: readonly AspectPreset[];
  resolution: ResolutionPreset;
  resolutions: readonly ResolutionPreset[];
  quality: QualityPreset;
  qualities: readonly QualityPreset[];
  dimensions: { width: number; height: number };
  batchSize: number;
  seed: number;
  keepOriginalAspect: boolean;
  editMethod: EditMethod;
  isBusy: boolean;
  isGenerating: boolean;
  canGenerate: boolean;
  modelType: string;
  isRefining: boolean;
  mockMode: boolean;
  error: string | null;
  onModeChange: (mode: StudioMode) => void;
  onPromptChange: (prompt: string) => void;
  onSourceFile: (file: File) => void;
  onRemoveSource: () => void;
  onAspectChange: (aspect: AspectPreset) => void;
  onResolutionChange: (resolution: ResolutionPreset) => void;
  onQualityChange: (quality: QualityPreset) => void;
  onBatchSizeChange: (size: number) => void;
  onSeedChange: (seed: number) => void;
  onKeepOriginalAspectChange: (keep: boolean) => void;
  onEditMethodChange: (method: EditMethod) => void;
  onRefine: () => void;
  onGenerate: () => void;
}

const aspectIcons = {
  square: Square,
  landscape: RectangleHorizontal,
  portrait: RectangleVertical,
};

export function ControlPanel(props: ControlPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const actionDisabled = !props.canGenerate || props.isBusy || !props.prompt.trim() || (props.mode === "edit" && !props.sourceImage);
  const sizeDisabled = props.mode === "edit" && props.keepOriginalAspect;

  const chooseFirstFile = (files: FileList | null) => {
    if (props.isBusy) return;
    const file = files?.[0];
    if (file) props.onSourceFile(file);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    chooseFirstFile(event.target.files);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsDragging(false);
    chooseFirstFile(event.dataTransfer.files);
  };

  return (
    <aside className="control-panel">
      <ModeTabs mode={props.mode} disabled={props.isBusy} onChange={props.onModeChange} />
      <div className="control-scroll">
        {props.mockMode ? (
          <div className="mock-warning" role="status">
            <TestTube2 size={17} strokeWidth={1.6} />
            <span><strong>Mock mode is on.</strong> Results are test placeholders.</span>
          </div>
        ) : null}

        {props.mode === "edit" ? (
          <section className="control-section source-section">
            <label className="field-label">Source image</label>
            <input ref={inputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={handleFileChange} />
            <button
              className={isDragging ? "upload-zone is-dragging" : "upload-zone"}
              type="button"
              disabled={props.isBusy}
              onClick={() => inputRef.current?.click()}
              onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              {props.sourceImage ? <img src={props.sourceImage.dataUrl} alt="Selected source" /> : (
                <span className="upload-placeholder" aria-hidden="true"><ImageUp size={25} strokeWidth={1.5} /></span>
              )}
              <span>Drop an image or <strong>browse</strong></span>
            </button>
            {props.sourceImage ? (
              <button className="text-action remove-action" type="button" disabled={props.isBusy} onClick={props.onRemoveSource}>
                <Trash2 size={16} strokeWidth={1.5} /> Remove
              </button>
            ) : null}
          </section>
        ) : null}

        <section className="control-section prompt-section">
          <label className="field-label" htmlFor="prompt">{props.mode === "edit" ? "Edit instructions" : "Prompt"}</label>
          <textarea
            id="prompt"
            value={props.prompt}
            disabled={props.isBusy}
            onChange={(event) => props.onPromptChange(event.target.value)}
            placeholder={props.mode === "edit" ? "Describe what you want to change…" : "Describe the image you want to create…"}
            maxLength={12000}
            rows={props.mode === "edit" ? 4 : 7}
          />
          <button className="text-action" type="button" disabled={props.isBusy || props.isRefining || !props.prompt.trim()} onClick={props.onRefine}>
            <WandSparkles size={16} strokeWidth={1.5} />
            {props.isRefining ? "Refining…" : "Refine prompt"}
          </button>
        </section>

        {props.mode === "edit" ? (
          <section className="control-section edit-options">
            <label className="toggle-row">
              <span>Keep original aspect</span>
              <input type="checkbox" checked={props.keepOriginalAspect} disabled={props.isBusy} onChange={(event) => props.onKeepOriginalAspectChange(event.target.checked)} />
              <span className="toggle" aria-hidden="true" />
            </label>
            {props.modelType === "dev" ? <><span className="field-label">Edit method</span>
            <div className="method-grid" role="radiogroup" aria-label="Edit method">
              <button className={props.editMethod === "flow_match" ? "method is-selected" : "method"} type="button" role="radio" aria-checked={props.editMethod === "flow_match"} disabled={props.isBusy} onClick={() => props.onEditMethodChange("flow_match")}>
                <Sparkles size={18} strokeWidth={1.5} /> Natural
              </button>
              <button className={props.editMethod === "flash" ? "method is-selected" : "method"} type="button" role="radio" aria-checked={props.editMethod === "flash"} disabled={props.isBusy} onClick={() => props.onEditMethodChange("flash")}>
                <Lightbulb size={18} strokeWidth={1.5} /> Creative
              </button>
            </div>
            </> : <p className="field-hint">Uses the full model’s default scheduler.</p>}
          </section>
        ) : null}

        <section className="control-section size-section">
          <div className="section-heading">
            <span className="field-label">Aspect ratio</span>
            <span>{`${props.dimensions.width} × ${props.dimensions.height}`}</span>
          </div>
          <div className="aspect-grid" role="radiogroup" aria-label="Aspect ratio">
            {props.aspects.map((preset) => {
              const Icon = aspectIcons[preset.orientation];
              return (
                <button key={preset.id} className={props.aspect.id === preset.id ? "aspect-preset is-selected" : "aspect-preset"} type="button" role="radio" aria-checked={props.aspect.id === preset.id} disabled={sizeDisabled || props.isBusy} onClick={() => props.onAspectChange(preset)}>
                  <Icon size={17} strokeWidth={1.5} /><span>{preset.id}</span>
                </button>
              );
            })}
          </div>
          <span className="field-label secondary-label">Resolution</span>
          <div className="segment-grid three" role="radiogroup" aria-label="Resolution">
            {props.resolutions.map((preset) => (
              <button key={preset.id} className={props.resolution.id === preset.id ? "segment is-selected" : "segment"} type="button" role="radio" aria-checked={props.resolution.id === preset.id} disabled={props.isBusy} onClick={() => props.onResolutionChange(preset)}>{preset.label}</button>
            ))}
          </div>
        </section>

        <section className="control-section generation-options">
          <div className="option-label"><Gauge size={16} strokeWidth={1.5} /><span className="field-label">Quality</span></div>
          <div className="quality-grid" role="radiogroup" aria-label="Quality">
            {props.qualities.map((preset) => (
              <button key={preset.id} className={props.quality.id === preset.id ? "quality-preset is-selected" : "quality-preset"} type="button" role="radio" aria-checked={props.quality.id === preset.id} disabled={props.isBusy} onClick={() => props.onQualityChange(preset)}>
                <strong>{preset.label}</strong><span>{preset.steps} steps</span>
              </button>
            ))}
          </div>
          <p className="field-hint">{props.modelType === "dev" ? "28 steps uses the trained dev schedule." : "50 steps recommended for the full model; fewer steps trade detail for speed."}</p>
          <div className="option-label batch-label"><Layers3 size={16} strokeWidth={1.5} /><span className="field-label">Batch</span></div>
          <div className="segment-grid four" role="radiogroup" aria-label="Batch size">
            {[1, 2, 3, 4].map((size) => (
              <button key={size} className={props.batchSize === size ? "segment is-selected" : "segment"} type="button" role="radio" aria-checked={props.batchSize === size} disabled={props.isBusy} onClick={() => props.onBatchSizeChange(size)}>{size}</button>
            ))}
          </div>
        </section>

        <section className="control-section seed-section">
          <label className="field-label" htmlFor="seed">Seed</label>
          <input id="seed" className="seed-input" type="number" min="0" max={4294967295 - props.batchSize + 1} step="1" value={props.seed} disabled={props.isBusy} onChange={(event) => props.onSeedChange(Number(event.target.value) || 0)} />
          {props.batchSize > 1 ? <p className="field-hint">Each result increments this seed by one.</p> : null}
        </section>

        {!props.canGenerate ? <p className="field-hint">Load a model or enable mock generation in Runtime settings.</p> : null}
        {sizeDisabled ? <p className="field-hint">Source proportions, rounded to 64 pixels. Resolution sets the long edge.</p> : null}
        {!props.mockMode ? <p className="field-hint">HiDream renders at about 4 MP or larger, then resizes to your output size. Extra model passes near the end reduce patch-grid artifacts. Allow several minutes.</p> : null}
        {props.error ? <p className="control-error" role="alert">{props.error}</p> : null}
      </div>

      <div className="panel-action">
        <div className="output-summary" aria-label="Generation output summary">
          <span><strong>{`${props.dimensions.width} × ${props.dimensions.height}`}</strong><small>{`${((props.dimensions.width * props.dimensions.height) / 1_000_000).toFixed(1)} MP`}</small></span>
          <span><strong>{props.quality.steps} steps</strong><small>{props.batchSize === 1 ? "1 image" : `${props.batchSize} images · sequential`}</small></span>
        </div>
        <button className="primary-action" type="button" disabled={actionDisabled} onClick={props.onGenerate}>
          <Sparkles size={20} strokeWidth={1.7} />
          {props.isGenerating ? `Generating ${props.batchSize > 1 ? "batch" : "image"}…` : props.isRefining ? "Refining prompt…" : props.mode === "edit" ? `Apply ${props.batchSize > 1 ? `${props.batchSize} edits` : "edit"}` : `Generate ${props.batchSize > 1 ? `${props.batchSize} images` : "image"}`}
        </button>
      </div>
    </aside>
  );
}
