import {
  AlertCircle,
  Check,
  CheckCircle2,
  Clock3,
  Copy,
  Download,
  Hourglass,
  LoaderCircle,
  Sparkles,
  TestTube2,
} from "lucide-react";

import type { GenerationState } from "../hooks/useGeneration";
import type { BatchItemState, SourceImage, StudioMode } from "../types";

interface WorkspaceProps {
  mode: StudioMode;
  sourceImage: SourceImage | null;
  generation: GenerationState;
  selectedImageUrl: string | null;
  selectedEngine?: string;
  mockMode: boolean;
  notice: string | null;
  onDownload: (url: string, seed?: number) => void;
  onCopy: (url: string) => void;
}

const phaseLabel = (item: BatchItemState) => {
  if (item.phase === "complete") return item.engine === "background_composite" ? "Subject preserved" : "Complete";
  if (item.phase === "generating") return `Step ${item.step} of ${item.total}`;
  if (item.phase === "starting") return "Starting";
  if (item.phase === "error") return "Failed";
  if (item.phase === "skipped") return "Not started";
  return "Queued";
};

const phaseIcon = (item: BatchItemState) => {
  if (item.phase === "complete") return <CheckCircle2 size={15} strokeWidth={1.8} />;
  if (item.phase === "generating" || item.phase === "starting") {
    return <LoaderCircle className="spinner" size={15} strokeWidth={1.8} />;
  }
  if (item.phase === "error") return <AlertCircle size={15} strokeWidth={1.8} />;
  return <Hourglass size={15} strokeWidth={1.7} />;
};

function BatchSlot({ item, onCopy, onDownload }: {
  item: BatchItemState;
  onCopy: (url: string) => void;
  onDownload: (url: string, seed?: number) => void;
}) {
  const imageUrl = item.resultUrl || item.previewUrl;
  const progress = item.phase === "complete" ? 100 : item.total ? (item.step / item.total) * 100 : 0;
  return (
    <figure className={`batch-slot is-${item.phase}`}>
      <div className="batch-slot-media">
        {imageUrl ? <img src={imageUrl} alt={item.resultUrl ? `Result ${item.index + 1}` : `Preview ${item.index + 1}`} /> : (
          <div className="slot-placeholder" aria-hidden="true">
            {item.phase === "queued" ? <Hourglass size={25} strokeWidth={1.25} /> : <Sparkles size={27} strokeWidth={1.25} />}
          </div>
        )}
        {item.phase === "starting" || item.phase === "generating" ? (
          <div className="slot-active-overlay">
            <LoaderCircle className="spinner" size={22} strokeWidth={1.6} />
            <strong>{item.phase === "starting" ? "Starting model job" : `Rendering ${item.step} / ${item.total}`}</strong>
          </div>
        ) : null}
        {item.resultUrl ? (
          <figcaption className="result-actions">
            <button type="button" onClick={() => onCopy(item.resultUrl!)} aria-label={`Copy result ${item.index + 1}`} title="Copy image"><Copy size={16} strokeWidth={1.6} /></button>
            <button type="button" onClick={() => onDownload(item.resultUrl!, item.seed)} aria-label={`Download result ${item.index + 1}`} title="Download PNG"><Download size={16} strokeWidth={1.6} /></button>
          </figcaption>
        ) : null}
      </div>
      <div className="slot-status">
        <div className="slot-status-line">
          <span className="slot-phase">{phaseIcon(item)}{phaseLabel(item)}</span>
          <span>{item.width ? `${item.width} × ${item.height} · ` : ""}Seed {item.seed}</span>
        </div>
        <span className="slot-progress" aria-label={`${Math.round(progress)} percent complete`}><span style={{ width: `${Math.max(progress, item.phase === "starting" ? 2 : 0)}%` }} /></span>
        {item.error ? <p>{item.error}</p> : null}
      </div>
    </figure>
  );
}

export function Workspace({ mode, sourceImage, generation, selectedImageUrl, selectedEngine, mockMode, notice, onDownload, onCopy }: WorkspaceProps) {
  const hasBatch = generation.items.length > 0;
  const completed = generation.items.filter((item) => item.phase === "complete").length;
  const overall = hasBatch
    ? generation.items.reduce((sum, item) => {
        if (item.phase === "complete") return sum + 1;
        if (item.phase === "generating" && item.total) return sum + item.step / item.total;
        return sum;
      }, 0) / generation.items.length * 100
    : 0;
  const current = generation.items[generation.currentIndex] ?? null;
  const idleImage = selectedImageUrl || (mode === "edit" ? sourceImage?.dataUrl : null) || null;

  return (
    <main className={hasBatch ? "workspace-stage has-batch" : "workspace-stage"}>
      {mockMode && !hasBatch ? <span className="stage-mock-badge"><TestTube2 size={14} strokeWidth={1.7} />Mock generation</span> : null}
      {notice ? <span className="copy-notice"><Check size={14} />{notice}</span> : null}

      {hasBatch ? (
        <div className="batch-workspace">
          <header className="batch-summary">
            <div>
              <span className="summary-kicker">{mockMode ? "Mock batch" : generation.phase === "complete" ? "Batch results" : "Active batch"}</span>
              <strong>{generation.phase === "complete" ? `${completed} ${completed === 1 ? "image" : "images"} complete` : `Image ${generation.currentIndex + 1} of ${generation.totalImages}`}</strong>
            </div>
            <div className="batch-overall">
              <span>{current ? phaseLabel(current) : "Preparing"}</span>
              <span className="overall-track" aria-label={`${Math.round(overall)} percent of batch complete`}><span style={{ width: `${overall}%` }} /></span>
              <b>{Math.round(overall)}%</b>
            </div>
          </header>
          <div className="batch-content">
            <div className={`generation-grid count-${generation.items.length}`}>
              {generation.items.map((item) => <BatchSlot key={item.index} item={item} onCopy={onCopy} onDownload={onDownload} />)}
            </div>
            <aside className="batch-queue" aria-label="Batch queue">
              <div className="queue-heading"><span>Queue</span><b>{completed}/{generation.totalImages}</b></div>
              <ol>
                {generation.items.map((item) => (
                  <li className={item.index === generation.currentIndex && generation.phase !== "complete" ? "is-current" : ""} key={item.index}>
                    <span className={`queue-state is-${item.phase}`}>{phaseIcon(item)}</span>
                    <span><strong>Image {item.index + 1}</strong><small>{phaseLabel(item)}</small></span>
                    <em>{item.seed}</em>
                  </li>
                ))}
              </ol>
              {generation.phase === "error" ? <p className="queue-error"><AlertCircle size={14} />{generation.error}</p> : null}
              <div className="queue-note"><Clock3 size={14} /><span>GPU jobs run one at a time. Completed images stay visible while the next starts.</span></div>
            </aside>
          </div>
        </div>
      ) : idleImage ? (
        <div className="single-workspace">
          <figure className="single-result-frame"><img src={idleImage} alt={selectedImageUrl ? "Selected result" : "Source image"} /></figure>
          <div className="stage-caption">
            <span>{selectedImageUrl ? selectedEngine === "background_composite" ? "Result · Subject preserved" : "Result" : sourceImage ? `Source · ${sourceImage.width} × ${sourceImage.height}` : "Image"}</span>
            {selectedImageUrl ? <div className="caption-actions"><button className="download-action" type="button" onClick={() => onCopy(idleImage)}><Copy size={16} />Copy</button><button className="download-action" type="button" onClick={() => onDownload(idleImage)}><Download size={16} />Download PNG</button></div> : null}
          </div>
        </div>
      ) : generation.phase === "error" ? (
        <div className="empty-stage error-stage" role="alert"><AlertCircle size={38} strokeWidth={1.3} /><h1>Generation stopped</h1><p>{generation.error}</p></div>
      ) : (
        <div className="empty-stage"><span className="empty-mark" aria-hidden="true"><Sparkles size={32} strokeWidth={1.3} /></span><h1>{mode === "edit" ? "Begin with an image" : "Start with an idea"}</h1><p>{mode === "edit" ? "Drop in a source image, then describe only what you want to change." : "Describe a scene, subject, or style. HiDream will render it locally."}</p></div>
      )}
    </main>
  );
}
