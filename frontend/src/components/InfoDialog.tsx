import { AlertTriangle, TestTube2, X } from "lucide-react";
import { useEffect, useRef } from "react";

import type { RuntimeSettings } from "../types";

interface InfoDialogProps {
  panel: "settings" | "help" | null;
  runtime: RuntimeSettings | null;
  runtimeBusy: boolean;
  onMockModeChange: (enabled: boolean) => void;
  onClose: () => void;
}

export function InfoDialog({ panel, runtime, runtimeBusy, onMockModeChange, onClose }: InfoDialogProps) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (panel) dialog.current?.showModal();
    else dialog.current?.close();
  }, [panel]);
  return (
      <dialog ref={dialog} className="info-dialog" aria-labelledby="dialog-title" onCancel={onClose}>
        <div className="dialog-header"><h2 id="dialog-title">{panel === "settings" ? "Runtime settings" : "Quick help"}</h2><button className="icon-button" type="button" aria-label="Close" onClick={onClose}><X size={19} strokeWidth={1.6} /></button></div>
        {panel === "settings" ? (
          <>
            <dl className="runtime-list">
              <div><dt>Backend</dt><dd>Local Flask API</dd></div>
              <div><dt>Attention</dt><dd>{runtime?.attention_backend ?? "Connecting…"}</dd></div>
              <div><dt>Model</dt><dd>{runtime ? `${runtime.model_type} · ${runtime.model_loaded ? "loaded" : "not loaded"}` : "Connecting…"}</dd></div>
              {runtime?.checkpoint ? <div><dt>Checkpoint</dt><dd>{runtime.checkpoint.id}</dd></div> : null}
            </dl>
            <div className="runtime-toggle-block">
              <div className="runtime-toggle-copy"><TestTube2 size={19} strokeWidth={1.6} /><span><strong>Mock generation</strong><small>Return synthetic test images without running HiDream.</small></span></div>
              <label className="toggle-row compact" aria-label="Toggle mock generation">
                <input type="checkbox" checked={runtime?.mock_mode ?? false} disabled={!runtime || runtimeBusy} onChange={(event) => onMockModeChange(event.target.checked)} />
                <span className="toggle" aria-hidden="true" />
              </label>
            </div>
            <p className="runtime-warning"><AlertTriangle size={15} strokeWidth={1.7} />Off by default. Mock images are visibly labelled and are never model output.</p>
          </>
        ) : (
          <div className="help-copy"><p><strong>Create</strong> turns a text description into one to four new images.</p><p><strong>Edit</strong> uses one source image and your change instructions. With subject preservation enabled, simple solid-background requests keep the original colors and detail. Other edits regenerate the image at about 4 MP before resizing to the output size, so unchanged details can vary.</p><p><strong>Refine prompt</strong> calls the configured prompt model; it never adds canned text.</p><p>The first real run can be slower while Triton compiles local GPU kernels.</p></div>
        )}
      </dialog>
  );
}
