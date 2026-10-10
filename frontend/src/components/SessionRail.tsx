import { Copy, Download, Images, Trash2 } from "lucide-react";

import type { HistoryItem } from "../types";

interface SessionRailProps {
  items: HistoryItem[];
  disabled: boolean;
  selectedId: string | null;
  onSelect: (item: HistoryItem) => void;
  onDownload: (item: HistoryItem) => void;
  onCopy: (item: HistoryItem) => void;
  onClear: () => void;
}

export function SessionRail({
  items,
  disabled,
  selectedId,
  onSelect,
  onDownload,
  onCopy,
  onClear,
}: SessionRailProps) {
  return (
    <section className="session-rail" aria-labelledby="session-title">
      <div className="session-header">
        <h2 id="session-title">This session</h2>
        <div className="session-tools">
          <Images size={18} strokeWidth={1.5} aria-hidden="true" />
          <button
            className="icon-button"
            type="button"
            aria-label="Clear session images"
            title="Clear session"
            disabled={disabled || !items.length}
            onClick={onClear}
          >
            <Trash2 size={18} strokeWidth={1.5} />
          </button>
        </div>
      </div>
      {items.length ? (
        <div className="history-list">
          {items.map((item) => (
            <article
              key={item.id}
              className={item.id === selectedId ? "history-item is-selected" : "history-item"}
            >
              <button className="history-select" type="button" disabled={disabled} onClick={() => onSelect(item)}>
                <img src={item.imageUrl} alt="Generated result" />
                <span className="history-copy">
                  <strong>{item.mockMode ? "Mock · " : ""}{item.mode === "edit" ? "Edited image" : "Generated image"}</strong>
                  <span>{item.width} × {item.height} · Seed {item.seed}</span><span>{item.prompt}</span>
                </span>
              </button>
              <div className="history-actions">
                <button type="button" aria-label="Copy this image" title="Copy image" onClick={() => onCopy(item)}><Copy size={16} strokeWidth={1.5} /></button>
                <button type="button" aria-label="Download this image" title="Download PNG" onClick={() => onDownload(item)}><Download size={16} strokeWidth={1.5} /></button>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <p className="session-empty">Your generated images will appear here.</p>
      )}
    </section>
  );
}
