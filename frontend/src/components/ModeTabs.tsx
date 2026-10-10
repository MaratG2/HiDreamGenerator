import { PencilLine, Sparkles } from "lucide-react";

import type { StudioMode } from "../types";

interface ModeTabsProps {
  mode: StudioMode;
  disabled?: boolean;
  onChange: (mode: StudioMode) => void;
}

export function ModeTabs({ mode, disabled = false, onChange }: ModeTabsProps) {
  return (
    <div className="mode-tabs" role="tablist" aria-label="Generation mode">
      <button
        className={mode === "t2i" ? "mode-tab is-active" : "mode-tab"}
        type="button"
        role="tab"
        disabled={disabled}
        aria-selected={mode === "t2i"}
        onClick={() => onChange("t2i")}
      >
        <Sparkles size={19} strokeWidth={1.6} />
        Create
      </button>
      <button
        className={mode === "edit" ? "mode-tab is-active" : "mode-tab"}
        type="button"
        role="tab"
        disabled={disabled}
        aria-selected={mode === "edit"}
        onClick={() => onChange("edit")}
      >
        <PencilLine size={19} strokeWidth={1.6} />
        Edit
      </button>
    </div>
  );
}
