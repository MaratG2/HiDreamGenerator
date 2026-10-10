import { CircleHelp, Settings2, TestTube2 } from "lucide-react";

import type { RuntimeSettings } from "../types";

interface AppHeaderProps {
  runtime: RuntimeSettings | null;
  onOpenInfo: (panel: "settings" | "help") => void;
}

export function AppHeader({ runtime, onOpenInfo }: AppHeaderProps) {
  const label = runtime?.mock_mode
    ? "Mock mode"
    : runtime
      ? `Local · ${runtime.model_type} · ${runtime.model_loaded ? "ready" : "loading"}`
      : "Local backend";
  return (
    <header className="topbar">
      <div className="brand-lockup" aria-label="HiDreamGenerator"><span className="brand-mark" aria-hidden="true">H</span><span className="brand-name">HiDreamGenerator</span></div>
      <div className="topbar-tools">
        <span className={runtime?.mock_mode ? "local-status is-mock" : "local-status"}>
          {runtime?.mock_mode ? <TestTube2 size={15} strokeWidth={1.7} /> : <span className="status-dot" aria-hidden="true" />}
          <span>{label}</span>
        </span>
        <span className="tool-divider" aria-hidden="true" />
        <button className="icon-button" type="button" title="Runtime settings" aria-label="Open runtime settings" onClick={() => onOpenInfo("settings")}><Settings2 size={20} strokeWidth={1.6} /></button>
        <button className="icon-button" type="button" title="Quick help" aria-label="Open quick help" onClick={() => onOpenInfo("help")}><CircleHelp size={20} strokeWidth={1.6} /></button>
      </div>
    </header>
  );
}
