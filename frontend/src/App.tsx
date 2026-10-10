import { useEffect, useMemo, useState } from "react";

import { getRuntimeSettings, refinePrompt, updateRuntimeSettings } from "./api/hidream";
import { ASPECTS, dimensionsFor, sourceDimensions, QUALITIES, DEV_QUALITIES, RESOLUTIONS } from "./config";
import { AppHeader } from "./components/AppHeader";
import { ControlPanel } from "./components/ControlPanel";
import { InfoDialog } from "./components/InfoDialog";
import { SessionRail } from "./components/SessionRail";
import { Workspace } from "./components/Workspace";
import { useGeneration } from "./hooks/useGeneration";
import { copyDataUrl, createSourceImage, downloadDataUrl } from "./lib/files";
import type {
  AspectPreset,
  EditMethod,
  HistoryItem,
  QualityPreset,
  ResolutionPreset,
  RuntimeSettings,
  SourceImage,
  StudioMode,
} from "./types";

export default function App() {
  const [mode, setMode] = useState<StudioMode>("t2i");
  const [prompts, setPrompts] = useState<Record<StudioMode, string>>({ t2i: "", edit: "" });
  const [sourceImage, setSourceImage] = useState<SourceImage | null>(null);
  const [aspect, setAspect] = useState<AspectPreset>(ASPECTS[0]);
  const [resolution, setResolution] = useState<ResolutionPreset>(RESOLUTIONS[0]);
  const [quality, setQuality] = useState<QualityPreset>(QUALITIES[3]);
  const [batchSize, setBatchSize] = useState(1);
  const [seed, setSeed] = useState(32);
  const [keepOriginalAspect, setKeepOriginalAspect] = useState(true);
  const [editMethod, setEditMethod] = useState<EditMethod>("flow_match");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isRefining, setIsRefining] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [dialog, setDialog] = useState<"settings" | "help" | null>(null);
  const [runtime, setRuntime] = useState<RuntimeSettings | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const { state: generation, generate, reset } = useGeneration();

  useEffect(() => {
    getRuntimeSettings().then((settings) => {
      setRuntime(settings);
      setQuality(settings.model_type === "dev" ? DEV_QUALITIES[2] : QUALITIES[3]);
    }).catch((error) => {
      setLocalError(error instanceof Error ? error.message : "Could not read runtime settings.");
    });
  }, []);

  const prompt = prompts[mode];
  const isBusy = generation.phase === "starting" || generation.phase === "generating";
  const selectedItem = useMemo(
    () => history.find((item) => item.id === selectedId) ?? null,
    [history, selectedId],
  );
  const dimensions = mode === "edit" && keepOriginalAspect && sourceImage
    ? sourceDimensions(sourceImage.width, sourceImage.height, resolution.longSide)
    : dimensionsFor(aspect, resolution);
  const interactionBusy = isBusy || isRefining || isUploading || runtimeBusy;
  const canGenerate = !!runtime && (runtime.model_loaded || runtime.mock_mode);

  const changeMode = (nextMode: StudioMode) => {
    if (interactionBusy) return;
    setMode(nextMode);
    setSelectedId(null);
    setLocalError(null);
    reset();
  };

  const updatePrompt = (value: string) => {
    setPrompts((current) => ({ ...current, [mode]: value }));
    setLocalError(null);
  };

  const handleSourceFile = async (file: File) => {
    if (interactionBusy) return;
    setIsUploading(true);
    setLocalError(null);
    try {
      setSourceImage(await createSourceImage(file));
      setSelectedId(null);
      reset();
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "The image could not be loaded.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleRefine = async () => {
    if (!prompt.trim() || isRefining) return;
    setIsRefining(true);
    setLocalError(null);
    try {
      const refined = await refinePrompt(prompt.trim());
      setPrompts((current) => ({ ...current, [mode]: refined }));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Prompt refinement is unavailable.");
    } finally {
      setIsRefining(false);
    }
  };

  const handleGenerate = async () => {
    if (!canGenerate || !prompt.trim() || interactionBusy || (mode === "edit" && !sourceImage)) return;
    setLocalError(null);
    setNotice(null);
    if (!Number.isInteger(seed) || seed < 0 || seed + batchSize - 1 > 4294967295) {
      setLocalError("Seed must be a whole number from 0 to 4294967295, with room for the batch.");
      return;
    }
    setSelectedId(null);
    const batchId = crypto.randomUUID();
    const requestSnapshot = {
      mode,
      prompt: prompt.trim(),
      width: keepOriginalAspect && mode === "edit" ? resolution.longSide : dimensions.width,
      height: keepOriginalAspect && mode === "edit" ? resolution.longSide : dimensions.height,
      seed,
      refs_b64: sourceImage && mode === "edit" ? [sourceImage.base64] : [],
      keep_original_aspect: mode === "edit" && keepOriginalAspect,
      editing_scheduler: runtime?.model_type === "dev" && mode === "edit" ? editMethod : undefined,
      num_inference_steps: quality.steps,
    };
    await generate(requestSnapshot, batchSize, (result, index) => {
      const item: HistoryItem = {
        id: crypto.randomUUID(),
        batchId,
        batchIndex: index,
        imageUrl: result.imageUrl,
        prompt: requestSnapshot.prompt,
        mode: requestSnapshot.mode,
        seed: result.seed,
        width: result.width,
        height: result.height,
        mockMode: result.mock_mode,
        createdAt: Date.now(),
      };
      setHistory((current) => [item, ...current].slice(0, 40));
      setSelectedId(item.id);
    });
  };

  const handleCopy = async (imageUrl: string) => {
    setLocalError(null);
    try {
      await copyDataUrl(imageUrl);
      setNotice("Image copied to clipboard.");
      window.setTimeout(() => setNotice(null), 1800);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Could not copy the image.");
    }
  };

  const handleMockMode = async (enabled: boolean) => {
    if (interactionBusy) return;
    setRuntimeBusy(true);
    setLocalError(null);
    try {
      setRuntime(await updateRuntimeSettings(enabled));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Could not update runtime settings.");
    } finally {
      setRuntimeBusy(false);
    }
  };

  const download = (url: string, imageSeed = selectedItem?.seed ?? seed, imageMode = selectedItem?.mode ?? mode) => {
    downloadDataUrl(url, `hidream-${imageMode}-${imageSeed}.png`);
  };

  return (
    <div className="studio-app">
      <AppHeader runtime={runtime} onOpenInfo={setDialog} />
      <div className="studio-shell">
        <ControlPanel
          mode={mode}
          prompt={prompt}
          sourceImage={sourceImage}
          aspect={aspect}
          aspects={ASPECTS}
          resolution={resolution}
          resolutions={RESOLUTIONS}
          quality={quality}
          qualities={runtime?.model_type === "dev" ? DEV_QUALITIES : QUALITIES}
          dimensions={dimensions}
          batchSize={batchSize}
          seed={seed}
          keepOriginalAspect={keepOriginalAspect}
          editMethod={editMethod}
          isBusy={interactionBusy}
          isGenerating={isBusy}
          canGenerate={canGenerate}
          modelType={runtime?.model_type ?? "full"}
          isRefining={isRefining}
          mockMode={runtime?.mock_mode ?? false}
          error={localError || generation.error}
          onModeChange={changeMode}
          onPromptChange={updatePrompt}
          onSourceFile={handleSourceFile}
          onRemoveSource={() => { setSourceImage(null); setSelectedId(null); reset(); }}
          onAspectChange={setAspect}
          onResolutionChange={setResolution}
          onQualityChange={setQuality}
          onBatchSizeChange={setBatchSize}
          onSeedChange={setSeed}
          onKeepOriginalAspectChange={setKeepOriginalAspect}
          onEditMethodChange={setEditMethod}
          onRefine={handleRefine}
          onGenerate={handleGenerate}
        />
        <div className="workspace-column">
          <Workspace
            mode={mode}
            sourceImage={sourceImage}
            generation={generation}
            selectedImageUrl={selectedItem?.imageUrl ?? null}
            mockMode={generation.items.length ? !!generation.items[0]?.mockMode || (isBusy && !!runtime?.mock_mode) : selectedItem?.mockMode ?? false}
            notice={notice}
            onDownload={download}
            onCopy={handleCopy}
          />
          <SessionRail
            items={history}
            disabled={interactionBusy}
            selectedId={selectedId}
            onSelect={(item) => { setSelectedId(item.id); reset(); }}
            onDownload={(item) => download(item.imageUrl, item.seed, item.mode)}
            onCopy={(item) => handleCopy(item.imageUrl)}
            onClear={() => { setHistory([]); setSelectedId(null); reset(); }}
          />
        </div>
      </div>
      <InfoDialog
        panel={dialog}
        runtime={runtime}
        runtimeBusy={interactionBusy}
        onMockModeChange={handleMockMode}
        onClose={() => setDialog(null)}
      />
    </div>
  );
}
