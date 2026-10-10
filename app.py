"""Local HiDreamGenerator API, model host and legacy interface.

Features
--------
- Text-to-image, image editing, and multi-reference subject-driven generation
- Optional prompt refinement through an OpenAI-compatible API
- Server-Sent Events for real-time per-step progress
- Serves the bundled React studio, with the embedded legacy UI at /legacy

Run
---
    python app.py --model_path /path/to/HiDream-O1-Image --model_type full
"""

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

load_dotenv(Path(__file__).resolve().parent / ".env")
from generation_settings import aspect_dimensions, number, render_dimensions, resolve_settings, validate_dimensions
from generation_jobs import GenerationJob, MAX_ACTIVE_JOBS, prune_jobs
from model_runtime import checkpoint_metadata, configure_image_memory, image_device_map
import background_edit
from prompt_agent_v2 import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, refine_prompt


# ── Globals ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
_GEN_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_ADMISSION = threading.BoundedSemaphore(MAX_ACTIVE_JOBS)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["TRUSTED_HOSTS"] = os.environ.get("HIDREAM_TRUSTED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")


@app.before_request
def check_api_origin():
    if request.path.startswith("/api/") and request.method == "POST":
        origin = request.headers.get("Origin")
        allowed = {request.host_url.rstrip("/"), "http://127.0.0.1:5173", "http://localhost:5173"}
        allowed.update(filter(None, os.environ.get("HIDREAM_ALLOWED_ORIGINS", "").split(",")))
        if origin and origin not in allowed:
            return jsonify({"error": "Cross-origin API requests are not allowed"}), 403


@app.errorhandler(HTTPException)
def api_http_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": error.description}), error.code
    return error


def request_object():
    data = request.get_json()
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def prompt_from(data):
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > 12000:
        raise ValueError("prompt must be at most 12000 characters")
    return prompt.strip()


def decode_references(refs):
    images = []
    for encoded in refs:
        if not isinstance(encoded, str) or len(encoded) > 32 * 1024 * 1024:
            raise ValueError("Each reference must be a base64 image of at most 24 MB")
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("References must be PNG, JPEG, or WebP images")
                if source.width * source.height > 24_000_000:
                    raise ValueError("Reference images must be at most 24 megapixels")
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("Reference image is not readable") from error
    return images


def generate_image(**kwargs):
    # Model dependencies are unnecessary for mock mode and API contract tests.
    from models.pipeline import generate_image as run_pipeline
    return run_pipeline(**kwargs)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_STATE = {
    "model": None,
    "processor": None,
    "model_type": "full",
    "mock_mode": _env_flag("HIDREAM_MOCK_MODE", False),
}
_JOBS = {}
_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"


def _add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def load_image_model(model_path):
    import torch
    from transformers import AutoConfig, AutoProcessor
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    try:
        import sdnq  # noqa: F401 - registers quantized checkpoints
    except ImportError:
        pass
    torch.set_float32_matmul_precision("high")
    print(f"[app] Loading checkpoint from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    # Preserve BF16 image weights; vocabulary lookup and the unused head stay on CPU.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=image_device_map(config)
    ).eval()
    configure_image_memory(model)
    _STATE["checkpoint"] = checkpoint_metadata(model_path)
    _add_special_tokens(_get_tokenizer(processor))
    return processor, model


# ── HTML ─────────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>HiDream-O1-Image{% if model_type == 'dev' %}-Dev{% endif %}</title>
<style>
  :root {
    --bg: #fbfbfd;
    --panel: #ffffff;
    --panel-2: #f5f5f7;
    --border: rgba(0, 0, 0, 0.08);
    --border-strong: rgba(0, 0, 0, 0.14);
    --text: #1d1d1f;
    --muted: #86868b;
    --accent: #1d1d1f;
    --accent-soft: #f0f0f2;
    --blue: #0071e3;
    --blue-hover: #0077ed;
    --purple: #8a5cf6;
    --pink: #ff6ea1;
    --mint: #22c9a4;
    --gradient: linear-gradient(135deg, #8a5cf6 0%, #0071e3 45%, #22c9a4 100%);
    --gradient-warm: linear-gradient(135deg, #ff6ea1 0%, #8a5cf6 100%);
    --danger: #d70015;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.06);
    --radius: 12px;
    --radius-sm: 8px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
      "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px;
    background: rgba(251, 251, 253, 0.85);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  .brand {
    font-weight: 600; font-size: 17px; letter-spacing: -0.01em;
    background: var(--gradient);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }
  .topbar-meta { color: var(--muted); font-size: 12px; }
  .layout {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 24px;
    padding: 24px 32px 40px;
    max-width: 1600px; margin: 0 auto;
  }
  .sidebar, .canvas {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
  }
  .sidebar {
    padding: 24px; max-height: calc(100vh - 130px);
    overflow-y: auto; align-self: start;
  }
  .canvas {
    min-height: calc(100vh - 130px);
    padding: 28px; display: flex; flex-direction: column;
  }
  h2 {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin: 0 0 12px;
  }
  .group { margin-bottom: 22px; }
  .group:last-of-type { margin-bottom: 0; }
  label {
    display: block; font-size: 12px; font-weight: 500;
    color: var(--text); margin-bottom: 8px;
  }
  input[type=text], input[type=number], input[type=password],
  textarea, select {
    width: 100%;
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
    -webkit-appearance: none;
  }
  input:focus, textarea:focus, select:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15);
  }
  textarea { resize: vertical; min-height: 96px; line-height: 1.5; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .tabs {
    display: flex; gap: 0; margin-bottom: 22px;
    background: var(--panel-2);
    border-radius: 10px; padding: 3px;
  }
  .tab {
    flex: 1; text-align: center; padding: 8px 10px;
    font-size: 12.5px; font-weight: 500;
    color: var(--muted); border-radius: 7px; cursor: pointer;
    user-select: none; transition: all 0.2s;
  }
  .tab.active {
    background: var(--panel); color: var(--text);
    box-shadow: var(--shadow-sm);
  }
  .tab:hover:not(.active) { color: var(--text); }
  details {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 14px; margin-bottom: 16px;
    background: var(--panel-2);
    transition: background 0.15s;
  }
  details summary {
    cursor: pointer; color: var(--text);
    font-size: 12px; font-weight: 500;
    outline: none; list-style: none;
    display: flex; align-items: center; justify-content: space-between;
  }
  details summary::after {
    content: "⌄"; color: var(--muted); font-size: 14px;
    transition: transform 0.2s;
  }
  details[open] summary { margin-bottom: 14px; }
  details[open] summary::after { transform: rotate(180deg); }
  details summary::-webkit-details-marker { display: none; }
  .file-input {
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 16px; text-align: center; color: var(--muted);
    cursor: pointer; transition: all 0.15s;
    font-size: 12.5px;
  }
  .file-input:hover {
    border-color: var(--blue); color: var(--blue);
    background: rgba(0, 113, 227, 0.03);
  }
  .file-input input { display: none; }
  .thumbs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .thumb {
    width: 64px; height: 64px; border-radius: var(--radius-sm);
    overflow: hidden; border: 1px solid var(--border);
    position: relative; box-shadow: var(--shadow-sm);
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb .x {
    position: absolute; top: 4px; right: 4px;
    width: 18px; height: 18px; border-radius: 50%;
    background: rgba(0,0,0,0.65); color: #fff;
    font-size: 11px; line-height: 18px; text-align: center;
    cursor: pointer; backdrop-filter: blur(4px);
  }
  button {
    font-family: inherit; cursor: pointer;
    transition: all 0.15s; -webkit-appearance: none;
  }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-primary {
    width: 100%; background: var(--gradient); color: #fff;
    border: none; border-radius: 980px; padding: 12px 20px;
    font-size: 14px; font-weight: 600;
    letter-spacing: -0.01em;
    box-shadow: 0 4px 14px rgba(138, 92, 246, 0.25);
  }
  .btn-primary:hover:not(:disabled) {
    box-shadow: 0 6px 20px rgba(138, 92, 246, 0.35);
    transform: translateY(-1px);
  }
  .btn-secondary {
    width: 100%; background: var(--panel); color: var(--text);
    border: 1px solid var(--border-strong); border-radius: 980px;
    padding: 11px 20px; font-size: 13px; font-weight: 500;
    margin-bottom: 10px;
  }
  .btn-secondary:hover:not(:disabled) {
    background: var(--panel-2); border-color: var(--text);
  }
  .btn-link {
    background: none; border: none; color: var(--blue);
    padding: 4px 0; font-size: 12px; font-weight: 500;
  }
  .btn-link:hover:not(:disabled) { text-decoration: underline; }
  .canvas-empty {
    flex: 1; display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 14px;
  }
  .canvas-image {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: var(--panel-2);
    border-radius: var(--radius);
    overflow: hidden; padding: 16px; min-height: 400px;
  }
  .canvas-image img {
    max-width: 100%; max-height: calc(100vh - 220px);
    border-radius: var(--radius-sm);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .meta {
    margin-top: 18px; padding: 16px 18px;
    background: var(--panel-2);
    border-radius: var(--radius-sm);
    font-size: 13px; color: var(--text);
  }
  .meta .label {
    color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 10.5px; margin-bottom: 6px;
  }
  .meta pre {
    margin: 0 0 14px; white-space: pre-wrap; word-break: break-word;
    color: var(--text); font-family: inherit; line-height: 1.55;
  }
  .meta pre:last-child { margin-bottom: 0; }
  .progress {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: var(--muted); padding: 24px;
    position: relative;
  }
  .progress-preview {
    width: 100%; max-width: 520px; aspect-ratio: 1;
    border-radius: var(--radius);
    background: var(--panel-2);
    overflow: hidden; position: relative;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08),
                0 0 0 1px var(--border);
  }
  .progress-preview::before {
    content: ""; position: absolute; inset: 0;
    background: var(--gradient);
    opacity: 0.08; pointer-events: none;
    mix-blend-mode: screen;
  }
  .progress-preview img {
    width: 100%; height: 100%; object-fit: cover;
    display: block; transition: filter 0.4s ease;
  }
  .progress-preview.empty { display: flex;
    align-items: center; justify-content: center; }
  .progress-preview.empty::after {
    content: ""; width: 80px; height: 80px;
    border-radius: 50%; background: var(--gradient);
    filter: blur(28px); opacity: 0.85;
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse {
    0%, 100% { transform: scale(0.85); opacity: 0.65; }
    50% { transform: scale(1.2); opacity: 1; }
  }
  .progress-meta {
    display: flex; align-items: center; justify-content: space-between;
    width: 100%; max-width: 520px; margin-top: 20px;
    font-variant-numeric: tabular-nums;
  }
  .progress-label {
    font-size: 13px; font-weight: 600; color: var(--text);
    letter-spacing: -0.01em;
  }
  .progress-step {
    font-size: 12px; color: var(--muted);
    font-feature-settings: "tnum";
  }
  .progress-bar {
    width: 100%; max-width: 520px; height: 3px; border-radius: 999px;
    background: var(--panel-2); overflow: hidden; margin-top: 10px;
  }
  .progress-bar-fill {
    height: 100%; background: var(--gradient);
    border-radius: 999px; width: 0%;
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .refine-preview {
    margin-top: 12px; padding: 12px 14px;
    background: rgba(0, 113, 227, 0.06);
    border: 1px solid rgba(0, 113, 227, 0.2);
    border-radius: var(--radius-sm);
    font-size: 12.5px;
  }
  .refine-preview .label {
    color: var(--blue); font-weight: 600;
    font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .refine-preview pre {
    margin: 0; white-space: pre-wrap; word-break: break-word;
    font-family: inherit; line-height: 1.5;
  }
  .refine-actions { display: flex; gap: 8px; margin-top: 10px; }
  .refine-actions button { flex: 1; padding: 7px 10px; font-size: 12px; }
  .err {
    color: var(--danger); margin-top: 12px; font-size: 12.5px;
    padding: 8px 12px; background: rgba(215, 0, 21, 0.06);
    border-radius: var(--radius-sm);
  }
  .toggle-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0; margin-bottom: 12px;
  }
  .toggle-row label { margin: 0; font-weight: 500; }
  .toggle-row .hint { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .divider {
    height: 1px; background: var(--border);
    margin: 22px 0;
  }
  .spinner-inline {
    width: 14px; height: 14px; display: inline-block;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: -2px; margin-right: 8px;
  }
  .spinner-blue {
    border: 2px solid rgba(0, 113, 227, 0.2);
    border-top-color: var(--blue);
  }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { max-height: none; }
  }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <span class="brand">HiDream-O1-Image{% if model_type == 'dev' %}-Dev{% endif %}</span>
    </div>
    <span class="topbar-meta"></span>
  </div>

  <div class="layout">
    <aside class="sidebar">
      <div class="tabs" id="tabs">
        <div class="tab active" data-mode="t2i">Text → Image</div>
        <div class="tab" data-mode="edit">Edit</div>
        <div class="tab" data-mode="subject">Subject</div>
      </div>

      <div class="group">
        <label>Prompt</label>
        <textarea id="prompt" placeholder="Describe the image you want to create..."></textarea>
      </div>

      <div class="group" id="refs-group" style="display:none">
        <label id="refs-label">Reference image</label>
        <label class="file-input">
          <input id="refs" type="file" accept="image/*" multiple />
          <span>Click to upload images</span>
        </label>
        <div class="thumbs" id="thumbs"></div>
        <label id="keep-aspect-row" style="display:none; margin-top:10px; font-weight:400; text-transform:none; letter-spacing:0; cursor:pointer;">
          <input id="keep-aspect" type="checkbox" style="vertical-align:middle; margin-right:6px;" />
          Keep reference aspect (use selected long edge)
        </label>
        <div id="edit-scheduler-row" style="display:none; margin-top:12px;">
          <label>Scheduler</label>
          <select id="edit-scheduler">
            <option value="flow_match" selected>flow_match (default)</option>
            <option value="flash">flash</option>
          </select>
        </div>
      </div>

      <details id="refine-section">
        <summary>Prompt Refiner</summary>
        <div style="font-size:11.5px; color:var(--muted); margin-bottom:10px">
          Calls an OpenAI-compatible endpoint serving <code>HiDream-ai/gemma-4-31B-it-Prompt-Refine</code>.
          Start the server first with <code>bash start_vllm_server.sh</code>.
        </div>
        <label>Base URL</label>
        <input id="api-base" type="text" autocomplete="off" name="hd-base-url"
               placeholder="http://localhost:8000/v1" value="{{ env_base_url }}" />
        <label style="margin-top:10px">Model</label>
        <input id="api-model" type="text" autocomplete="off" name="hd-model"
               placeholder="HiDream-ai/gemma-4-31B-it-Prompt-Refine" value="{{ env_model }}" />
        <button class="btn-secondary" id="refine-btn" style="margin-top: 14px; margin-bottom: 0">
          Refine Prompt
        </button>
        <div id="refine-preview" class="refine-preview" style="display:none"></div>
      </details>

      <details>
        <summary>Generation Settings</summary>
        <div class="row" style="margin-bottom: 12px">
          <div>
            <label>Width</label>
            <input id="width" type="number" value="2048" step="64" min="512" />
          </div>
          <div>
            <label>Height</label>
            <input id="height" type="number" value="2048" step="64" min="512" />
          </div>
        </div>
        <label>Seed</label>
        <input id="seed" type="number" value="32" />
      </details>

      <button class="btn-primary" id="go">Generate</button>
      <div class="err" id="err" style="display:none"></div>
    </aside>

    <main class="canvas">
      <div class="canvas-empty" id="empty">
        <div style="text-align: center">
          <div style="font-size: 32px; opacity: 0.3; margin-bottom: 8px">◍</div>
          <div>Your generated image will appear here</div>
        </div>
      </div>
      <div class="progress" id="progress" style="display:none">
        <div class="progress-preview empty" id="progress-preview">
          <img id="progress-img" style="display:none" />
        </div>
        <div class="progress-meta">
          <span class="progress-label" id="progress-text">Preparing</span>
          <span class="progress-step" id="progress-sub">—</span>
        </div>
        <div class="progress-bar"><div class="progress-bar-fill" id="progress-fill"></div></div>
      </div>
      <div class="canvas-image" id="out" style="display:none">
        <img id="img" />
      </div>
      <div class="meta" id="meta" style="display:none"></div>
    </main>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const MODEL_TYPE = "{{ model_type }}";
let mode = "t2i";
let refFiles = [];
let lastRefined = null;
let originalPrompt = null;


document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    mode = t.dataset.mode;
    const refsGroup = $("refs-group");
    const refineSection = $("refine-section");
    const keepAspectRow = $("keep-aspect-row");
    const editSchedRow = $("edit-scheduler-row");
    if (mode === "t2i") {
      refsGroup.style.display = "none";
      refFiles = []; renderThumbs();
      refineSection.style.display = "";
      keepAspectRow.style.display = "none";
      $("keep-aspect").checked = false;
      editSchedRow.style.display = "none";
    } else {
      refsGroup.style.display = "";
      $("refs-label").textContent = mode === "edit"
        ? "Source image (1)"
        : "Reference images (2+)";
      $("refs").multiple = mode !== "edit";
      // Refine only available for T2I per design spec.
      refineSection.style.display = "none";
      refineSection.removeAttribute("open");
      clearRefinePreview();
      // `keep_original_aspect` only applies when there is exactly one ref image.
      keepAspectRow.style.display = mode === "edit" ? "" : "none";
      if (mode !== "edit") $("keep-aspect").checked = false;
      // Editing scheduler selector only applies to the Dev model + Edit tab.
      editSchedRow.style.display = (mode === "edit" && MODEL_TYPE === "dev") ? "" : "none";
    }
  };
});

$("refs").onchange = (e) => {
  const files = Array.from(e.target.files);
  refFiles = mode === "edit" ? files.slice(0, 1) : refFiles.concat(files);
  renderThumbs();
  e.target.value = "";
};

function renderThumbs() {
  const c = $("thumbs"); c.innerHTML = "";
  refFiles.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const el = document.createElement("div");
    el.className = "thumb";
    el.innerHTML = `<img src="${url}" /><div class="x" data-i="${i}">×</div>`;
    el.querySelector(".x").onclick = () => {
      refFiles.splice(i, 1); renderThumbs();
    };
    c.appendChild(el);
  });
}

function fileToB64(f) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(f);
  });
}

function showErr(msg) {
  const e = $("err"); e.textContent = msg; e.style.display = msg ? "" : "none";
}

function clearRefinePreview() {
  lastRefined = null; originalPrompt = null;
  $("refine-preview").style.display = "none";
  $("refine-preview").innerHTML = "";
}

function renderRefinePreview(refined) {
  lastRefined = refined;
  const html = `
    <div class="label">
      <span>Refined Prompt</span>
      <span style="font-weight: 400; color: var(--muted); text-transform: none; letter-spacing: 0">Preview</span>
    </div>
    <pre>${escapeHtml(refined.prompt)}</pre>
    <div class="refine-actions">
      <button class="btn-secondary" id="refine-apply" style="margin: 0">Use This Prompt</button>
      <button class="btn-secondary" id="refine-discard" style="margin: 0">Discard</button>
    </div>
  `;
  const box = $("refine-preview");
  box.innerHTML = html; box.style.display = "";
  $("refine-apply").onclick = () => {
    $("prompt").value = refined.prompt;
    clearRefinePreview();
  };
  $("refine-discard").onclick = () => clearRefinePreview();
}

$("refine-btn").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showErr("Please enter a prompt to refine."); return; }
  showErr("");
  const btn = $("refine-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline spinner-blue"></span>Refining…';
  try {
    const body = {
      prompt,
      base_url: $("api-base").value.trim(),
      model: $("api-model").value.trim(),
    };
    const r = await fetch("/api/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Refine failed");
    renderRefinePreview(data);
  } catch (e) {
    showErr(e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Refine Prompt";
  }
};

$("go").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showErr("Please enter a prompt."); return; }
  if (mode === "edit" && refFiles.length !== 1) {
    showErr("Edit mode requires exactly one source image."); return;
  }
  if (mode === "subject" && refFiles.length < 2) {
    showErr("Subject mode requires at least two reference images."); return;
  }
  showErr("");
  const btn = $("go"); btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline"></span>Generating…';
  $("empty").style.display = "none";
  $("out").style.display = "none";
  $("meta").style.display = "none";
  $("progress").style.display = "";
  $("progress-text").textContent = "Preparing";
  $("progress-sub").textContent = "Encoding inputs";
  $("progress-fill").style.width = "0%";
  $("progress-img").style.display = "none";
  $("progress-img").removeAttribute("src");
  $("progress-preview").classList.add("empty");

  try {
    const refs_b64 = await Promise.all(refFiles.map(fileToB64));
    const keepAspect = mode === "edit" && $("keep-aspect").checked && refFiles.length === 1;
    const editingScheduler = (mode === "edit" && MODEL_TYPE === "dev")
      ? $("edit-scheduler").value : null;
    const startResp = await fetch("/api/generate/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode, prompt,
        width: parseInt($("width").value),
        height: parseInt($("height").value),
        seed: parseInt($("seed").value),
        refs_b64,
        keep_original_aspect: keepAspect,
        editing_scheduler: editingScheduler,
      }),
    });
    const startData = await startResp.json();
    if (!startResp.ok) throw new Error(startData.error || "Failed to start");
    const jobId = startData.job_id;

    await new Promise((resolve, reject) => {
      const es = new EventSource(`/api/generate/stream/${jobId}`);
      es.onmessage = (ev) => {
        const d = JSON.parse(ev.data);
        if (d.type === "progress") {
          const pct = Math.round((d.step / d.total) * 100);
          $("progress-text").textContent = `Generating · ${pct}%`;
          $("progress-sub").textContent = `Step ${d.step} / ${d.total}`;
          $("progress-fill").style.width = pct + "%";
          if (d.preview) {
            $("progress-img").src = "data:image/jpeg;base64," + d.preview;
            $("progress-img").style.display = "";
            $("progress-preview").classList.remove("empty");
          }
        } else if (d.type === "done") {
          $("img").src = "data:image/png;base64," + d.image;
          $("progress").style.display = "none";
          $("out").style.display = "";
          es.close(); resolve();
        } else if (d.type === "error") {
          es.close(); reject(new Error(d.message));
        }
      };
      es.onerror = () => { es.close(); reject(new Error("Stream connection lost")); };
    });
  } catch (e) {
    showErr(e.message);
    $("progress").style.display = "none";
    $("empty").style.display = "";
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate";
  }
};

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[c]);
}
</script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    if (_FRONTEND_DIST / "index.html").is_file():
        return send_from_directory(_FRONTEND_DIST, "index.html")
    return legacy_index()


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(_FRONTEND_DIST / "assets", filename)


@app.route("/legacy")
def legacy_index():
    return render_template_string(
        INDEX_HTML,
        model_type=_STATE["model_type"],
        env_base_url=os.environ.get("HIDREAM_REFINE_BASE_URL", DEFAULT_BASE_URL),
        env_model=os.environ.get("HIDREAM_REFINE_MODEL", DEFAULT_MODEL_ID),
    )


@app.route("/api/status")
def api_status():
    with _JOBS_LOCK:
        prune_jobs(_JOBS)
    return jsonify({
        "status": "ready" if _STATE["model"] is not None or _STATE["mock_mode"] else "starting",
        "model_type": _STATE["model_type"],
        "mock_mode": _STATE["mock_mode"],
        "checkpoint": _STATE.get("checkpoint"),
        "device": _STATE.get("device", "CPU"),
        "attention_backend": os.environ.get("HIDREAM_ATTENTION_BACKEND", "auto"),
        "supported_modes": ["t2i", "edit", "subject"],
        "supported_schedulers": ["default", "flow_match", "flash"],
        "background_edit_available": background_edit.available(),
        "max_concurrent_generations": 1,
    })


@app.route("/api/runtime", methods=["GET", "POST"])
def api_runtime():
    if request.method == "POST":
        try:
            data = request_object()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        if "mock_mode" not in data or not isinstance(data["mock_mode"], bool):
            return jsonify({"error": "mock_mode must be a boolean"}), 400
        with _JOBS_LOCK:
            if any(job.finished_at is None for job in _JOBS.values()):
                return jsonify({"error": "Wait for active generations before changing runtime mode"}), 409
            _STATE["mock_mode"] = data["mock_mode"]
    return jsonify({
        "mock_mode": _STATE["mock_mode"],
        "model_loaded": _STATE["model"] is not None,
        "background_edit_available": background_edit.available(),
        "model_type": _STATE["model_type"],
        "checkpoint": _STATE.get("checkpoint"),
        "attention_backend": os.environ.get("HIDREAM_ATTENTION_BACKEND", "auto"),
    })


@app.route("/api/refine", methods=["POST"])
def api_refine():
    try:
        data = request_object()
        prompt = prompt_from(data)
        base_url = os.environ.get("HIDREAM_REFINE_BASE_URL", DEFAULT_BASE_URL).strip()
        model_id = os.environ.get("HIDREAM_REFINE_MODEL", DEFAULT_MODEL_ID).strip()
        if data.get("base_url") and data["base_url"] != base_url:
            raise ValueError("Configure the refiner endpoint using HIDREAM_REFINE_BASE_URL on the server")
        if data.get("model"):
            if not isinstance(data["model"], str):
                raise ValueError("model must be a string")
            model_id = data["model"].strip()
        if urlsplit(base_url).scheme not in {"http", "https"}:
            raise ValueError("The configured refiner URL must use HTTP or HTTPS")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        refined = refine_prompt(
            prompt,
            model_id=model_id,
            base_url=base_url,
            max_attempts=1,
            api_max_attempts=1,
            api_retry_delay=0,
            api_key=os.environ.get("HIDREAM_REFINE_API_KEY", "EMPTY"),
            request_timeout=float(os.environ.get("HIDREAM_REFINE_TIMEOUT", "120")),
            raise_on_failure=True,
        )
        return jsonify({"prompt": refined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    try:
        data = request_object()
        allowed = {"prompt", "mode", "width", "height", "seed", "refs_b64", "keep_original_aspect",
                   "editing_scheduler", "scheduler_name", "num_inference_steps", "guidance_scale",
                   "shift", "noise_scale_start", "noise_scale_end", "noise_clip_std", "preserve_background_subject"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown generation settings: {', '.join(sorted(unknown))}")
        prompt = prompt_from(data)
        mode = data.get("mode", "t2i")
        if mode not in ("t2i", "edit", "subject"):
            raise ValueError("mode must be t2i, edit, or subject")
        width, height = validate_dimensions(data.get("width", 1024), data.get("height", 1024))
        seed = number(data.get("seed", 32), "seed", 0, 4_294_967_295, integer=True)
        refs_b64 = data.get("refs_b64", [])
        if not isinstance(refs_b64, list) or len(refs_b64) > 8:
            raise ValueError("refs_b64 must be a list of up to eight images")
        if mode == "t2i" and refs_b64:
            raise ValueError("Text-to-image mode cannot contain reference images")
        if mode == "edit" and len(refs_b64) != 1:
            raise ValueError("Edit mode requires exactly one reference image")
        if mode == "subject" and len(refs_b64) < 2:
            raise ValueError("Subject mode requires at least two reference images")
        keep_original_aspect = data.get("keep_original_aspect", False)
        if not isinstance(keep_original_aspect, bool):
            raise ValueError("keep_original_aspect must be a boolean")
        if keep_original_aspect and mode != "edit":
            raise ValueError("keep_original_aspect is only available in edit mode")
        preserve_subject = data.get("preserve_background_subject", True)
        if not isinstance(preserve_subject, bool):
            raise ValueError("preserve_background_subject must be a boolean")
        generation_kwargs = resolve_settings(_STATE["model_type"], mode, data)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if not _ADMISSION.acquire(blocking=False):
        return jsonify({"error": "Generation queue is full; wait for a job to finish"}), 429
    try:
        refs = decode_references(refs_b64)
        if keep_original_aspect:
            width, height = aspect_dimensions(*refs[0].size, max(width, height))
    except (ValueError, OSError) as error:
        _ADMISSION.release()
        return jsonify({"error": str(error)}), 400
    except Exception:
        _ADMISSION.release()
        raise
    job_id = uuid.uuid4().hex
    q = GenerationJob()
    with _JOBS_LOCK:
        mock_mode = _STATE["mock_mode"]
        background_color = (background_edit.background_color_for_prompt(prompt)
                            if not mock_mode and mode == "edit" and preserve_subject and background_edit.available() else None)
        if not mock_mode and not background_color and (_STATE["model"] is None or _STATE["processor"] is None):
            _ADMISSION.release()
            return jsonify({"error": "The HiDream model is not loaded; start the server with a checkpoint or enable mock mode"}), 503
        prune_jobs(_JOBS)
        _JOBS[job_id] = q
    render_width, render_height = (width, height) if mock_mode or background_color else render_dimensions(width, height, bool(refs))
    metadata = {
        "width": width, "height": height, "seed": seed, "mode": mode,
        "render_width": render_width, "render_height": render_height,
        "mock_mode": mock_mode, "model_type": _STATE["model_type"],
        "checkpoint": None if mock_mode else _STATE.get("checkpoint"),
        "parameters": generation_kwargs,
        "engine": "mock" if mock_mode else "hidream",
    }
    if background_color:
        metadata.update(engine="background_composite", model_type="segmentation",
                        checkpoint=background_edit.checkpoint_metadata(),
                        parameters={"background_color": background_color, "preserve_subject": True})

    def worker():
        tmp_paths = []
        try:
            for reference in refs:
                with tempfile.NamedTemporaryFile(suffix=".png", prefix="hidream_", delete=False) as file:
                    tmp_paths.append(file.name)
                    reference.save(file, format="PNG")

            def cb(step, total, get_preview=None):
                msg = {"type": "progress", "step": step + 1, "total": total}
                # Only send a preview image at the 1/4, 1/2, and 3/4 milestones
                # to avoid flooding the SSE stream and keep UI progress in sync.
                milestones = {total // 4, total // 2, (3 * total) // 4}
                want_preview = (
                    get_preview is not None
                    and (step + 1) in milestones
                )
                if want_preview:
                    try:
                        img = get_preview()
                        # Downscale to keep payload tiny — full image is sent at the end.
                        max_side = 384
                        w, h = img.size
                        if max(w, h) > max_side:
                            scale = max_side / max(w, h)
                            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        buf2 = io.BytesIO()
                        img.save(buf2, format="JPEG", quality=72, optimize=True)
                        msg["preview"] = base64.b64encode(buf2.getvalue()).decode("ascii")
                    except Exception:
                        pass
                q.put(msg)

            if mock_mode:
                mock_step_delay = max(0.0, float(os.environ.get("HIDREAM_MOCK_STEP_DELAY", "0")))
                q.put({"type": "progress", "step": 1, "total": 2})
                if mock_step_delay:
                    time.sleep(mock_step_delay)
                image = Image.new("RGB", (width, height), "#e8e3d8")
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()
                gap = max(48, min(width, height) // 12)
                for x in range(0, width, gap):
                    draw.line((x, 0, x, height), fill="#c9c1b3", width=max(1, width // 1024))
                for y in range(0, height, gap):
                    draw.line((0, y, width, y), fill="#c9c1b3", width=max(1, height // 1024))
                inset = max(28, min(width, height) // 18)
                draw.rounded_rectangle(
                    (inset, inset, width - inset, height - inset),
                    radius=max(18, inset // 2), fill="#f7f3eb", outline="#1757ff",
                    width=max(4, min(width, height) // 160),
                )
                lines = [
                    "MOCK MODE — NOT MODEL OUTPUT",
                    f"Mode: {mode}   Seed: {seed}",
                    f"Size: {width} x {height}   Steps: {generation_kwargs['num_inference_steps']}",
                    "",
                    prompt[:240],
                ]
                y = inset * 2
                for line in lines:
                    draw.text((inset * 2, y), line, fill="#171714", font=font)
                    y += max(24, min(width, height) // 28)
                q.put({"type": "progress", "step": 2, "total": 2})
                if mock_step_delay:
                    time.sleep(mock_step_delay)
            elif background_color:
                with _GEN_LOCK:
                    image = background_edit.replace_background(refs[0], background_color, (width, height), cb)
            else:
                if _STATE["model"] is None or _STATE["processor"] is None:
                    raise RuntimeError(
                        "The HiDream model is not loaded. Start the image server normally, "
                        "or explicitly enable mock mode in Runtime settings for UI testing."
                    )
                with _GEN_LOCK:
                    image = generate_image(
                        model=_STATE["model"],
                        processor=_STATE["processor"],
                        prompt=prompt,
                        ref_image_paths=tmp_paths if tmp_paths else None,
                        height=height,
                        width=width,
                        seed=seed,
                        keep_original_aspect=keep_original_aspect,
                        callback=cb,
                        **generation_kwargs,
                    )
            buf = io.BytesIO()
            if image.size != (width, height):
                raise RuntimeError(f"Model returned {image.width}x{image.height}; expected {width}x{height}")
            from PIL.PngImagePlugin import PngInfo
            png_info = PngInfo()
            png_info.add_text("hidream", json.dumps({**metadata, "prompt": prompt}))
            image.save(buf, format="PNG", pnginfo=png_info)
            q.put({"type": "done", "image": base64.b64encode(buf.getvalue()).decode("ascii"), **metadata})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            for p in tmp_paths:
                try: os.remove(p)
                except OSError: pass
            _ADMISSION.release()

    # Give Flask enough time to flush the small start response before the model
    # thread begins CPU-heavy prompt preparation. Without this handoff, the
    # worker can hold the GIL and make POST /generate/start appear to hang until
    # the first inference step.
    starter = threading.Timer(0.05, worker)
    starter.daemon = True
    try:
        starter.start()
    except RuntimeError:
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)
        _ADMISSION.release()
        return jsonify({"error": "Could not start a generation worker"}), 503
    return jsonify({"job_id": job_id, **metadata})


@app.route("/api/generate/stream/<job_id>")
def api_generate_stream(job_id):
    with _JOBS_LOCK:
        prune_jobs(_JOBS)
        q = _JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown or expired job"}), 404
    try:
        cursor = max(0, int(request.headers.get("Last-Event-ID", "0")))
    except ValueError:
        return jsonify({"error": "Last-Event-ID must be an integer"}), 400
    with q.condition:
        if cursor > len(q.events):
            return jsonify({"error": "Last-Event-ID exceeds the job event count"}), 400

    def gen():
        for event_id, item in q.stream(cursor):
            if item is None:
                yield ": keep-alive\n\n"
            else:
                yield f"id: {event_id}\ndata: {json.dumps(item)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser("HiDream-O1-Image Flask app")
    p.add_argument("--model_path", type=str,
                   default=os.environ.get("HIDREAM_MODEL_PATH"),
                   help="Path to HiDream-O1-Image checkpoint directory. "
                        "Defaults to $HIDREAM_MODEL_PATH from .env.")
    p.add_argument("--model_type", type=str,
                   default=os.environ.get("HIDREAM_MODEL_TYPE", "full"),
                   choices=["full", "dev"])
    p.add_argument("--host", type=str,
                   default=os.environ.get("HIDREAM_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("HIDREAM_PORT", "7860")))
    p.add_argument("--mock", action="store_true",
                   help="Start without loading the model and return clearly labelled test images. "
                        "Mock mode is disabled by default.")
    args = p.parse_args()

    if args.mock:
        _STATE["mock_mode"] = True
        _STATE["model_type"] = args.model_type
    elif not args.model_path:
        p.error("--model_path is required (or set HIDREAM_MODEL_PATH in .env)")
    else:
        import torch
        if not torch.cuda.is_available():
            p.error("CUDA-enabled PyTorch and a compatible GPU are required for inference")
        _STATE["device"] = torch.cuda.get_device_name(0)
        processor, model = load_image_model(args.model_path)
        _STATE["processor"] = processor
        _STATE["model"] = model
        _STATE["model_type"] = args.model_type

    print(f"[app] Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
