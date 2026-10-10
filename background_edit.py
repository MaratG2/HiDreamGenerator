"""Subject-preserving solid-background edits; no diffusion repainting."""

import os
import re
from pathlib import Path

from PIL import Image, ImageColor

MODEL_ID = "ZhengPeng7/BiRefNet"
MODEL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
_MODEL = None

_COLOR = r"(#[0-9a-f]{6}|black|white|red|green|blue|yellow|orange|purple|pink|gray|grey|navy|teal|cyan|magenta)"
_BACKGROUND_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:change|replace|make|set|turn)\s+(?:the\s+)?background\s+"
    r"(?:(?:to|with|into)\s+)?"
    r"(?:(?:a|an|fully|solid|pure|completely|uniform|flat|one[- ]colou?red|single[- ]colou?red)\s+)*"
    + _COLOR + r"(?:\s+background)?\s*[.!]?\s*$", re.IGNORECASE,
)


def background_color_for_prompt(prompt):
    """Recognize only an unambiguous, single-operation background request."""
    match = _BACKGROUND_REQUEST.fullmatch(prompt)
    if not match:
        return None
    return "#%02x%02x%02x" % ImageColor.getrgb(match.group(1))


def model_path():
    root = Path(__file__).resolve().parent
    configured = Path(os.environ.get("HIDREAM_BACKGROUND_MODEL_PATH", "HiDream-ai/BiRefNet")).expanduser()
    return configured if configured.is_absolute() else root / configured


def available():
    root = model_path()
    return (root / "config.json").is_file() and (root / "model.safetensors").is_file()


def checkpoint_metadata():
    return {"id": MODEL_ID, "source_model": MODEL_ID, "source_revision": MODEL_REVISION,
            "quantization": "none", "variant": "foreground-mask"}


def foreground_mask(image):
    """Run a local mask prediction, releasing its GPU memory after the edit."""
    global _MODEL
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    if not available():
        raise RuntimeError("Background mask model is missing. Run scripts/install_background_model.py first.")
    if _MODEL is None:
        _MODEL = AutoModelForImageSegmentation.from_pretrained(
            str(model_path()), trust_remote_code=True, local_files_only=True,
            dtype=torch.float32,
        ).eval().cpu()
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] >= 6 * 1024**3:
            device = "cuda"
    previous_threads = torch.get_num_threads()
    try:
        _MODEL.to(device)
        if device == "cpu":
            torch.set_num_threads(min(previous_threads, 8))
        with torch.inference_mode():
            prediction = _MODEL(transform(image).unsqueeze(0).to(device))[-1].sigmoid()[0, 0].cpu()
    finally:
        _MODEL.cpu()
        torch.set_num_threads(previous_threads)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mask = transforms.ToPILImage()(prediction)
    return mask.resize(image.size, Image.Resampling.LANCZOS)


def composite_background(source, mask, color):
    """Keep confident foreground pixels exact and retain soft silhouette edges."""
    import numpy as np
    pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    values = np.asarray(mask.convert("L"))
    alpha = np.where(values <= 5, 0, np.where(values >= 250, 255, values)).astype(np.float32)[..., None] / 255
    samples = pixels[values <= 5]
    foreground = pixels
    if (len(samples) >= source.width * source.height * 0.05
            and np.max(np.percentile(samples, 95, axis=0) - np.percentile(samples, 5, axis=0)) <= 6):
        # Remove the old flat backdrop's contribution from antialiased edges.
        # Opaque subject pixels remain exactly the source values.
        background = np.median(samples, axis=0)
        foreground = np.clip((pixels - (1 - alpha) * background) / np.maximum(alpha, 1 / 255), 0, 255)
    replacement = np.asarray(ImageColor.getrgb(color), dtype=np.float32)
    result = np.rint(foreground * alpha + replacement * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


def replace_background(source, color, size, callback=None):
    source = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    if callback:
        callback(0, 2)
    mask = foreground_mask(source)
    result = composite_background(source, mask, color)
    if callback:
        callback(1, 2, lambda: result)
    return result
