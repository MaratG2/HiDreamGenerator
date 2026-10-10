"""Image-only loading and portable checkpoint provenance."""

import json
from pathlib import Path


def checkpoint_metadata(model_path):
    """Report identifiers and precision, never the user's absolute model path."""
    root = Path(model_path)
    config_path = root / "config.json"
    summary_path = root / "quantization_summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    quant = config.get("quantization_config") or {}
    return {
        "id": summary.get("repo_id") or root.name,
        "source_model": summary.get("source_model"),
        "source_revision": summary.get("source_snapshot"),
        "quantization": quant.get("weights_dtype") or config.get("dtype", config.get("torch_dtype", "unknown")),
        "variant": summary.get("variant"),
    }


def image_device_map(config):
    """Keep the vocabulary tables on CPU for image inference.

    The untied input table and output head each occupy about 1.16 GiB in BF16.
    The head is unused; configure_image_memory handles the input lookup on CPU.
    Tied embeddings must stay together.
    """
    if getattr(config, "tie_word_embeddings", False):
        return "cuda"
    return {"": "cuda", "lm_head": "cpu", "model.language_model.embed_tokens": "cpu"}


def configure_image_memory(model):
    """Look up prompt embeddings on CPU; transfer only the selected token rows.

    Accelerate normally copies an offloaded module's entire weights to CUDA
    for each call. A vocabulary lookup needs only a few rows from the 1.16 GiB
    table, so execute it on CPU and return the embeddings to the input device.
    All decoder/image computation and precision are preserved.
    """
    if getattr(model.config, "tie_word_embeddings", False):
        return
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

    embeddings = model.get_input_embeddings()
    remove_hook_from_module(embeddings)
    add_hook_to_module(embeddings, AlignDevicesHook(execution_device="cpu", io_same_device=True))
