"""Image-only loading and portable checkpoint provenance."""

import json
import os
from pathlib import Path


def checkpoint_metadata(model_path):
    """Report identifiers and precision, never the user's absolute model path."""
    root = Path(model_path)
    config_path = root / "config.json"
    summary_path = root / "quantization_summary.json"
    provenance_path = root / "model_provenance.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    quant = config.get("quantization_config") or {}
    return {
        "id": summary.get("repo_id") or provenance.get("id") or root.name,
        "source_model": summary.get("source_model") or provenance.get("source_model"),
        "source_revision": summary.get("source_snapshot") or provenance.get("source_revision"),
        "quantization": quant.get("weights_dtype") or provenance.get("quantization") or config.get("dtype", config.get("torch_dtype", "unknown")),
        "variant": summary.get("variant"),
    }


def image_device_map(config):
    """Keep the vocabulary tables on CPU for image inference.

    The untied input table and output head each occupy about 1.16 GiB in BF16.
    The head is unused; configure_image_memory handles the input lookup on CPU.
    Tied embeddings must stay together.
    """
    layer_count = getattr(getattr(config, "text_config", None), "num_hidden_layers", 0)
    offload_layers = int(os.environ.get("HIDREAM_OFFLOAD_LAYERS", "0"))
    if not 0 <= offload_layers <= layer_count:
        raise ValueError(f"HIDREAM_OFFLOAD_LAYERS must be between 0 and {layer_count}")
    tied = getattr(config, "tie_word_embeddings", False)
    if tied and not offload_layers:
        return "cuda"
    if not layer_count:
        return {"": "cuda", "lm_head": "cpu", "model.language_model.embed_tokens": "cpu"}
    # A CUDA root plus CPU children makes Accelerate initially move *all*
    # parameters to CUDA, before installing the child offload hooks. Explicit
    # component placement avoids that spike and also supports disk offloading.
    mapping = {name: "cuda" for name in (
        "model.visual", "model.language_model.norm", "model.language_model.rotary_emb",
        "model.t_embedder1", "model.x_embedder", "model.final_layer2",
    )}
    mapping.update({f"model.language_model.layers.{i}": "cpu" if i < offload_layers else "cuda" for i in range(layer_count)})
    vocabulary_device = "cuda" if tied else "cpu"
    mapping.update({"lm_head": vocabulary_device, "model.language_model.embed_tokens": vocabulary_device})
    return mapping


def configure_image_memory(model):
    """Look up prompt embeddings on CPU; transfer only the selected token rows.

    Accelerate normally copies an offloaded module's entire weights to CUDA
    for each call. A vocabulary lookup needs only a few rows from the 1.16 GiB
    table, so execute it on CPU and return the embeddings to the input device.
    All decoder/image computation and precision are preserved.
    """
    from accelerate import cpu_offload
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

    if not getattr(model.config, "tie_word_embeddings", False):
        embeddings = model.get_input_embeddings()
        remove_hook_from_module(embeddings)
        add_hook_to_module(embeddings, AlignDevicesHook(execution_device="cpu", io_same_device=True))

    # Hold an offloaded decoder block on CUDA for its entire forward pass.
    # Otherwise MLP token chunking reloads each large projection for every chunk.
    decoder = model.model.language_model
    device_map = getattr(model, "hf_device_map", {})
    for i, layer in enumerate(getattr(decoder, "layers", ())):
        if device_map.get(f"model.language_model.layers.{i}") == "cpu":
            remove_hook_from_module(layer, recurse=True)
            cpu_offload(layer, execution_device=model.device,
                        preload_module_classes=[type(layer).__name__])
