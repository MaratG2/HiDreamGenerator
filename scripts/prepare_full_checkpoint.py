"""Prepare the original O1 weights in BF16 for memory-efficient loading.

This performs the same dtype conversion as from_pretrained(dtype=bfloat16),
once on disk. It does not apply SDNQ, remove layers, or change the architecture.
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

REPOSITORY = "HiDream-ai/HiDream-O1-Image"
REVISION = "407e54f7dc519bab53c9686f03489043065b0f81"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="HiDream-ai/HiDream-O1-Image")
    parser.add_argument("--output-dir", default="HiDream-ai/HiDream-O1-Image-BF16")
    args = parser.parse_args()
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    if source == output:
        parser.error("source and output directories must differ")

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshot_download(REPOSITORY, revision=REVISION, local_dir=source,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt"], max_workers=2)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "model_provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("source_revision") != REVISION or previous.get("id") != REPOSITORY:
            parser.error("output contains another checkpoint; choose a new directory")
    elif any(output.glob("*.safetensors")):
        parser.error("output already contains unidentified weights; choose a new directory")

    provenance = {"id": REPOSITORY, "source_model": REPOSITORY,
                  "source_revision": REVISION, "weight_precision": "bfloat16", "quantization": "none"}
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    total_size = 0
    for name in sorted(set(index["weight_map"].values())):
        target = output / name
        expected_keys = {key for key, shard in index["weight_map"].items() if shard == name}
        reusable = False
        if target.exists():
            with safe_open(target, framework="pt") as stored:
                reusable = ((stored.metadata() or {}).get("source_revision") == REVISION
                            and set(stored.keys()) == expected_keys
                            and all(stored.get_slice(key).get_dtype() == "BF16" for key in expected_keys))
        if not reusable:
            with safe_open(source / name, framework="pt") as original:
                weights = {key: original.get_tensor(key).to(torch.bfloat16).contiguous()
                           for key in original.keys()}
            temporary = target.with_suffix(".incomplete")
            save_file(weights, temporary, metadata={"format": "pt", "source_revision": REVISION})
            del weights
            gc.collect()
            temporary.replace(target)
        with safe_open(target, framework="pt") as stored:
            for key in stored.keys():
                shape = stored.get_slice(key).get_shape()
                count = 1
                for dimension in shape:
                    count *= dimension
                total_size += count * 2
        print(f"Ready: {name}", flush=True)

    for path in source.glob("*.json"):
        if path.name != "model.safetensors.index.json":
            shutil.copy2(path, output / path.name)
    shutil.copy2(source / "merges.txt", output / "merges.txt")
    index["metadata"]["total_size"] = total_size
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Original weights prepared in BF16: {output}", flush=True)


if __name__ == "__main__":
    main()
