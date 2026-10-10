"""Download the pinned, local BiRefNet mask model used for background edits."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from background_edit import MODEL_ID, MODEL_REVISION, model_path

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    print(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_path(),
                            allow_patterns=["*.json", "*.py", "*.safetensors", "README.md"], max_workers=2))
