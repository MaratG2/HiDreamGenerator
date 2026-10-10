import json
import tempfile
import unittest
from pathlib import Path

from model_runtime import checkpoint_metadata


class CheckpointMetadataTests(unittest.TestCase):
    def test_renamed_checkpoint_keeps_publisher_identity_without_local_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "renamed-model"
            root.mkdir()
            (root / "config.json").write_text(json.dumps({"quantization_config": {"weights_dtype": "uint4"}}))
            (root / "quantization_summary.json").write_text(json.dumps({
                "repo_id": "publisher/model", "source_model": "upstream/model",
                "source_snapshot": "known-revision", "variant": "protected-projections",
            }))
            metadata = checkpoint_metadata(root)
            self.assertEqual(metadata["id"], "publisher/model")
            self.assertEqual(metadata["source_revision"], "known-revision")
            self.assertEqual(metadata["quantization"], "uint4")
            self.assertNotIn(directory, json.dumps(metadata))

    def test_original_checkpoint_without_quantizer_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"dtype":"bfloat16"}')
            metadata = checkpoint_metadata(root)
            self.assertEqual(metadata["quantization"], "bfloat16")
            self.assertIsNone(metadata["variant"])
