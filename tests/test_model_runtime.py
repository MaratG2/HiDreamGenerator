import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_runtime import checkpoint_metadata, image_device_map


class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"HIDREAM_OFFLOAD_LAYERS": "0"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_explicit_decoder_offload_and_invalid_counts(self):
        config = SimpleNamespace(tie_word_embeddings=False, text_config=SimpleNamespace(num_hidden_layers=36))
        with patch.dict("os.environ", {"HIDREAM_OFFLOAD_LAYERS": "16"}):
            mapping = image_device_map(config)
        self.assertEqual(mapping["model.language_model.layers.15"], "cpu")
        self.assertEqual(mapping["model.language_model.layers.16"], "cuda")
        for count in ("-1", "37", "invalid"):
            with self.subTest(count=count), patch.dict("os.environ", {"HIDREAM_OFFLOAD_LAYERS": count}):
                with self.assertRaises(ValueError):
                    image_device_map(config)

    def test_unquantized_copy_keeps_source_identity_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"dtype":"bfloat16"}')
            (root / "model_provenance.json").write_text(json.dumps({
                "id": "publisher/original", "source_model": "publisher/original",
                "source_revision": "pinned-revision", "quantization": "none",
            }))
            metadata = checkpoint_metadata(root)
            self.assertEqual(metadata["id"], "publisher/original")
            self.assertEqual(metadata["source_revision"], "pinned-revision")
            self.assertEqual(metadata["quantization"], "none")

    def test_image_device_map_does_not_materialize_cpu_tables_on_cuda_root(self):
        config = SimpleNamespace(tie_word_embeddings=False, text_config=SimpleNamespace(num_hidden_layers=36))
        mapping = image_device_map(config)
        self.assertNotIn("", mapping)
        self.assertEqual(mapping["lm_head"], "cpu")
        self.assertEqual(mapping["model.language_model.embed_tokens"], "cpu")
        for i in range(36):
            self.assertEqual(mapping[f"model.language_model.layers.{i}"], "cuda")

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
