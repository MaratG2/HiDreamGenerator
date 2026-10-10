"""Optional numerical checks; the lightweight API CI does not install PyTorch."""
import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires the inference environment")
class TokenChunkingTests(unittest.TestCase):
    def test_inference_chunks_preserve_mlp_output_and_training_gradients(self):
        import torch
        from models.qwen3_vl_transformers import Qwen3VLTextMLP

        torch.manual_seed(123)
        layer = Qwen3VLTextMLP(SimpleNamespace(hidden_size=16, intermediate_size=32, hidden_act="silu"))
        # More than two chunks and a partial final chunk, with a batch dimension.
        x = torch.randn(2, 4129, 16, requires_grad=True)
        dense = layer(x)
        dense.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with torch.inference_mode():
            actual = layer(x.detach())
        torch.testing.assert_close(actual, dense.detach(), rtol=1e-5, atol=1e-6)

    def test_cpu_embeddings_return_identical_rows_to_cuda_without_moving_table(self):
        import torch
        from accelerate import dispatch_model
        from model_runtime import configure_image_memory, image_device_map

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(tie_word_embeddings=False)
                self.gpu_layer = torch.nn.Linear(16, 16).cuda()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.embed_tokens = torch.nn.Embedding(128, 16).bfloat16()
                self.lm_head = torch.nn.Linear(16, 128)

            def get_input_embeddings(self):
                return self.model.language_model.embed_tokens

        model = TinyModel()
        indices = torch.tensor([[5, 3, 90, 5]], device="cuda")
        expected = model.get_input_embeddings()(indices.cpu()).cuda()
        model = dispatch_model(model, device_map=image_device_map(model.config))
        configure_image_memory(model)
        actual = model.get_input_embeddings()(indices)
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(model.get_input_embeddings().weight.device.type, "cpu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
