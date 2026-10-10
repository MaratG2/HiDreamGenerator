"""Optional numerical checks; the lightweight API CI does not install PyTorch."""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires the inference environment")
class TokenChunkingTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"HIDREAM_OFFLOAD_LAYERS": "0"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_generation_uses_execution_gpu_when_first_parameter_is_offloaded(self):
        import torch
        from models.pipeline import generate_image

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        class Model:
            device = torch.device("meta")
            hf_device_map = {"": "cuda:0", "model.visual": "cpu"}
            config = SimpleNamespace()

            def __call__(self, **kwargs):
                self.execution_device = kwargs["vinputs"].device.type
                return SimpleNamespace(x_pred=torch.zeros((1, 256, 3072), device="cuda"))

        sample = {"input_ids": torch.zeros((1, 1), dtype=torch.long),
                  "position_ids": torch.zeros((3, 1, 256), dtype=torch.long),
                  "token_types": torch.ones((1, 256), dtype=torch.long),
                  "vinput_mask": torch.ones((1, 256), dtype=torch.bool)}
        scheduler = SimpleNamespace(timesteps=torch.tensor([500.]),
                                    step=lambda prediction, timestep, state, **kwargs: (torch.zeros_like(state),))
        model = Model()
        with patch("models.pipeline.build_t2i_text_sample", return_value=sample), patch("models.pipeline.build_scheduler", return_value=scheduler):
            result = generate_image(model, SimpleNamespace(), "test", width=512, height=512,
                                    num_inference_steps=1, guidance_scale=1, use_flash_attn=False)
        self.assertEqual(model.execution_device, "cuda")
        self.assertEqual(result.getpixel((100, 100)), (128, 128, 128))

    def test_sampler_preserves_updates_smaller_than_bfloat16_resolution(self):
        import torch
        from models.pipeline import generate_image

        class Model:
            device = torch.device("cpu")
            config = SimpleNamespace()

            def __call__(self, **kwargs):
                self.input_dtype = kwargs["vinputs"].dtype
                return SimpleNamespace(x_pred=torch.zeros((1, 256, 3072)))

        sample = {"input_ids": torch.zeros((1, 1), dtype=torch.long),
                  "position_ids": torch.zeros((3, 1, 256), dtype=torch.long),
                  "token_types": torch.ones((1, 256), dtype=torch.long),
                  "vinput_mask": torch.ones((1, 256), dtype=torch.bool)}
        for scheduler_name in ("default", "flash"):
            with self.subTest(scheduler=scheduler_name):
                states = []

                def step(prediction, timestep, state, **kwargs):
                    states.append(state[0, 0, 0].item())
                    self.assertEqual(state.dtype, torch.float32)
                    return (torch.full_like(state, 0.5) if len(states) == 1 else state + 0.0004,)

                scheduler = SimpleNamespace(timesteps=torch.tensor([900., 700., 500., 300.]), step=step)
                model = Model()
                with patch("models.pipeline.build_t2i_text_sample", return_value=sample), patch("models.pipeline.build_scheduler", return_value=scheduler):
                    generate_image(model, SimpleNamespace(), "test", width=512, height=512,
                                   num_inference_steps=4, guidance_scale=1, noise_scale_start=0,
                                   scheduler_name=scheduler_name, use_flash_attn=False)
                # Recasting the state to BF16 each step loses both small updates.
                self.assertAlmostEqual(states[-1], 0.5008, places=6)
                self.assertEqual(model.input_dtype, torch.bfloat16)

    def test_preview_matches_guided_prediction_instead_of_unguided_branch(self):
        import torch
        from models.pipeline import generate_image

        class Model:
            device = torch.device("cpu")
            config = SimpleNamespace()
            calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(x_pred=torch.full((1, 256, 3072), 0.25 if self.calls == 1 else 0.0))

        sample = {"input_ids": torch.zeros((1, 1), dtype=torch.long),
                  "position_ids": torch.zeros((3, 1, 256), dtype=torch.long),
                  "token_types": torch.ones((1, 256), dtype=torch.long),
                  "vinput_mask": torch.ones((1, 256), dtype=torch.bool)}
        scheduler = SimpleNamespace(timesteps=torch.tensor([500.0]),
                                    step=lambda prediction, timestep, state, **kwargs: (state - 0.5 * prediction,))
        previews = []
        with patch("models.pipeline.build_t2i_text_sample", return_value=sample), patch("models.pipeline.build_scheduler", return_value=scheduler):
            result = generate_image(Model(), SimpleNamespace(), "test", width=512, height=512,
                                    num_inference_steps=1, guidance_scale=3, noise_scale_start=0,
                                    use_flash_attn=False, callback=lambda step, total, preview: previews.append(preview()))
        # Guided x0 is 0.75 -> RGB 223; the misleading conditional preview was 159.
        self.assertEqual(previews[0].getpixel((100, 100)), (223, 223, 223))
        self.assertEqual(previews[0].getpixel((100, 100)), result.getpixel((100, 100)))

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

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(16, 16)

            def forward(self, x):
                # Direct child-weight access requires loading the whole block.
                return torch.nn.functional.linear(x, self.proj.weight, self.proj.bias)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(tie_word_embeddings=False)
                self.gpu_layer = torch.nn.Linear(16, 16).cuda()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.embed_tokens = torch.nn.Embedding(128, 16).bfloat16()
                self.model.language_model.layers = torch.nn.ModuleList([Decoder()])
                self.lm_head = torch.nn.Linear(16, 128)
                self.device = torch.device("cuda")

            def get_input_embeddings(self):
                return self.model.language_model.embed_tokens

        model = TinyModel()
        indices = torch.tensor([[5, 3, 90, 5]], device="cuda")
        expected = model.get_input_embeddings()(indices.cpu()).cuda()
        x = torch.randn(2, 16)
        decoder_expected = model.model.language_model.layers[0](x).cuda()
        mapping = image_device_map(model.config)
        mapping["model.language_model.layers.0"] = "cpu"
        model = dispatch_model(model, device_map=mapping)
        configure_image_memory(model)
        actual = model.get_input_embeddings()(indices)
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(model.get_input_embeddings().weight.device.type, "cpu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        decoder = model.model.language_model.layers[0]
        torch.testing.assert_close(decoder(x.cuda()), decoder_expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(decoder.proj.weight.device.type, "meta")
