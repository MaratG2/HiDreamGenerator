import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires the inference environment")
class PatchGridTests(unittest.TestCase):
    def test_grid_predictions_cancel_without_changing_color_or_reference(self):
        import torch
        from models.patch_grid import PatchGridAverager

        grid = PatchGridAverager(128, 160)
        y, x = torch.meshgrid(torch.arange(128), torch.arange(160), indexing="ij")
        # A real image-like signal with both color and fine detail.
        pixels = torch.stack((x.float() / 160, y.float() / 128, (x % 7).float() / 7))[None]
        artifacts = (torch.cos(2 * torch.pi * y / 32) + torch.sin(2 * torch.pi * x / 32))[None, None] * 0.1
        reference = torch.randn(1, 7, 3072)
        inputs = torch.cat((grid.patches(pixels), reference), dim=1)
        calls = []

        def predict(current):
            calls.append(current)
            torch.testing.assert_close(current[:, grid.count:], reference, rtol=0, atol=0)
            return grid.patches(grid.pixels(current[:, :grid.count]) + artifacts)

        natural = predict(inputs)
        corrected = grid.pixels(grid(predict, inputs, natural))
        torch.testing.assert_close(corrected[..., 40:-40, 40:-40], pixels[..., 40:-40, 40:-40], rtol=0, atol=2e-7)
        torch.testing.assert_close(corrected[..., :32, :], grid.pixels(natural)[..., :32, :], rtol=0, atol=0)
        self.assertEqual(len(calls), 4)
        torch.testing.assert_close(grid.pixels(grid.patches(pixels)), pixels, rtol=0, atol=0)

    def test_equivariant_denoiser_is_unchanged(self):
        import torch
        from models.patch_grid import PatchGridAverager

        grid = PatchGridAverager(128, 128)
        patches = torch.randn(2, 16, 3072)
        predict = lambda value: value * 0.5 + 0.125
        original = predict(patches)
        torch.testing.assert_close(grid(predict, patches, original), original, rtol=0, atol=0)

    def test_edit_reference_stays_aligned_so_averaging_does_not_blur_detail(self):
        import torch
        from models.patch_grid import PatchGridAverager

        grid = PatchGridAverager(128, 128, aligned_reference=True)
        source = torch.randn(1, 16, 3072)
        inputs = torch.cat((torch.zeros_like(source), source), dim=1)
        predict = lambda value: value[:, grid.count:]
        torch.testing.assert_close(grid(predict, inputs, source), source, rtol=0, atol=0)
