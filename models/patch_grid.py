"""Average aligned HiDream predictions across pixel-patch origins."""

import torch


class PatchGridAverager:
    """Shift the denoiser input, undo the shift on its output, then average.

    The model still predicts every output pixel. This changes sampling rather
    than filtering a finished image or replacing any region with source pixels.
    """

    def __init__(self, height, width, patch_size=32, aligned_reference=False):
        self.height, self.width, self.patch_size = height, width, patch_size
        self.count = height * width // patch_size**2
        self.aligned_reference = aligned_reference

    def pixels(self, patches):
        p = self.patch_size
        return patches.reshape(-1, self.height // p, self.width // p, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, self.height, self.width)

    def patches(self, pixels):
        p = self.patch_size
        return pixels.reshape(-1, 3, self.height // p, p, self.width // p, p).permute(0, 2, 4, 1, 3, 5).reshape(-1, self.count, 3 * p**2)

    def __call__(self, predict, inputs, original):
        target = self.pixels(inputs[:, :self.count])
        original_pixels = self.pixels(original).float()
        total = original_pixels.clone()
        half = self.patch_size // 2
        for offset in ((half, 0), (0, half), (half, half)):
            shifted = self.patches(torch.roll(target, offset, dims=(-2, -1)))
            if inputs.shape[1] > self.count:
                reference = inputs[:, self.count:]
                if self.aligned_reference:
                    reference = self.patches(torch.roll(self.pixels(reference), offset, dims=(-2, -1)))
                shifted = torch.cat((shifted, reference), dim=1)
            prediction = self.pixels(predict(shifted)).float()
            total.add_(torch.roll(prediction, tuple(-v for v in offset), dims=(-2, -1)))
        averaged = total / 4
        # Circular shifts contaminate the outer strip. Keep the natural-grid
        # prediction there and feather the transition over four pixels.
        p = self.patch_size
        y = torch.arange(self.height, device=target.device)
        x = torch.arange(self.width, device=target.device)
        wy = ((torch.minimum(y, self.height - 1 - y) - p).float() / 4).clamp(0, 1)
        wx = ((torch.minimum(x, self.width - 1 - x) - p).float() / 4).clamp(0, 1)
        weight = wy[:, None] * wx[None, :]
        return self.patches(torch.lerp(original_pixels, averaged, weight))
