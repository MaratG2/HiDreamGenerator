import unittest

import numpy as np
from PIL import Image

from background_edit import background_color_for_prompt, composite_background


class BackgroundEditTests(unittest.TestCase):
    def test_only_unambiguous_background_requests_are_recognized(self):
        for prompt in ("Change the background to fully one colored solid black", "Make the background black.",
                       "Please replace the background with pure black"):
            self.assertEqual(background_color_for_prompt(prompt), "#000000")
        self.assertEqual(background_color_for_prompt("Set the background to #123AbC"), "#123abc")
        for prompt in ("Change the background to black and the shirt to red", "Don't change the background to black",
                       "Change the foreground to black", "Change the background to a black forest"):
            self.assertIsNone(background_color_for_prompt(prompt))

    def test_opaque_subject_is_exact_background_is_exact_and_white_edge_spill_is_removed(self):
        pixels = np.full((20, 20, 3), 255, dtype=np.uint8)
        alpha = np.zeros((20, 20), dtype=np.uint8)
        pixels[5:15, 5:15] = (31, 63, 95)
        alpha[5:15, 5:15] = 255
        pixels[4, 10] = (143, 159, 175)  # Subject blended halfway with white.
        alpha[4, 10] = 128
        output = np.asarray(composite_background(Image.fromarray(pixels), Image.fromarray(alpha), "#000000"))
        np.testing.assert_array_equal(output[5:15, 5:15], pixels[5:15, 5:15])
        self.assertEqual(tuple(output[0, 0]), (0, 0, 0))
        np.testing.assert_allclose(output[4, 10], (16, 32, 48), atol=1)

    def test_textured_background_does_not_trigger_flat_color_edge_correction(self):
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        alpha = np.zeros((20, 20), dtype=np.uint8)
        alpha[10, 10] = 128
        result = np.asarray(composite_background(Image.fromarray(pixels), Image.fromarray(alpha), "#000000"))
        np.testing.assert_allclose(result[10, 10], pixels[10, 10].astype(float) * 128 / 255, atol=0.5)
