import unittest
from unittest.mock import patch
from generation_settings import DEFAULT_TIMESTEPS, aspect_dimensions, render_dimensions, resolve_settings
from generation_jobs import GenerationJob, JOB_TTL_SECONDS, MAX_RETAINED_JOBS, prune_jobs


class GenerationSettingsTests(unittest.TestCase):
    def test_full_and_dev_defaults(self):
        full = resolve_settings("full", "t2i", {})
        self.assertEqual((full["num_inference_steps"], full["guidance_scale"], full["scheduler_name"]), (50, 5, "default"))
        for mode, scheduler in (("t2i", "flash"), ("edit", "flow_match"), ("subject", "flash")):
            dev = resolve_settings("dev", mode, {"num_inference_steps": 28})
            self.assertEqual(dev["scheduler_name"], scheduler)
            self.assertEqual(dev["timesteps_list"], DEFAULT_TIMESTEPS)

    def test_explicit_steps_shift_and_scheduler_are_not_ignored(self):
        settings = resolve_settings("dev", "t2i", {"num_inference_steps": 12, "guidance_scale": 2, "shift": 3})
        self.assertEqual(settings["num_inference_steps"], 12)
        self.assertEqual(settings["guidance_scale"], 2)
        self.assertEqual(settings["shift"], 3)
        self.assertIsNone(settings["timesteps_list"])
        self.assertIsNone(resolve_settings("dev", "t2i", {"shift": 2})["timesteps_list"])
        self.assertEqual(resolve_settings("full", "edit", {"editing_scheduler": "flash"})["scheduler_name"], "flash")

    def test_aspect_size_is_bounded_by_requested_long_side(self):
        for source, expected in (((600, 800), (768, 1024)), ((4000, 2000), (1024, 512)), ((777, 1000), (768, 1024))):
            self.assertEqual(aspect_dimensions(*source, 1024), expected)
        with self.assertRaises(ValueError):
            aspect_dimensions(100000, 1, 1024)

    @patch("generation_jobs.time.monotonic", return_value=100.0)
    def test_retention_prunes_finished_jobs_only(self, clock):
        jobs = {}
        for index in range(MAX_RETAINED_JOBS + 3):
            job = GenerationJob()
            job.put({"type": "done"})
            jobs[str(index)] = job
        active = GenerationJob()
        jobs["active"] = active
        prune_jobs(jobs)
        self.assertLess(len(jobs), MAX_RETAINED_JOBS)
        self.assertIs(jobs["active"], active)
        clock.return_value += JOB_TTL_SECONDS + 1
        prune_jobs(jobs)
        self.assertEqual(list(jobs), ["active"])

    def test_reference_render_canvas_preserves_output_contract(self):
        self.assertEqual(render_dimensions(768, 576, False), (768, 576))
        self.assertEqual(render_dimensions(768, 576, True), (1728, 1280))
        self.assertEqual(render_dimensions(2048, 2048, True), (2048, 2048))
        for width, height in ((1024, 512), (512, 1024), (4096, 512), (64, 1024)):
            rw, rh = render_dimensions(width, height, True)
            self.assertGreaterEqual(rw, width)
            self.assertGreaterEqual(rh, height)
            self.assertLessEqual(max(rw, rh), 4096)
            self.assertEqual(rw % 64, 0)
            self.assertEqual(rh % 64, 0)
