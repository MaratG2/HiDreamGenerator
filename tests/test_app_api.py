from __future__ import annotations

import json
import base64
import io
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

import app as app_module


class HiDreamAPIContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_state = app_module._STATE.copy()
        app_module._STATE.update({"model": object(), "processor": object(), "model_type": "dev", "mock_mode": False})
        self.client = app_module.app.test_client()

    def tearDown(self) -> None:
        app_module._STATE.clear()
        app_module._STATE.update(self.old_state)
        app_module._JOBS.clear()

    def test_status_and_validation(self) -> None:
        status = self.client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.get_json()["status"], "ready")
        self.assertIn("subject", status.get_json()["supported_modes"])

        invalid = self.client.post("/api/generate/start", json={
            "prompt": "test",
            "mode": "t2i",
            "width": 1000,
            "height": 1024,
        })
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("multiple of 64", invalid.get_json()["error"])

    def test_background_edits_preserve_source_and_report_the_actual_engine(self):
        reference = io.BytesIO()
        Image.new("RGB", (64, 64), (25, 60, 140)).save(reference, "PNG")
        payload = {"prompt": "Change the background to fully one colored solid black", "mode": "edit",
                   "width": 512, "height": 512, "refs_b64": [base64.b64encode(reference.getvalue()).decode()]}

        def composite(source, color, size, callback):
            self.assertEqual(source.getpixel((32, 32)), (25, 60, 140))
            self.assertEqual(color, "#000000")
            return source.resize(size)

        with patch.object(app_module.background_edit, "available", return_value=True), \
             patch.object(app_module.background_edit, "replace_background", side_effect=composite) as editor, \
             patch.object(app_module, "generate_image", return_value=Image.new("RGB", (512, 512))) as generator:
            started = self.client.post("/api/generate/start", json=payload).get_json()
            stream = self.client.get(f"/api/generate/stream/{started['job_id']}").get_data(as_text=True)
            done = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")][-1]
            self.assertEqual(done["type"], "done")
            self.assertEqual(done["engine"], "background_composite")
            self.assertEqual(done["parameters"], {"background_color": "#000000", "preserve_subject": True})
            self.assertEqual(done["render_width"], 512)
            editor.assert_called_once()
            generator.assert_not_called()
            image = Image.open(io.BytesIO(base64.b64decode(done["image"])))
            self.assertEqual(json.loads(image.info["hidream"])["engine"], "background_composite")

            payload["preserve_background_subject"] = False
            started = self.client.post("/api/generate/start", json=payload).get_json()
            self.client.get(f"/api/generate/stream/{started['job_id']}").get_data()
            generator.assert_called_once()

    def test_mock_mode_is_off_by_default_and_runtime_toggle_is_explicit(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(app_module._env_flag("HIDREAM_MOCK_MODE", False))
        app_module._STATE["mock_mode"] = False
        current = self.client.get("/api/runtime")
        self.assertFalse(current.get_json()["mock_mode"])

        invalid = self.client.post("/api/runtime", json={"mock_mode": "yes"})
        self.assertEqual(invalid.status_code, 400)

        enabled = self.client.post("/api/runtime", json={"mock_mode": True})
        self.assertTrue(enabled.get_json()["mock_mode"])
        self.assertTrue(self.client.get("/api/status").get_json()["mock_mode"])

        disabled = self.client.post("/api/runtime", json={"mock_mode": False})
        self.assertFalse(disabled.get_json()["mock_mode"])

    def test_mock_generation_is_clearly_labelled(self) -> None:
        app_module._STATE["model"] = None
        app_module._STATE["processor"] = None
        app_module._STATE["mock_mode"] = True
        started = self.client.post("/api/generate/start", json={
            "prompt": "a test scene",
            "mode": "t2i",
            "width": 512,
            "height": 512,
            "seed": 4,
            "num_inference_steps": 12,
        })
        stream = self.client.get(f"/api/generate/stream/{started.get_json()['job_id']}")
        events = [
            json.loads(line[6:])
            for line in stream.get_data(as_text=True).splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1]["type"], "done")

    def test_refiner_failure_is_reported_instead_of_faking_a_prompt(self) -> None:
        with patch.object(app_module, "refine_prompt", side_effect=RuntimeError("refiner offline")):
            response = self.client.post("/api/refine", json={"prompt": "a small observatory"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "refiner offline")

    def test_advanced_parameters_reach_pipeline(self) -> None:
        captured: dict = {}

        def fake_generate_image(**kwargs):
            captured.update(kwargs)
            callback = kwargs.get("callback")
            if callback:
                callback(0, 2, None)
                callback(1, 2, None)
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), (200, 40, 30))

        with patch.object(app_module, "generate_image", side_effect=fake_generate_image):
            started = self.client.post("/api/generate/start", json={
                "prompt": "test image",
                "mode": "t2i",
                "width": 1024,
                "height": 768,
                "seed": 9,
                "scheduler_name": "flash",
                "num_inference_steps": 12,
                "guidance_scale": 1.25,
                "shift": 2.0,
                "noise_scale_start": 7.0,
                "noise_scale_end": 5.0,
                "noise_clip_std": 0.5,
            })
            self.assertEqual(started.status_code, 200)
            job_id = started.get_json()["job_id"]
            stream = self.client.get(f"/api/generate/stream/{job_id}")

        events = [
            json.loads(line[6:])
            for line in stream.get_data(as_text=True).splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(captured["scheduler_name"], "flash")
        self.assertEqual(captured["num_inference_steps"], 12)
        self.assertIsNone(captured["timesteps_list"])
        self.assertEqual(captured["guidance_scale"], 1.25)
        self.assertEqual(captured["shift"], 2.0)
        self.assertEqual(captured["noise_scale_start"], 7.0)
        self.assertEqual(captured["noise_scale_end"], 5.0)
        self.assertEqual(captured["noise_clip_std"], 0.5)
        self.assertEqual(captured["width"], 1024)
        self.assertEqual(captured["height"], 768)

    def test_generation_start_returns_before_worker_enters_pipeline(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def blocked_generate_image(**kwargs):
            entered.set()
            release.wait(timeout=2)
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), (20, 40, 80))

        with patch.object(app_module, "generate_image", side_effect=blocked_generate_image):
            started_at = time.monotonic()
            response = self.client.post("/api/generate/start", json={
                "prompt": "start timing",
                "mode": "t2i",
                "width": 1024,
                "height": 1024,
            })
            elapsed = time.monotonic() - started_at
            self.assertEqual(response.status_code, 200)
            self.assertLess(elapsed, 0.5)
            self.assertTrue(entered.wait(timeout=1))
            release.set()
            self.client.get(f"/api/generate/stream/{response.get_json()['job_id']}").get_data()

    def events(self, job_id, **kwargs):
        response = self.client.get(f"/api/generate/stream/{job_id}", **kwargs)
        return [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines() if line.startswith("data: ")]

    def test_strict_input_validation(self):
        cases = [{"prompt": "x", "seed": 10 ** 1000}, {"prompt": "x", "image_size": 2048}, [], None, {"prompt": 3}, {"prompt": "x", "width": 1024.5},
                 {"prompt": "x", "seed": True}, {"prompt": "x", "seed": 1e100},
                 {"prompt": "x", "refs_b64": "bad"}, {"prompt": "x", "keep_original_aspect": "false"},
                 {"prompt": "x", "num_inference_steps": 1.2}, {"prompt": "x", "shift": float("inf")},
                 {"prompt": "x", "guidance_scale": float("nan")},
                 {"prompt": "x", "mode": "edit", "refs_b64": ["not base64!"]},
                 {"prompt": "x", "mode": "edit", "refs_b64": [base64.b64encode(b"not an image").decode()]},
                 {"prompt": "x", "refs_b64": ["unexpected"]},
                 {"prompt": "x", "scheduler_name": "default", "noise_clip_std": 2}]
        for payload in cases:
            with self.subTest(payload=payload):
                response = self.client.post("/api/generate/start", data=json.dumps(payload), content_type="application/json")
                self.assertEqual(response.status_code, 400)
                self.assertIsInstance(response.get_json()["error"], str)
        self.assertEqual(self.client.post("/api/generate/start", data="prompt=x").status_code, 415)
        self.assertEqual(self.client.post("/api/generate/start", data="{", content_type="application/json").status_code, 400)

    def test_model_not_loaded_rejects_before_creating_job(self):
        app_module._STATE["model"] = None
        response = self.client.post("/api/generate/start", json={"prompt": "x"})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(app_module._JOBS)

    def test_source_aspect_respects_resolution_and_png_metadata(self):
        source = io.BytesIO()
        Image.new("RGB", (600, 800)).save(source, format="PNG")
        app_module._STATE["mock_mode"] = True
        for long_side, expected in ((1024, (768, 1024)), (1536, (1152, 1536))):
            started = self.client.post("/api/generate/start", json={
                "prompt": "portrait edit", "mode": "edit", "width": long_side, "height": long_side,
                "keep_original_aspect": True, "refs_b64": [base64.b64encode(source.getvalue()).decode()]})
            self.assertEqual(started.status_code, 200)
            event = self.events(started.get_json()["job_id"])[-1]
            self.assertEqual((event["width"], event["height"]), expected)
            self.assertTrue(event["mock_mode"])
            with Image.open(io.BytesIO(base64.b64decode(event["image"]))) as result:
                self.assertEqual(result.size, expected)
                self.assertTrue(json.loads(result.info["hidream"])["mock_mode"])

    def test_stream_replays_terminal_result_and_resumes_by_id(self):
        app_module._STATE["mock_mode"] = True
        response = self.client.post("/api/generate/start", json={"prompt": "replay", "width": 512, "height": 512})
        job_id = response.get_json()["job_id"]
        first = self.events(job_id)
        self.assertEqual(self.events(job_id), first)
        self.assertEqual(self.events(job_id, headers={"Last-Event-ID": "2"}), [first[-1]])
        self.assertEqual(self.client.get(f"/api/generate/stream/{job_id}", headers={"Last-Event-ID": "999"}).status_code, 400)

    def test_reference_render_size_is_disclosed_without_changing_output(self):
        checkpoint = {"id": "example/quality-checkpoint", "quantization": "uint4", "variant": "protected-projections"}
        app_module._STATE["checkpoint"] = checkpoint
        self.assertEqual(self.client.get("/api/runtime").get_json()["checkpoint"], checkpoint)
        self.assertEqual(self.client.get("/api/status").get_json()["checkpoint"], checkpoint)
        source = io.BytesIO()
        Image.new("RGB", (800, 600)).save(source, format="PNG")
        with patch.object(app_module, "generate_image", return_value=Image.new("RGB", (768, 576))):
            response = self.client.post("/api/generate/start", json={
                "prompt": "blue teapot", "mode": "edit", "width": 768, "height": 768,
                "keep_original_aspect": True, "refs_b64": [base64.b64encode(source.getvalue()).decode()]})
            data = response.get_json()
            self.assertEqual((data["width"], data["height"]), (768, 576))
            self.assertEqual((data["render_width"], data["render_height"]), (2368, 1792))
            event = self.events(data["job_id"])[-1]
        with Image.open(io.BytesIO(base64.b64decode(event["image"]))) as result:
            metadata = json.loads(result.info["hidream"])
            self.assertEqual(result.size, (768, 576))
            self.assertEqual(metadata["render_width"], 2368)
            self.assertFalse(metadata["mock_mode"])
            self.assertEqual(metadata["checkpoint"], checkpoint)
            self.assertEqual(event["checkpoint"], checkpoint)

    def test_wrong_pipeline_dimensions_fail_instead_of_claiming_success(self):
        with patch.object(app_module, "generate_image", return_value=Image.new("RGB", (64, 64))):
            response = self.client.post("/api/generate/start", json={"prompt": "x"})
            event = self.events(response.get_json()["job_id"])[-1]
        self.assertEqual(event["type"], "error")
        self.assertIn("expected 1024x1024", event["message"])

    def test_busy_runtime_and_bounded_queue(self):
        release = threading.Event()
        def blocked(**kwargs):
            release.wait(5)
            return Image.new("RGB", (kwargs["width"], kwargs["height"]))
        with patch.object(app_module, "generate_image", side_effect=blocked):
            jobs = []
            try:
                for _ in range(4):
                    response = self.client.post("/api/generate/start", json={"prompt": "x", "width": 512, "height": 512})
                    self.assertEqual(response.status_code, 200)
                    jobs.append(response.get_json()["job_id"])
                self.assertEqual(self.client.post("/api/generate/start", json={"prompt": "x"}).status_code, 429)
                self.assertEqual(self.client.post("/api/runtime", json={"mock_mode": True}).status_code, 409)
            finally:
                release.set()
                for job in jobs:
                    self.events(job)

    def test_origin_host_and_refiner_configuration(self):
        response = self.client.post("/api/runtime", json={"mock_mode": True}, headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/status", headers={"Host": "untrusted.example"}).status_code, 400)
        with patch.dict("os.environ", {"HIDREAM_REFINE_BASE_URL": "http://configured:8000/v1", "HIDREAM_REFINE_MODEL": "configured-model"}):
            with patch.object(app_module, "refine_prompt", return_value="refined") as refiner:
                response = self.client.post("/api/refine", json={"prompt": "x"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(refiner.call_args.kwargs["base_url"], "http://configured:8000/v1")
                self.assertEqual(refiner.call_args.kwargs["model_id"], "configured-model")
            self.assertEqual(self.client.post("/api/refine", json={"prompt": "x", "base_url": "http://elsewhere"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
