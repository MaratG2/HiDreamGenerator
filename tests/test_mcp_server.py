from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mcp import Client
from mcp.types import ImageContent
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import mcp_server


def make_png(color: tuple[int, int, int] = (220, 45, 35)) -> bytes:
    buffer = io.BytesIO()
    info = PngInfo()
    info.add_text("hidream", json.dumps({"mock_mode": True, "model_type": "full", "parameters": {"num_inference_steps": 12}}))
    Image.new("RGB", (96, 64), color).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


FAKE_PNG = make_png()


class FakeHiDreamHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []

    def log_message(self, format: str, *args) -> None:
        pass

    def _json(self, status: int, data: dict) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._json(200, {
                "status": "ready",
                "model_type": "dev",
                "supported_modes": ["t2i", "edit", "subject"],
                "mock_mode": False,
            })
            return
        if self.path.startswith("/api/generate/stream/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for event in (
                {"type": "progress", "step": 1, "total": 2},
                {"type": "progress", "step": 2, "total": 2},
                {"type": "done", "image": base64.b64encode(FAKE_PNG).decode("ascii")},
            ):
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        if self.path == "/api/refine":
            self._json(200, {"prompt": payload["prompt"] + ", refined detail"})
            return
        if self.path == "/api/runtime":
            self._json(200, {"mock_mode": bool(payload["mock_mode"]), "model_loaded": True, "model_type": "dev", "attention_backend": "auto"})
            return
        if self.path == "/api/generate/start":
            self.payloads.append(payload)
            self._json(200, {"job_id": f"job-{len(self.payloads)}"})
            return
        self._json(404, {"error": "not found"})


class HiDreamMCPTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHiDreamHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.old_api_url = os.environ.get("HIDREAM_API_URL")
        cls.old_output_dir = os.environ.get("HIDREAM_MCP_OUTPUT_DIR")
        os.environ["HIDREAM_API_URL"] = f"http://127.0.0.1:{cls.httpd.server_port}"
        os.environ["HIDREAM_MCP_OUTPUT_DIR"] = cls.temp_dir.name

        cls.source_a = Path(cls.temp_dir.name) / "source-a.png"
        cls.source_b = Path(cls.temp_dir.name) / "source-b.png"
        cls.source_a.write_bytes(make_png((30, 80, 200)))
        cls.source_b.write_bytes(make_png((50, 180, 90)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)
        cls.temp_dir.cleanup()
        if cls.old_api_url is None:
            os.environ.pop("HIDREAM_API_URL", None)
        else:
            os.environ["HIDREAM_API_URL"] = cls.old_api_url
        if cls.old_output_dir is None:
            os.environ.pop("HIDREAM_MCP_OUTPUT_DIR", None)
        else:
            os.environ["HIDREAM_MCP_OUTPUT_DIR"] = cls.old_output_dir

    async def test_tool_discovery_status_and_refinement(self) -> None:
        async with Client(mcp_server.mcp) as client:
            tools = await client.list_tools()
            self.assertEqual(
                {tool.name for tool in tools.tools},
                {"get_status", "set_mock_mode", "refine_prompt", "generate_image", "edit_image", "compose_images"},
            )
            status = await client.call_tool("get_status", {})
            self.assertFalse(status.is_error)
            self.assertEqual(status.structured_content["status"], "ready")

            mock = await client.call_tool("set_mock_mode", {"enabled": True})
            self.assertFalse(mock.is_error)
            self.assertTrue(mock.structured_content["mock_mode"])

            refined = await client.call_tool("refine_prompt", {"prompt": "a red crane"})
            self.assertFalse(refined.is_error)
            self.assertIn("refined detail", refined.content[0].text)

    async def test_text_generation_returns_image_and_metadata(self) -> None:
        progress: list[tuple[float, float | None, str | None]] = []

        async def on_progress(value: float, total: float | None, message: str | None) -> None:
            progress.append((value, total, message))

        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(
                "generate_image",
                {
                    "prompt": "a folded paper bird",
                    "width": 1024,
                    "height": 768,
                    "seed": 77,
                    "scheduler": "flash",
                    "num_inference_steps": 12,
                    "guidance_scale": 1.5,
                    "patch_grid_smoothing": False,
                    "output_path": "generated.png",
                },
                progress_callback=on_progress,
            )

        self.assertFalse(result.is_error, result.content)
        self.assertTrue(any(isinstance(item, ImageContent) for item in result.content))
        self.assertEqual(result.structured_content["mode"], "t2i")
        self.assertEqual(result.structured_content["seed"], 77)
        self.assertTrue(result.structured_content["mock_mode"])
        self.assertEqual(result.structured_content["effective_parameters"]["num_inference_steps"], 12)
        self.assertIn("MOCK OUTPUT", result.content[0].text)
        self.assertTrue(Path(result.structured_content["path"]).is_file())
        self.assertTrue(Path(result.structured_content["metadata_path"]).is_file())
        self.assertGreaterEqual(len(progress), 3)
        payload = FakeHiDreamHandler.payloads[-1]
        self.assertEqual(payload["scheduler_name"], "flash")
        self.assertEqual(payload["num_inference_steps"], 12)
        self.assertEqual(payload["guidance_scale"], 1.5)
        self.assertFalse(payload["patch_grid_smoothing"])

    async def test_batch_generation_is_sequential_and_increments_seed(self) -> None:
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool("generate_image", {
                "prompt": "three studies",
                "seed": 100,
                "batch_size": 3,
                "output_path": "batch.png",
            })
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["count"], 3)
        self.assertEqual([item["seed"] for item in result.structured_content["results"]], [100, 101, 102])
        self.assertEqual([item["seed"] for item in FakeHiDreamHandler.payloads[-3:]], [100, 101, 102])

    async def test_edit_and_multi_reference_modes(self) -> None:
        async with Client(mcp_server.mcp) as client:
            edited = await client.call_tool(
                "edit_image",
                {
                    "prompt": "make the background midnight blue",
                    "source_image_path": str(self.source_a),
                    "output_path": "edited.png",
                },
            )
            self.assertFalse(edited.is_error)
            edit_payload = FakeHiDreamHandler.payloads[-1]
            self.assertEqual(edit_payload["mode"], "edit")
            self.assertEqual(len(edit_payload["refs_b64"]), 1)
            self.assertTrue(edit_payload["keep_original_aspect"])
            self.assertTrue(edit_payload["patch_grid_smoothing"])

            composed = await client.call_tool(
                "compose_images",
                {
                    "prompt": "place both subjects in one studio scene",
                    "source_image_paths": [str(self.source_a), str(self.source_b)],
                    "output_path": "composed.png",
                },
            )
            self.assertFalse(composed.is_error)
            compose_payload = FakeHiDreamHandler.payloads[-1]
            self.assertEqual(compose_payload["mode"], "subject")
            self.assertEqual(len(compose_payload["refs_b64"]), 2)
            self.assertTrue(compose_payload["patch_grid_smoothing"])

    async def test_batch_rejects_seed_overflow_before_gpu_request(self):
        count = len(FakeHiDreamHandler.payloads)
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool("generate_image", {"prompt": "overflow", "seed": 4294967295, "batch_size": 2})
        self.assertTrue(result.is_error)
        self.assertEqual(len(FakeHiDreamHandler.payloads), count)

    async def test_existing_batch_output_rejects_before_gpu_request(self):
        (Path(self.temp_dir.name) / "conflict-2.json").write_text("existing")
        count = len(FakeHiDreamHandler.payloads)
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool("generate_image", {"prompt": "conflict", "batch_size": 2, "output_path": "conflict.png"})
        self.assertTrue(result.is_error)
        self.assertEqual(len(FakeHiDreamHandler.payloads), count)

    def test_atomic_write_does_not_replace_an_existing_file(self):
        target = Path(self.temp_dir.name) / "atomic.png"
        mcp_server._atomic_write(target, b"original", False)
        with self.assertRaises(FileExistsError):
            mcp_server._atomic_write(target, b"changed", False)
        self.assertEqual(target.read_bytes(), b"original")
        mcp_server._atomic_write(target, b"explicit", True)
        self.assertEqual(target.read_bytes(), b"explicit")


if __name__ == "__main__":
    unittest.main()
