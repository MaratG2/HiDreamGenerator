"""MCP server exposing the local HiDream image service to agents.

The MCP process is intentionally lightweight. It talks to the Flask image server
over HTTP so the model is loaded only once and shared by the web UI and agents.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from PIL import Image as PILImage
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_API_URL = "http://127.0.0.1:7860"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "mcp"
logging.getLogger("httpx").setLevel(logging.WARNING)

SERVER_INSTRUCTIONS = (
    "Use generate_image for text-to-image, edit_image for exactly one source image, "
    "and compose_images for two or more reference images. The local HiDream web/API "
    "server must be running first. Every generation saves a full PNG plus JSON metadata "
    "and returns an image preview. Prefer 1024px for iteration; 2048px can take minutes. "
    "GPU jobs are serialized, so use batch_size instead of launching generations in parallel. "
    "Mock mode is off by default and should only be enabled explicitly for integration testing."
)

mcp = MCPServer(
    name="HiDream",
    title="HiDream Local Image Generator",
    description="Generate and edit images with a locally hosted HiDream-O1 model.",
    instructions=SERVER_INSTRUCTIONS,
    version="0.1.0",
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
GENERATES_FILES = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

Dimension = Annotated[
    int,
    Field(
        ge=512,
        le=4096,
        multiple_of=64,
        description="Output dimension in pixels; multiples of 64 only.",
    ),
]
Seed = Annotated[int, Field(ge=0, le=4_294_967_295, description="Deterministic random seed.")]
Timeout = Annotated[
    int,
    Field(ge=30, le=3600, description="Maximum time to wait for the local generator, in seconds."),
]
PreviewSize = Annotated[
    int,
    Field(
        ge=256,
        le=2048,
        description="Maximum side of the image returned to the agent. The saved PNG remains full size.",
    ),
]
BatchSize = Annotated[int, Field(ge=1, le=4, description="Number of sequential images to generate.")]
Scheduler = Literal["auto", "default", "flow_match", "flash"]


class HiDreamStatus(BaseModel):
    status: str
    api_url: str
    output_directory: str
    model_type: str | None = None
    device: str | None = None
    attention_backend: str | None = None
    supported_modes: list[str] = Field(default_factory=list)
    supported_schedulers: list[str] = Field(default_factory=list)
    max_concurrent_generations: int | None = None
    mock_mode: bool = False
    checkpoint: dict | None = None
    background_edit_available: bool = False
    error: str | None = None


class HiDreamRuntime(BaseModel):
    mock_mode: bool
    model_loaded: bool
    model_type: str
    attention_backend: str
    checkpoint: dict | None = None
    background_edit_available: bool = False


def _api_url(path: str) -> str:
    base = os.environ.get("HIDREAM_API_URL", DEFAULT_API_URL).strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("HIDREAM_API_URL must start with http:// or https://")
    return f"{base}{path}"


def _output_dir() -> Path:
    configured = os.environ.get("HIDREAM_MCP_OUTPUT_DIR", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_OUTPUT_DIR


async def _response_json(response: httpx.Response, action: str) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"HiDream returned an invalid response while {action}.") from exc
    if response.is_error:
        detail = data.get("error") if isinstance(data, dict) else None
        raise RuntimeError(detail or f"HiDream {action} failed with HTTP {response.status_code}.")
    if not isinstance(data, dict):
        raise RuntimeError(f"HiDream returned an unexpected response while {action}.")
    return data


async def _post_json(path: str, payload: dict, timeout_seconds: int = 30) -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=5.0)) as client:
            response = await client.post(_api_url(path), json=payload)
        return await _response_json(response, "processing the request")
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "The HiDream API is not reachable. Start it with .\\start_image_server.ps1 first."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("The HiDream API request timed out.") from exc


def _resolve_input_image(path_value: str) -> tuple[Path, str, tuple[int, int]]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise ValueError(f"Source image does not exist: {path}")
    if path.stat().st_size > 24 * 1024 * 1024:
        raise ValueError(f"Source image is larger than 24 MB: {path}")

    raw = path.read_bytes()
    try:
        with PILImage.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"} or image.width * image.height > 24_000_000:
                raise ValueError("Use PNG, JPEG, or WebP up to 24 megapixels")
            image.verify()
        with PILImage.open(io.BytesIO(raw)) as image:
            size = image.size
    except Exception as exc:
        raise ValueError(f"Source file is not a readable image: {path}") from exc
    return path, base64.b64encode(raw).decode("ascii"), size


def _slug(prompt: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", prompt.strip().lower()).strip("-")
    return value[:48] or "image"


def _resolve_output_path(prompt: str, seed: int, output_path: str | None) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            path = _output_dir() / path
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = _output_dir() / f"{stamp}-{_slug(prompt)}-s{seed}-{uuid.uuid4().hex[:8]}.png"
    if not path.suffix:
        path = path.with_suffix(".png")
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must use the .png extension")
    return path.resolve()


def _atomic_write(path: Path, data: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Choose another path or set overwrite=true.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic create-if-absent: another writer cannot win the exists/replace race.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _preview_bytes(png: bytes, max_side: int) -> tuple[bytes, tuple[int, int]]:
    with PILImage.open(io.BytesIO(png)) as image:
        image.load()
        full_size = image.size
        image.thumbnail((max_side, max_side), PILImage.Resampling.LANCZOS)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), full_size


async def _generate_via_api(payload: dict, ctx: Context, timeout_seconds: int) -> bytes:
    timeout = httpx.Timeout(timeout_seconds, connect=5.0, read=timeout_seconds)
    try:
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(timeout=timeout) as client:
                start_response = await client.post(_api_url("/api/generate/start"), json=payload)
                started = await _response_json(start_response, "starting generation")
                job_id = started.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise RuntimeError("HiDream did not return a generation job ID.")

                await ctx.report_progress(0, total=1, message="Generation queued on the local GPU")
                last_progress = 0
                async with client.stream("GET", _api_url(f"/api/generate/stream/{job_id}")) as stream:
                    if stream.is_error:
                        body = await stream.aread()
                        try:
                            detail = json.loads(body).get("error")
                        except Exception:
                            detail = None
                        raise RuntimeError(detail or f"HiDream progress stream failed with HTTP {stream.status_code}.")

                    async for line in stream.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            event = json.loads(line[5:].strip())
                        except json.JSONDecodeError as exc:
                            raise RuntimeError("HiDream sent malformed progress data.") from exc

                        event_type = event.get("type")
                        if event_type == "progress":
                            step = int(event.get("step", 0))
                            total = max(int(event.get("total", 1)), 1)
                            if step > last_progress:
                                last_progress = step
                                await ctx.report_progress(
                                    step,
                                    total=total,
                                    message=f"Rendering step {step} of {total}",
                                )
                        elif event_type == "done":
                            encoded = event.get("image")
                            if not isinstance(encoded, str):
                                raise RuntimeError("HiDream completed without returning image data.")
                            try:
                                return base64.b64decode(encoded, validate=True)
                            except ValueError as exc:
                                raise RuntimeError("HiDream returned invalid image data.") from exc
                        elif event_type == "error":
                            raise RuntimeError(str(event.get("message") or "HiDream generation failed."))
    except TimeoutError as exc:
        raise RuntimeError(f"HiDream generation exceeded the {timeout_seconds}-second timeout.") from exc
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "The HiDream API is not reachable. Start it with .\\start_image_server.ps1 first."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("The HiDream API stopped responding during generation.") from exc

    raise RuntimeError("HiDream closed the progress stream before returning an image.")


def _payload(
    *,
    mode: Literal["t2i", "edit", "subject"],
    prompt: str,
    refs_b64: list[str],
    width: int,
    height: int,
    seed: int,
    keep_original_aspect: bool,
    scheduler: Scheduler,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    shift: float | None,
    noise_scale_start: float | None,
    noise_scale_end: float | None,
    noise_clip_std: float | None,
) -> dict:
    data: dict = {
        "mode": mode,
        "prompt": prompt,
        "refs_b64": refs_b64,
        "width": width,
        "height": height,
        "seed": seed,
        "keep_original_aspect": keep_original_aspect,
    }
    if scheduler != "auto":
        data["scheduler_name"] = scheduler
    if num_inference_steps is not None:
        data["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        data["guidance_scale"] = guidance_scale
    if shift is not None:
        data["shift"] = shift
    if noise_scale_start is not None:
        data["noise_scale_start"] = noise_scale_start
    if noise_scale_end is not None:
        data["noise_scale_end"] = noise_scale_end
    if noise_clip_std is not None:
        data["noise_clip_std"] = noise_clip_std
    return data


async def _finish_generation(
    *,
    payload: dict,
    ctx: Context,
    output_path: str | None,
    overwrite: bool,
    preview_max_side: int,
    timeout_seconds: int,
    source_paths: list[str],
) -> CallToolResult:
    started = time.monotonic()
    path = _resolve_output_path(payload["prompt"], payload["seed"], output_path)
    metadata_path = path.with_suffix(".json")
    if not overwrite:
        for candidate in (path, metadata_path):
            if candidate.exists():
                raise FileExistsError(
                    f"Output already exists: {candidate}. Choose another path or set overwrite=true."
                )
    png = await _generate_via_api(payload, ctx, timeout_seconds)
    preview, full_size = _preview_bytes(png, preview_max_side)
    with PILImage.open(io.BytesIO(png)) as image:
        effective = json.loads(image.info.get("hidream", "{}"))
    _atomic_write(path, png, overwrite)

    metadata = {
        "status": "completed",
        "mode": payload["mode"],
        "prompt": payload["prompt"],
        "path": str(path),
        "metadata_path": str(metadata_path),
        "width": full_size[0],
        "height": full_size[1],
        "render_width": effective.get("render_width"),
        "render_height": effective.get("render_height"),
        "seed": payload["seed"],
        "source_images": source_paths,
        "parameters": {key: value for key, value in payload.items() if key != "refs_b64"},
        "effective_parameters": effective.get("parameters"),
        "mock_mode": effective.get("mock_mode"),
        "model_type": effective.get("model_type"),
        "checkpoint": effective.get("checkpoint"),
        "engine": effective.get("engine"),
        "sha256": hashlib.sha256(png).hexdigest(),
        "duration_seconds": round(time.monotonic() - started, 2),
    }
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        overwrite,
    )

    result_description = ("Background replaced with the original subject preserved."
                          if effective.get("engine") == "background_composite"
                          else f"HiDream {payload['mode']} generation completed{' (MOCK OUTPUT)' if effective.get('mock_mode') else ''}.")
    summary = (
        f"{result_description}\n"
        f"Full PNG: {path}\n"
        f"Metadata: {metadata_path}\n"
        f"Size: {full_size[0]} x {full_size[1]}\n"
        f"Seed: {payload['seed']}"
    )
    return CallToolResult(
        content=[
            TextContent(type="text", text=summary),
            ImageContent(
                type="image",
                data=base64.b64encode(preview).decode("ascii"),
                mimeType="image/png",
            ),
        ],
        structuredContent=metadata,
    )


async def _finish_batch(
    *,
    payload: dict,
    batch_size: int,
    ctx: Context,
    output_path: str | None,
    overwrite: bool,
    preview_max_side: int,
    timeout_seconds: int,
    source_paths: list[str],
) -> CallToolResult:
    if payload["seed"] + batch_size - 1 > 4_294_967_295:
        raise ValueError("The batch would exceed the maximum seed 4294967295")
    if not payload["prompt"].strip():
        raise ValueError("prompt must not be empty")
    # Fail before spending GPU time if any explicit batch destination conflicts.
    if output_path and not overwrite and batch_size > 1:
        base = Path(output_path)
        for index in range(batch_size):
            candidate = str(base.with_name(f"{base.stem}-{index + 1}{base.suffix or '.png'}"))
            resolved = _resolve_output_path(payload["prompt"], payload["seed"] + index, candidate)
            for target in (resolved, resolved.with_suffix(".json")):
                if target.exists():
                    raise FileExistsError(f"Output already exists: {target}")
    if batch_size == 1:
        return await _finish_generation(
            payload=payload,
            ctx=ctx,
            output_path=output_path,
            overwrite=overwrite,
            preview_max_side=preview_max_side,
            timeout_seconds=timeout_seconds,
            source_paths=source_paths,
        )

    content = []
    results = []
    for index in range(batch_size):
        item_payload = {**payload, "seed": payload["seed"] + index}
        item_output = output_path
        if output_path:
            path = Path(output_path)
            suffix = path.suffix or ".png"
            item_output = str(path.with_name(f"{path.stem}-{index + 1}{suffix}"))
        result = await _finish_generation(
            payload=item_payload,
            ctx=ctx,
            output_path=item_output,
            overwrite=overwrite,
            preview_max_side=preview_max_side,
            timeout_seconds=timeout_seconds,
            source_paths=source_paths,
        )
        content.extend(result.content)
        results.append(result.structured_content)
    return CallToolResult(
        content=content,
        structuredContent={"status": "completed", "count": batch_size, "results": results},
    )


async def _maybe_refine(prompt: str, refine: bool) -> str:
    if not refine:
        return prompt.strip()
    data = await _post_json("/api/refine", {"prompt": prompt.strip()}, timeout_seconds=120)
    refined = data.get("prompt")
    if not isinstance(refined, str) or not refined.strip():
        raise RuntimeError("The prompt refiner returned an empty prompt.")
    return refined.strip()


@mcp.tool(title="Check HiDream status", annotations=READ_ONLY)
async def get_status() -> HiDreamStatus:
    """Check whether the local HiDream image API is ready and report its capabilities."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=3)) as client:
            response = await client.get(_api_url("/api/status"))
        status = await _response_json(response, "checking status")
    except Exception as exc:
        status = {
            "status": "unavailable",
            "error": (
                f"{exc} Start or restart the image server with "
                ".\\start_image_server.ps1 so it exposes the current API."
            ),
        }
    status["api_url"] = _api_url("")
    status["output_directory"] = str(_output_dir().resolve())
    return HiDreamStatus.model_validate(status)


@mcp.tool(title="Set HiDream mock mode", annotations=GENERATES_FILES)
async def set_mock_mode(enabled: bool) -> HiDreamRuntime:
    """Explicitly enable or disable synthetic test images. Real inference is the default."""
    return HiDreamRuntime.model_validate(await _post_json("/api/runtime", {"mock_mode": enabled}))


@mcp.tool(title="Refine an image prompt", annotations=READ_ONLY)
async def refine_prompt(
    prompt: Annotated[str, Field(min_length=1, max_length=12000, description="Prompt to improve.")],
) -> str:
    """Improve an image prompt with the configured local prompt-refinement model."""
    return await _maybe_refine(prompt, True)


@mcp.tool(title="Generate an image", annotations=GENERATES_FILES, structured_output=False)
async def generate_image(
    prompt: Annotated[str, Field(min_length=1, max_length=12000, description="Image description.")],
    ctx: Context,
    width: Dimension = 1024,
    height: Dimension = 1024,
    seed: Seed = 32,
    scheduler: Scheduler = "auto",
    num_inference_steps: Annotated[int | None, Field(ge=1, le=100)] = None,
    guidance_scale: Annotated[float | None, Field(ge=0, le=20)] = None,
    shift: Annotated[float | None, Field(gt=0, le=20)] = None,
    noise_scale_start: Annotated[float | None, Field(ge=0, le=32)] = None,
    noise_scale_end: Annotated[float | None, Field(ge=0, le=32)] = None,
    noise_clip_std: Annotated[float | None, Field(ge=0, le=32)] = None,
    refine: bool = False,
    output_path: Annotated[str | None, Field(description="Optional .png path; relative paths use the MCP output directory.")] = None,
    overwrite: bool = False,
    preview_max_side: PreviewSize = 1024,
    timeout_seconds: Timeout = 1200,
    batch_size: BatchSize = 1,
) -> CallToolResult:
    """Create a new image from text and return a preview plus the saved full-resolution PNG path."""
    final_prompt = await _maybe_refine(prompt, refine)
    payload = _payload(
        mode="t2i",
        prompt=final_prompt,
        refs_b64=[],
        width=width,
        height=height,
        seed=seed,
        keep_original_aspect=False,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        noise_clip_std=noise_clip_std,
    )
    return await _finish_batch(
        payload=payload,
        batch_size=batch_size,
        ctx=ctx,
        output_path=output_path,
        overwrite=overwrite,
        preview_max_side=preview_max_side,
        timeout_seconds=timeout_seconds,
        source_paths=[],
    )


@mcp.tool(title="Edit an image", annotations=GENERATES_FILES, structured_output=False)
async def edit_image(
    prompt: Annotated[str, Field(min_length=1, max_length=12000, description="Describe only the requested change.")],
    source_image_path: Annotated[str, Field(description="Absolute or working-directory-relative source image path.")],
    ctx: Context,
    width: Dimension = 1024,
    height: Dimension = 1024,
    seed: Seed = 32,
    keep_original_aspect: bool = True,
    preserve_background_subject: Annotated[bool, Field(description="Preserve source colors and detail for simple solid-background requests when the local mask model is installed. Set false for generative editing.")] = True,
    scheduler: Scheduler = "auto",
    num_inference_steps: Annotated[int | None, Field(ge=1, le=100)] = None,
    guidance_scale: Annotated[float | None, Field(ge=0, le=20)] = None,
    shift: Annotated[float | None, Field(gt=0, le=20)] = None,
    noise_scale_start: Annotated[float | None, Field(ge=0, le=32)] = None,
    noise_scale_end: Annotated[float | None, Field(ge=0, le=32)] = None,
    noise_clip_std: Annotated[float | None, Field(ge=0, le=32)] = None,
    refine: bool = False,
    output_path: Annotated[str | None, Field(description="Optional .png path; relative paths use the MCP output directory.")] = None,
    overwrite: bool = False,
    preview_max_side: PreviewSize = 1024,
    timeout_seconds: Timeout = 1200,
    batch_size: BatchSize = 1,
) -> CallToolResult:
    """Edit exactly one local image and return a preview plus the saved full-resolution PNG path."""
    source_path, encoded, _ = _resolve_input_image(source_image_path)
    final_prompt = await _maybe_refine(prompt, refine)
    payload = _payload(
        mode="edit",
        prompt=final_prompt,
        refs_b64=[encoded],
        width=width,
        height=height,
        seed=seed,
        keep_original_aspect=keep_original_aspect,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        noise_clip_std=noise_clip_std,
    )
    payload["preserve_background_subject"] = preserve_background_subject
    return await _finish_batch(
        payload=payload,
        batch_size=batch_size,
        ctx=ctx,
        output_path=output_path,
        overwrite=overwrite,
        preview_max_side=preview_max_side,
        timeout_seconds=timeout_seconds,
        source_paths=[str(source_path)],
    )


@mcp.tool(title="Compose reference images", annotations=GENERATES_FILES, structured_output=False)
async def compose_images(
    prompt: Annotated[str, Field(min_length=1, max_length=12000, description="Describe how to combine or preserve the referenced subjects.")],
    source_image_paths: Annotated[
        list[str],
        Field(min_length=2, max_length=8, description="Two to eight local reference-image paths."),
    ],
    ctx: Context,
    width: Dimension = 1024,
    height: Dimension = 1024,
    seed: Seed = 32,
    scheduler: Scheduler = "auto",
    num_inference_steps: Annotated[int | None, Field(ge=1, le=100)] = None,
    guidance_scale: Annotated[float | None, Field(ge=0, le=20)] = None,
    shift: Annotated[float | None, Field(gt=0, le=20)] = None,
    noise_scale_start: Annotated[float | None, Field(ge=0, le=32)] = None,
    noise_scale_end: Annotated[float | None, Field(ge=0, le=32)] = None,
    noise_clip_std: Annotated[float | None, Field(ge=0, le=32)] = None,
    refine: bool = False,
    output_path: Annotated[str | None, Field(description="Optional .png path; relative paths use the MCP output directory.")] = None,
    overwrite: bool = False,
    preview_max_side: PreviewSize = 1024,
    timeout_seconds: Timeout = 1200,
    batch_size: BatchSize = 1,
) -> CallToolResult:
    """Generate a subject-consistent composition from two or more local reference images."""
    resolved = [_resolve_input_image(value) for value in source_image_paths]
    final_prompt = await _maybe_refine(prompt, refine)
    payload = _payload(
        mode="subject",
        prompt=final_prompt,
        refs_b64=[item[1] for item in resolved],
        width=width,
        height=height,
        seed=seed,
        keep_original_aspect=False,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        noise_clip_std=noise_clip_std,
    )
    return await _finish_batch(
        payload=payload,
        batch_size=batch_size,
        ctx=ctx,
        output_path=output_path,
        overwrite=overwrite,
        preview_max_side=preview_max_side,
        timeout_seconds=timeout_seconds,
        source_paths=[str(item[0]) for item in resolved],
    )


if __name__ == "__main__":
    # Never print to stdout: stdio is the MCP protocol transport.
    sys.stderr.write("HiDream MCP server starting over stdio.\n")
    sys.stderr.flush()
    mcp.run()
