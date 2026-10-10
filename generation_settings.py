"""Shared, dependency-free generation settings for the API, CLI and pipeline."""

import math

DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]


def number(value, name, lower, upper, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a {'whole number' if integer else 'number'}")
    if value < lower or value > upper or (positive and value == 0):
        raise ValueError(f"{name} must be {'greater than' if positive else 'at least'} {lower} and at most {upper}")
    if not math.isfinite(value) or (integer and int(value) != value):
        raise ValueError(f"{name} must be a finite {'whole number' if integer else 'number'}")
    return int(value) if integer else float(value)


def validate_dimensions(width, height, *, minimum=512, alignment=64):
    for name, value in (("width", width), ("height", height)):
        number(value, name, minimum, 4096, integer=True)
        if value % alignment:
            raise ValueError(f"{name} must be a multiple of {alignment} from {minimum} to 4096")
    return int(width), int(height)


def aspect_dimensions(source_width, source_height, long_side):
    """Preserve source proportions at the requested long edge, rounded to 64px."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive")
    scale = long_side / max(source_width, source_height)
    # Explicit half-up rounding also matches JavaScript Math.round.
    size = tuple(math.floor(value * scale / 64 + 0.5) * 64 for value in (source_width, source_height))
    if min(size) < 64:
        raise ValueError("Source aspect ratio is too extreme; crop it or choose a larger resolution")
    return validate_dimensions(*size, minimum=64)


def render_dimensions(width, height, has_references):
    """Use the model's training-scale canvas; export stays exact.

    Small canvases can collapse into noise in creation as well as editing. Use the
    upstream-scale 4 MP minimum budget, preserving proportions and keeping
    either edge within 4096. The former 2 MP speed compromise is not the
    quality default. Export size must not silently reduce the model's
    working resolution. Larger outputs keep their size.
    """
    minimum_pixels = 2048 * 2048
    if width * height >= minimum_pixels:
        return width, height
    scale = min(math.sqrt(minimum_pixels / (width * height)), 4096 / max(width, height))
    return tuple(min(4096, math.ceil(value * scale / 64) * 64) for value in (width, height))


def resolve_settings(model_type, mode, data):
    if model_type not in ("full", "dev"):
        raise ValueError("model_type must be full or dev")
    default_scheduler = "default" if model_type == "full" else ("flow_match" if mode == "edit" else "flash")
    if mode != "edit" and data.get("editing_scheduler") is not None:
        raise ValueError("editing_scheduler only applies to edit mode")
    scheduler = data.get("scheduler_name")
    if scheduler in (None, "auto"):
        scheduler = data.get("editing_scheduler") if mode == "edit" else None
    scheduler = scheduler or default_scheduler
    if scheduler not in ("default", "flow_match", "flash"):
        raise ValueError("scheduler_name must be auto, default, flow_match, or flash")
    defaults = {
        "num_inference_steps": 50 if model_type == "full" else 28,
        # Strong CFG amplified texture artifacts in reference-preserving edits.
        # Keep the direct conditional prediction for edits unless overridden.
        "guidance_scale": (1.0 if mode == "edit" else 5.0) if model_type == "full" else 0.0,
        "shift": 3.0 if model_type == "full" else 1.0,
        "noise_scale_start": 8.0,
        "noise_scale_end": 8.0,
        "noise_clip_std": 0.0,
    }
    limits = {"num_inference_steps": (1, 100), "guidance_scale": (0, 20), "shift": (0, 20),
              "noise_scale_start": (0, 32), "noise_scale_end": (0, 32), "noise_clip_std": (0, 32)}
    result = {}
    for name, default in defaults.items():
        value = default if data.get(name) is None else data[name]
        result[name] = number(value, name, *limits[name], integer=name == "num_inference_steps", positive=name == "shift")
    for name in ("noise_scale_end", "noise_clip_std"):
        if scheduler != "flash" and data.get(name) is not None:
            raise ValueError(f"{name} only applies to the flash scheduler")
        if scheduler != "flash":
            result.pop(name)
    result["scheduler_name"] = scheduler
    smoothing = data.get("patch_grid_smoothing", True)
    if not isinstance(smoothing, bool):
        raise ValueError("patch_grid_smoothing must be a boolean")
    result["patch_grid_smoothing"] = smoothing
    # Keep the trained schedule when the caller explicitly selects its standard
    # step count too. Custom step counts/shift/scheduler use generated schedules.
    trained = (model_type == "dev" and result["num_inference_steps"] == 28
               and result["shift"] == 1.0
               and (scheduler == default_scheduler or (mode == "edit" and scheduler == "flash")))
    result["timesteps_list"] = list(DEFAULT_TIMESTEPS) if trained else None
    return result
