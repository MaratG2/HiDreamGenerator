import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from dotenv import load_dotenv
from transformers import AutoProcessor

# Allow TF32 tensor cores for the small amount of FP32 matrix math that remains
# around the BF16/quantized model. This does not change the model's BF16 dtype.
torch.set_float32_matmul_precision("high")
load_dotenv()

try:
    import sdnq  # noqa: F401 - registers the SDNQ Transformers integration
except ImportError:
    sdnq = None

from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.pipeline import generate_image
from generation_settings import resolve_settings, validate_dimensions, number

def add_special_tokens(tokenizer):
    """Attach the special-token shortcuts that the pipeline relies on."""
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"

def get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer

def main():
    p = argparse.ArgumentParser("Standalone Pixel-DiT inference")
    p.add_argument("--model_path", type=str, default=os.environ.get("HIDREAM_MODEL_PATH"), help="Path to model checkpoint (or HIDREAM_MODEL_PATH)")
    p.add_argument("--prompt", type=str, default = "medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.")
    p.add_argument("--ref_images", nargs="*", default=[], help="Path to reference images.")
    p.add_argument(
        "--layout_bboxes",
        type=str,
        default=None,
        help=(
            "Layout boxes as a JSON string or JSON file path. "
            "Input order is xxyy relative coordinates: [[x1,x2,y1,y2], ...], "
            "[{\"bbox\": [x1,x2,y1,y2], \"text\": \"...\"}, ...], "
            "or {\"bboxes\": ...}. Example: [0.1,0.15,0.2,0.36]."
        ),
    )
    p.add_argument("--output_image", type=str, default="output.png")
    p.add_argument("--height", type=int, default=2048)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--model_type", type=str, default=os.environ.get("HIDREAM_MODEL_TYPE", "full"), choices=["full", "dev"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--noise_scale_start", type=float, default=None)
    p.add_argument("--noise_scale_end", type=float, default=None)
    p.add_argument("--noise_clip_std", type=float, default=None)
    p.add_argument(
        "--editing_scheduler",
        type=str,
        default=None,
        choices=["flow_match", "flash"],
        help="Explicit editing scheduler; otherwise use the checkpoint default.",
    )
    p.add_argument(
        "--keep_original_aspect",
        action="store_true",
        help="Preserve one reference image's proportions at max(width, height), rounded to 64 pixels.",
    )
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--scheduler_name", choices=["auto", "default", "flow_match", "flash"], default="auto")
    args = p.parse_args()
    mode = "t2i" if not args.ref_images else "edit" if len(args.ref_images) == 1 else "subject"
    try:
        if not args.model_path:
            raise ValueError("--model_path or HIDREAM_MODEL_PATH is required")
        validate_dimensions(args.width, args.height, alignment=32)
        number(args.seed, "seed", 0, 4_294_967_295, integer=True)
        if args.keep_original_aspect and mode != "edit":
            raise ValueError("--keep_original_aspect requires exactly one reference image")
        if args.editing_scheduler and mode != "edit":
            raise ValueError("--editing_scheduler requires exactly one reference image")
        settings = resolve_settings(args.model_type, mode, vars(args))
    except ValueError as error:
        p.error(str(error))

    assert torch.cuda.is_available(), "CUDA is required for inference."
    print(f"[inference] Loading processor and model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    # BF16 is required for practical inference on consumer GPUs and is also
    # the runtime dtype recommended by the SDNQ quantized checkpoints.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_image)), exist_ok=True)
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    image = generate_image(
        model=model,
        processor=processor,
        prompt=args.prompt,
        ref_image_paths=args.ref_images,
        height=args.height,
        width=args.width,
        seed=args.seed,
        keep_original_aspect=args.keep_original_aspect,
        layout_bboxes = args.layout_bboxes,
        **settings,
    )
    image.save(args.output_image)
    print(f"[inference] Saved -> {args.output_image}")

if __name__ == "__main__":
    main()
