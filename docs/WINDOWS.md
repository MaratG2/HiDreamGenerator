# Windows setup and hardware notes

These are existing workstation setup notes. Earlier speed measurements are not
cross-platform benchmarks. Start with automatic attention selection or SDPA; the
specialized FlashAttention build is optional. Create `.venv` first with
`python -m venv .venv`.

## Windows quick start (16 GB NVIDIA GPU)

The full-precision checkpoints do not fit comfortably on a 16 GB GPU. This
repository is configured to use a CUDA 13.0 PyTorch build and the SDNQ 4-bit
HiDream checkpoint on Windows. Install the native build tools once from an
elevated PowerShell window:

```powershell
winget install --id Nvidia.CUDA --exact --version 13.0
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact `
    --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Kitware.CMake --exact
```

Open a new terminal, then install the Python dependencies and the specialized
FlashAttention wheel:

```powershell
uv pip install --python .\.venv\Scripts\python.exe `
    -r requirements-windows-cuda.txt

.\scripts\install_flash_attn_windows.ps1
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The FlashAttention build is a one-time operation and takes about 20 minutes on
this machine. Its wheel is retained under `.build/` for reuse. The verified
PyTorch output on an RTX 5060 Ti is `2.13.0+cu130 True`. Download the recommended
image checkpoint if it is not present:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('WaveCut/HiDream-O1-Image-SDNQ-4bit-dynamic-uint4-th1e-2', local_dir=r'HiDream-ai\HiDream-O1-Image-SDNQ-4bit')"
```

Then launch the web UI:

```powershell
.\start_image_server.ps1
```

The launcher builds the TypeScript/React **HiDreamGenerator** frontend and serves
it with the Flask API at `http://127.0.0.1:7860`. Studio supports text-to-image,
single-image edits, nine aspect ratios, 1K/1.5K/2K long-edge resolution,
four quality presets, batches of one to four, prompt refinement, live SSE
progress previews, copy/download actions, and in-memory session history. Batch
slots are allocated before generation begins, so every queued, active, completed,
or failed image remains visible in a stable contact-sheet grid. Requested output
dimensions are honored exactly; for example, the 1K square preset is
`1024x1024` and is no longer snapped to a larger model-training resolution. The
original single-file Flask interface remains available at
`http://127.0.0.1:7860/legacy`.

For frontend-only development, keep the Flask server running and start Vite in
a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Vite opens on `http://127.0.0.1:5173` and proxies `/api` to port 7860. See
`frontend/README.md` for the frontend commands and design references.

## MCP for Codex and other agents

The repository includes a local stdio MCP server in `mcp_server.py`. It is a
small client of the Flask API, so it does not load another copy of HiDream or
consume additional model VRAM. Start the image server first:

```powershell
.\start_image_server.ps1
```

Register the MCP server in Codex once, then restart Codex:

```powershell
.\scripts\install_codex_mcp.ps1
```

The installer uses absolute paths, configures `HIDREAM_API_URL`, stores outputs
under `results\mcp`, and raises Codex's tool timeout to 30 minutes for slow GPU
jobs. If an entry named `hidream` already exists, explicitly replace it with
`-Force`. After restarting, `/mcp` should show these tools:

- `get_status`
- `set_mock_mode`
- `refine_prompt`
- `generate_image`
- `edit_image`
- `compose_images`

Generation tools return an MCP image preview while saving the full PNG and a
JSON metadata sidecar. They support width, height, seed, scheduler, step count,
guidance scale, scheduler shift, Flash noise controls, prompt refinement, batches of one to four,
explicit output paths, overwrite policy, preview sizing, and request timeout.
The default 1024x1024 canvas is intended for faster agent iteration.

Mock generation is disabled by default (`HIDREAM_MOCK_MODE=0`). It can be
explicitly toggled in Studio's Runtime settings or through `set_mock_mode` for
integration testing; every synthetic result is visibly labelled as mock output.

See [MCP.md](MCP.md) for tool examples, manual configuration,
environment variables, and troubleshooting. The implementation targets the
official [MCP Python SDK 2.0](https://py.sdk.modelcontextprotocol.io/) and
Codex's documented local stdio MCP configuration.

This checkpoint uses `--model_type full` and supports text-to-image, editing,
and subject-driven generation. The default is now the balanced dynamic SDNQ
variant. It retains BF16 decoder output/down projections throughout the model;
the previous lowest-VRAM variant protected only the final eight layers. Both
retain all 36 decoder layers and use the full model's 50-step defaults.

The loader keeps the unused output head and the input vocabulary table on CPU
for untied checkpoints (about 2.3 GiB total). Prompt lookup runs on CPU and sends
only the selected rows to CUDA. The decoder's tokenwise MLP uses chunks of 2048
tokens during inference to reduce temporary memory; training is unchanged. A
BF16 GPU check at the model's actual layer dimensions saved 384 MiB at an
8192-token sequence length and produced identical output. These changes preserve
GPU space for the less aggressive checkpoint and image activations.
Edits now use an approximately 4 MP internal canvas, including 2048×2048 for
square sources, before resizing to the requested export. Allow several minutes.
An older `.env` may still select `HiDream-O1-Image-SDNQ-4bit-low-vram`; update
`HIDREAM_MODEL_PATH` to the path above when launching `app.py` directly. The
PowerShell launcher uses the new default unless `-ModelPath` is explicit.

A 50-step portrait-background edit at 2048×2048 internally / 1024×1024 export
showed that increasing resolution alone did not remove the low-VRAM checkpoint's
texture artifacts. The balanced checkpoint improved the result, but guidance 5
still exaggerated texture. Guidance 1 produced a substantially cleaner final
image and is now the default for full-model single-image edits. Text generation
and multi-reference composition retain guidance 5; explicit overrides are honored.
The final guidance-1 edit took 242 seconds on the RTX 5060 Ti; Windows GPU usage
was observed around 13.3–13.7 GiB during the balanced-checkpoint tests. These are
measurements for one example, not a general speed or quality guarantee. The
generated background remained dark blue-gray rather than exact RGB black;
generative editing does not guarantee a flat color or unchanged source pixels.
Progress previews now show the guided prediction instead of the cleaner
conditional branch that previously concealed this difference.

### Native patch-grid correction

The background-mask shortcut has been removed. All image requests use HiDream.
The default sampler now realigns and averages four denoiser predictions during
its final low-noise steps. A matching single-image reference shifts with the
target, avoiding the detail loss caused by misaligned reference conditioning.
No segmentation, source-pixel compositing, output blur or saturation filter is
applied. Text-only generation now also uses the roughly 4 MP internal canvas.

Completed local checks used the balanced full checkpoint on the RTX 5060 Ti:

- A 50-step, guidance-1 portrait edit took 254 seconds. In a flat 128×128
  background crop, the repeating-grid RMS fell from 0.467 in the earlier
  guidance-1 output to 0.152. This is a crop-specific reduction, not a guarantee
  that every artifact disappears.
- A same-prompt/seed 2048×2048 text-generation comparison visibly reduced the
  grid on both the background and objects. The unchanged early model predictions
  were cached for the second run; its elapsed time is not a full-run benchmark.
- The color drift remains unresolved: the portrait's sampled face saturation
  stayed near 0.38 versus about 0.42 in the source. Increasing guidance did not
  correct it, so edit guidance remains 1. The generated background also remains
  dark blue-gray rather than exact RGB black.

The full checkpoint remains the default. Alternative checkpoints and a proposed
lower-noise final sampling step have not been validated in this environment and
are not enabled. The retained change reduces native grid artifacts; it does not
claim to fix all color drift.

### Windows acceleration and measured speed

The upstream [`flash-attn`](https://github.com/Dao-AILab/flash-attention) package
does not publish a Windows wheel. The installer in `scripts/` builds a local
FlashAttention 2.8.3 wheel for this exact inference path: CUDA `sm_120`, BF16,
and head dimension 128. It intentionally rejects FP16, training, other head
dimensions, and non-RTX-50-series GPUs; it is not a general FlashAttention
wheel. Use `HIDREAM_ATTENTION_BACKEND=sdpa` or
`.\start_image_server.ps1 -AttentionBackend sdpa` for the portable fallback.

`triton-windows==3.7.1.post27` is also pinned. Without it, SDNQ prints that it is
falling back to PyTorch Eager mode, which slows every denoising step. The first
generation after installation is slower while Triton compiles and caches its
kernels. Run the verifier at any time:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_windows_acceleration.py
```

Earlier text-to-image 2048×2048 tests on the RTX 5060 Ti, using the low-VRAM checkpoint
loaded once and the same prompt/seed, measured:

| Backend | Seconds per step | Change |
| :--- | ---: | ---: |
| cuDNN SDPA + SDNQ eager | 7.93 | baseline |
| FlashAttention + SDNQ eager | 6.72 | 15% faster |
| FlashAttention + Triton SDNQ | 5.55 | 30% faster |
| FlashAttention + Triton SDNQ + TF32 | 5.42 | 32% faster |

That projected approximately 4 minutes 30 seconds for 50 text-to-image steps,
excluding checkpoint load and the first Triton JIT. These are historical
low-VRAM text-generation measurements, not timings for the new edit defaults.
Reference editing processes extra image tokens. FlashAttention helps, but
the quantized model's linear layers remain the main cost; a 28-step distilled
checkpoint is the next meaningful route to lower latency.

### Quantized Prompt-Refine on Windows

The official Prompt-Refine checkpoint is 31B BF16 and cannot fit on this GPU.
Use its Q4_K_M GGUF conversion with the native llama.cpp server instead:

```powershell
winget install --id ggml.llamacpp --exact

.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('tuolaku/Prompt-Refine-GGUF', 'Prompt-Refine-Q4_K_M.gguf', local_dir=r'HiDream-ai\Prompt-Refine-GGUF')"

.\start_prompt_refiner.ps1
```

The launcher partially offloads the model to the GPU because the 17.4 GiB Q4
model is too large for full GPU residency. The refiner and image model cannot
reside in 16 GiB VRAM together, so stop one before starting the other. Adjust
the split with `-GpuLayers N` (`0` means CPU-only). The API remains
OpenAI-compatible at `http://127.0.0.1:8000/v1` and is served under the model
name `HiDream-ai/Prompt-Refine`.
