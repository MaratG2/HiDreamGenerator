# HiDream MCP server

`mcp_server.py` exposes the running HiDream Flask service as local MCP tools.
The adapter does not import PyTorch or load a checkpoint, so the web studio and
all connected agents share the same serialized GPU generation queue.

## Skill for automatic selection

The portable [`hidream` skill](../skills/hidream/SKILL.md) teaches an agent how to
select generation, editing, or composition, preserve dimensions and model
defaults, handle long jobs without duplicate retries, and return actual output
metadata. Automatic selection is enabled for HiDream and local image requests;
the host/model still decides which skill to use. You can explicitly request
`$hidream` when needed.

From the repository root, install it for your local Codex user:

```powershell
.\scripts\install_codex_skill.ps1
```

It installs into `$CODEX_HOME/skills/hidream`, or `~/.codex/skills/hidream` when
`CODEX_HOME` is unset. Repeating an identical install is safe. To update a changed
installed copy, use `-Force`; local customizations in the three bundled files
will be replaced. `-SkillsDirectory` selects another skill root, such as
`~/.agents/skills`. Other systems can copy `skills/hidream` into their host's
discovered skill directory. Reload or restart Codex if it does not appear.

Example requests:

- "Use HiDream to generate a landscape at 1536x1024, seed 42."
- "Edit this image locally: make the teapot blue and keep its proportions."
- "Use $hidream to combine these two reference images."

Install/register MCP separately using the next section and keep the backend
running. The skill declares the `hidream` tool dependency but does not install
Python packages, download weights, or start inference automatically at install
time. MCP can also be called without the skill. Generic image requests may
select another provider; naming HiDream makes the intended backend clear.

This local installation does not add a ChatGPT cloud connection. OpenAI's
[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
supports private stdio servers, but needs its own account/tunnel configuration
and a running tunnel client. That path has not been tested with this project.
Remote image uploads also need handling because references currently use local
server paths. Public ChatGPT plugin distribution requires a stable HTTPS MCP
endpoint; the skill alone does not supply hosting or a public plugin.

## Install for Codex

Install the project requirements, start the image server, and register the MCP
adapter:

```powershell
uv pip install --python .\.venv\Scripts\python.exe `
    -r requirements-windows-cuda.txt

.\start_image_server.ps1
```

In a second PowerShell window:

```powershell
.\scripts\install_codex_mcp.ps1
```

The installer writes an stdio entry to `~/.codex/config.toml` with absolute
paths and these important timeout values:

```toml
[mcp_servers.hidream]
command = "C:\\absolute\\path\\to\\.venv\\Scripts\\python.exe"
args = ["C:\\absolute\\path\\to\\mcp_server.py"]
startup_timeout_sec = 20
tool_timeout_sec = 1800

[mcp_servers.hidream.env]
HIDREAM_API_URL = "http://127.0.0.1:7860"
HIDREAM_MCP_OUTPUT_DIR = "C:\\absolute\\path\\to\\results\\mcp"
```

Restart Codex after installation. Use `/mcp` to confirm that `hidream` is
connected, or ask Codex to call `get_status`.

To replace an existing entry or choose another API/output location:

```powershell
.\scripts\install_codex_mcp.ps1 -Force `
    -ApiUrl http://127.0.0.1:7861 `
    -OutputDirectory C:\AI\Generated\HiDream
```

The same stdio server can be launched manually for another MCP host with:

```powershell
.\start_mcp_server.ps1
```

Do not print or pipe other output into that process: stdout carries the MCP
protocol. Server diagnostics use stderr.

## Tools

### `get_status`

Checks API reachability, model type, CUDA device, attention backend, supported
modes and schedulers, and the configured MCP output directory.

### `set_mock_mode`

Explicitly enables or disables synthetic integration-test images. Mock mode is
off by default. Synthetic PNGs are visibly labelled and never use the model.

### `refine_prompt`

Runs the prompt through the configured Prompt-Refine endpoint. The separate
prompt-refiner server must be running. Prompt refinement remains opt-in on all
generation tools because the 31B refiner and image model may not fit in VRAM at
the same time on a 16 GB GPU.

### `generate_image`

Text-to-image generation. Example agent arguments:

```json
{
  "prompt": "A red origami crane on white paper, soft window light",
  "width": 1024,
  "height": 1024,
  "seed": 32,
  "output_path": "origami-crane.png"
}
```

### `edit_image`

Edits exactly one local source image. Relative input paths are resolved from the
MCP process working directory; absolute paths are safest.

```json
{
  "prompt": "Change only the background to a dark museum gallery",
  "source_image_path": "C:\\AI\\input\\chair.png",
  "keep_original_aspect": true,
  "seed": 41
}
```

### `compose_images`

Subject-driven generation from two to eight local reference images:

```json
{
  "prompt": "Place both referenced subjects in the same editorial portrait",
  "source_image_paths": [
    "C:\\AI\\input\\person.png",
    "C:\\AI\\input\\jacket.png"
  ],
  "width": 1024,
  "height": 1280
}
```

## Generation parameters

All generation tools support the relevant subset of these parameters:

| Parameter | Default | Notes |
| --- | ---: | --- |
| `width`, `height` | `1024` | Output size, 512-4096 in multiples of 64; reference generation uses a larger internal canvas when needed |
| `seed` | `32` | 0-4294967295 |
| `scheduler` | `auto` | `default`, `flow_match`, or `flash` override |
| `num_inference_steps` | model default | 1-100; the standard 28-step dev count preserves its trained schedule |
| `guidance_scale` | model default | 0-20 |
| `shift` | model default | Greater than 0, up to 20 |
| `noise_scale_start/end` | model default | Initial noise / Flash noise schedule, 0-32 |
| `noise_clip_std` | model default | Flash scheduler control, 0-32 |
| `refine` | `false` | Calls Prompt-Refine before generation |
| `output_path` | generated name | Relative paths are placed under the MCP output directory |
| `overwrite` | `false` | Refuses to replace existing PNG or metadata by default |
| `preview_max_side` | `1024` | MCP preview size; the disk PNG remains full resolution |
| `timeout_seconds` | `1200` | Per-call generation timeout, 30-3600 seconds |
| `batch_size` | `1` | 1-4 images, generated sequentially with incremented seeds |

Leaving scheduler parameters unset uses the checkpoint-specific configuration
from `app.py`. Override them only for deliberate experiments. GPU work is guarded
by one lock, so multiple agents can connect safely but generations execute one at
a time.

Each successful generation returns:

- a text summary with the absolute PNG and metadata paths;
- an `image/png` MCP content block for immediate agent inspection;
- structured metadata with mode, prompt, dimensions, seed, source files,
  effective request parameters, SHA-256, and elapsed time.

## Test without loading the model

The test suite runs a fake HTTP/SSE image service and exercises all six tools
through the SDK's in-memory MCP client:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- **Offline status:** run `.\start_image_server.ps1` and verify
  `http://127.0.0.1:7860/api/status`.
- **Tool stops near 60 seconds:** install through the supplied script or set
  `tool_timeout_sec = 1800` in the MCP table.
- **Server is missing after installation:** restart Codex; MCP configuration is
  loaded when the host starts.
- **Connection immediately closes:** use absolute command/script paths and make
  sure no wrapper writes ordinary output to stdout.
- **Prompt refinement fails:** start `start_prompt_refiner.ps1`, or call the
  generation tool with `refine=false`.
- **Output already exists:** choose another `output_path` or explicitly set
  `overwrite=true`.

## Output settings and limits

With `keep_original_aspect=true`, `max(width, height)` selects the long edge and
source proportions are rounded to 64 pixels. Both output dimensions are explicit
when that flag is false. Reference-conditioned generation renders internally at
at least roughly 4 MP (2048×2048 for square edits, up to 4096 per edge) and then resizes to the selected output; small
export sizes do not make editing faster. Inputs must be
PNG/JPEG/WebP, no larger than 24 MB or 24 megapixels each, within the API's 64 MB
request limit. Batches must not overflow the maximum seed.

JSON sidecars distinguish requested `parameters` from `effective_parameters` and
include actual PNG dimensions, `render_width`, `render_height`, `model_type`, `checkpoint` and `mock_mode`. Checkpoint metadata includes its publisher ID, source revision (when available), quantization and variant. The latter fields
are null with an older server lacking embedded metadata. Current mock outputs
are explicitly labelled in the tool response. No-overwrite file creation is
atomic. Existing explicit batch destinations are checked before generation starts.

The API retains completed SSE events up to 15 minutes (bounded to 16 jobs).
Closing a call or hitting its timeout does not cancel an accepted GPU job.
