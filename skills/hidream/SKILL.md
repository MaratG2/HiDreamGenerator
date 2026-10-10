---
name: hidream
description: Generate, edit, or combine images with the local HiDream MCP server. Use for HiDream requests, local image generation or editing, and image tasks where HiDream is the user's chosen backend.
---

# HiDream images

Use the connected HiDream MCP tools for image work. Tool prefixes depend on the
host; discover the tools under the configured HiDream server instead of assuming
a particular namespace. Respect an explicitly requested provider. This skill
does not apply to reviewing HiDream source code or merely explaining the model.

## Check availability

Call `get_status` before the first generation in a task. Check API reachability,
model type, supported modes/schedulers, and `mock_mode`. The MCP adapter shares
one GPU backend with the studio; a browser does not need to be open.

For an unavailable tool or backend, read [setup guidance](references/setup.md).
Report the actual missing prerequisite if it cannot be resolved in the current
environment. Do not substitute another image service without the user's choice.
Use mock mode only for an explicitly requested integration test. If it is on
during a real-image request, disable it with `set_mock_mode(false)` before
generating; if the backend refuses because it is busy, report that condition.

## Choose the tool and inputs

| User's task | Tool |
| --- | --- |
| Create from text, no references | `generate_image` |
| Change one existing image | `edit_image` with `source_image_path` |
| Combine or preserve subjects from 2-8 images | `compose_images` with `source_image_paths` |

Reference paths must exist on the machine running the MCP server. Prefer absolute
paths; relative paths use the server's working directory. Chat attachments and
remote URLs are not automatically local files. Use the host's supported file
access to obtain a real path, or identify the missing input. Inspect references
when the host allows it. For edits, describe the requested change and what must
remain consistent. For composition, preserve input order and identify each
reference's role in the prompt.

## Preserve settings

- Pass requested width, height, seed, and other supported controls explicitly.
  Dimensions must be multiples of 64 from 512 to 4096. If an exact requested size
  is unsupported, explain the constraint instead of silently substituting a
  square or rounding the user's requested output.
- For exact edit dimensions, set `keep_original_aspect=false`. When retaining
  source proportions, set it to `true`: the backend uses `max(width, height)` as
  the target long edge, rounds to 64 pixels, and can return a different short
  edge. It preserves proportions approximately, not the original pixel count.
- When dimensions are unspecified, start at 1024x1024 for text generation. For
  edits, keep source aspect at a 1024-pixel long edge. Honor a requested format
  such as landscape or portrait when choosing supported dimensions.
- Leave `scheduler="auto"` and omit steps, guidance, shift, and noise controls
  unless requested. Defaults depend on the loaded model: full uses 50 steps,
  shift 3, and guidance 1 for edits / 5 for text and composition; dev uses 28
  steps, guidance 0, shift 1. Do not choose the
  checkpoint type through a generation argument; the backend is already loaded.
  `noise_scale_end` and `noise_clip_std` apply only to the Flash scheduler.
- All generation modes render internally at about 4 MP (2048x2048 for square images) before resizing
  to the output size. A small export need not make an edit fast. `preview_max_side`
  limits the tool preview only; it does not change the saved PNG dimensions.
- Keep prompt refinement off unless requested. It uses a separate refiner service
  and can increase memory use. If refining explicitly with `refine_prompt`, pass
  that result with `refine=false` to avoid refining twice.

## Run and recover

All real edits run through HiDream; there is no background-mask shortcut.
`patch_grid_smoothing=true` averages shifted model predictions at low noise in
creation, editing and composition. Final steps take longer. Leave it enabled
unless the user requests a controlled comparison. The actual setting is saved
in PNG/MCP metadata. It reduces patch artifacts; do not promise a perfectly
artifact-free output or pixel-identical preservation of the source.

Use one generation call at a time. For requested variations, use `batch_size`
from 1 to 4; images run sequentially with incrementing seeds. Use a new seed range
for subsequent batches, staying within 0-4294967295. Keep the same seed for a
controlled comparison unless the user wants new variation.

Leave `output_path` unset for unique filenames, or honor the user's destination.
Keep `overwrite=false` unless replacing that file was requested.

GPU runs can take several minutes. The tool defaults to 1200 seconds and accepts
up to 3600; the host's tool timeout must also allow the run. A timeout or lost
connection does not cancel the GPU job. Do not blindly retry an ambiguous failed
call: inspect the error, available status, and outputs first, and report an
unresolved job rather than enqueueing duplicates. Correct validation failures
before resubmitting; do not reduce requested quality or dimensions silently.

## Return the result

Use the returned image preview and saved PNG path. Prefer the host's native image
display; when local files are accessible, show the full image with an absolute
Markdown image path. Do not invent public download URLs for server-local files.
Report actual output dimensions and seed from the returned metadata, not the
requested values or preview size. Mention a differing render size when relevant
to settings or performance. Clearly label mock results as test placeholders.
Use the saved JSON sidecar for additional effective settings when needed.
