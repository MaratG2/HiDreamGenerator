# HiDreamGenerator frontend

TypeScript/React interface for the Flask generation API in `../app.py`.

## Local bundled app

From the repository root, run:

```powershell
.\start_image_server.ps1
```

The launcher installs frontend dependencies when needed, creates `frontend/dist`,
loads the image model, and serves the Studio and API together at
`http://127.0.0.1:7860`.

## Frontend-only development

Start the Flask backend with its existing production bundle, then run Vite in a
second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to
`http://127.0.0.1:7860`; override that with `HIDREAM_BACKEND_URL` when needed.

Useful commands:

```powershell
npm run lint
npm run build
```

The original single-file Flask UI remains available at `/legacy`.

The Studio keeps aspect ratio and resolution separate, maps quality presets to
12/20/32/50 full-model steps or 12/20/28/50 dev steps, and runs batches sequentially to respect the
single-GPU queue. Every batch item gets a persistent grid slot and individual
queued/rendering/complete/error/skipped progress, while completed images remain visible
as later items run. The 1K square preset exports exactly `1024x1024`.
Small reference-conditioned jobs render internally at roughly 2 MP for model quality,
then resize to the chosen output size; the UI explains this before generation.
Runtime settings
include an explicit mock toggle. It is off by default; enable it only when
testing the UI without inference.

Source-aspect editing keeps Resolution enabled and shows resolved dimensions.
History uses actual dimensions and mock status, retains at most 40 images, and
cannot interrupt an active batch. SSE reconnects use event IDs. Reload clears history.
