# Security and deployment scope

HiDreamGenerator is intended for a trusted local user. The API can spend GPU time, toggle synthetic generation and invoke the server's configured prompt refiner. MCP can read sources and write outputs using the local user's permissions. Connect only trusted MCP clients.

The default binding is loopback. JSON requests, trusted hosts, origin checks, image size limits and a bounded queue reduce accidental exposure; they do not authenticate users. Do not expose this API directly to the internet.

Shared deployments need an authenticated reverse proxy, TLS, per-user request and GPU quotas, and a production server with exactly one model-owning process. Disable SSE proxy buffering and allow only intended hosts/origins. Multiple WSGI workers have separate job registries and model copies and are unsupported.

Jobs and browser history are in memory. Server restart loses jobs; browser reload loses history. Downloaded PNGs embed prompts and settings. Review metadata before sharing an image made with a private prompt.

Never commit `.env`, credentials, local model caches or private outputs. Use GitHub's private vulnerability reporting if enabled. Do not publish credentials or an exploit against a live service in a public issue. Maintainers should enable private vulnerability reporting before advertising the repository for public use.
