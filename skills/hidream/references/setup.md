# Missing HiDream connection or backend

The skill is instructions; it does not bundle model weights, start a GPU service,
or install an MCP connection. Use the existing HiDream checkout and environment.
If its location is unknown and needed, ask for the checkout location rather than
assuming a developer-specific path.

## Local Codex

With the project dependencies and checkpoint already installed, run from the
HiDream repository root in PowerShell:

```powershell
.\start_image_server.ps1 -SkipFrontendBuild
```

The process must stay running. If starting it from an agent on Windows, launch it
hidden with captured logs and retain its PID; do not open an interactive window.
Honor existing model path/type configuration. Do not start a second service when
the configured API is already reachable.

Register a missing MCP connection in another terminal:

```powershell
.\scripts\install_codex_mcp.ps1
```

The installer uses the project virtual environment and absolute server path,
with a 1800-second host tool timeout. An existing connection is preserved unless
explicitly replaced with `-Force`. Reload or restart the client if newly
registered tools are not visible. For an existing misconfigured connection,
inspect its API URL and timeout before changing it.

Other local MCP hosts can launch the project's virtual-environment Python with
the absolute path to `mcp_server.py`. Set `HIDREAM_API_URL` to the running backend
(default `http://127.0.0.1:7860`) and optionally `HIDREAM_MCP_OUTPUT_DIR`. Use stdio
transport and a tool timeout long enough for GPU inference. Never print extra
messages to the MCP process's stdout. See `docs/MCP.md` in the checkout for full
setup, or `docs/WINDOWS.md` for GPU dependencies. Missing dependencies or weights
are a setup task; report them instead of repeatedly launching a broken server.

## ChatGPT and remote API clients

Installing this local skill does not connect a cloud client to the GPU. The
current adapter uses stdio. OpenAI Secure MCP Tunnel supports private stdio
servers, but requires a configured tunnel, runtime credentials, account access,
and a running tunnel client. Use the current official documentation for setup:

- [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
- [Custom MCP servers](https://developers.openai.com/api/docs/guides/custom-mcp-server)

No ChatGPT tunnel or public plugin is bundled or verified by this skill.
Remote references still need files accessible to the MCP server. Do not assume a
ChatGPT upload or a cloud sandbox path exists on the GPU machine. Public plugin
distribution additionally requires a stable public HTTPS MCP endpoint; a local
skill installation does not provide one.
