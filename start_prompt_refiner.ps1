<#
.SYNOPSIS
Starts the quantized HiDream Prompt-Refine model with llama.cpp on Windows.

.DESCRIPTION
The Q4_K_M Prompt-Refine checkpoint is larger than a 16 GiB GPU. This launcher
partially offloads it to the GPU for speed. Stop the refiner before loading the
HiDream image model, or pass -GpuLayers 0 for CPU-only inference.
#>

[CmdletBinding()]
param(
    [string]$ModelPath = $env:HIDREAM_REFINER_GGUF,
    [ValidateRange(0, 1000)]
    [int]$GpuLayers = 40,
    [ValidateRange(512, 131072)]
    [int]$ContextSize = 8192,
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $PSScriptRoot "HiDream-ai\Prompt-Refine-GGUF\Prompt-Refine-Q4_K_M.gguf"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Quantized Prompt-Refine model not found: $ModelPath"
}
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path

$server = Get-Command llama-server -ErrorAction SilentlyContinue
if (-not $server) {
    $wingetServer = Get-ChildItem `
        -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
        -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $wingetServer) {
        throw "llama-server was not found. Install it with: winget install --id ggml.llamacpp --exact"
    }
    $serverPath = $wingetServer.FullName
} else {
    $serverPath = $server.Source
}

$serverArgs = @(
    "--model", $ModelPath,
    "--alias", "HiDream-ai/Prompt-Refine",
    "--host", "127.0.0.1",
    "--port", $Port.ToString(),
    "--ctx-size", $ContextSize.ToString(),
    "--gpu-layers", $GpuLayers.ToString(),
    "--parallel", "1",
    "--reasoning-budget", "0",
    "--jinja"
)

if ($DryRun) {
    Write-Host $serverPath
    $serverArgs | ForEach-Object { Write-Host "  $_" }
    exit 0
}

if ($GpuLayers -gt 0) {
    Write-Warning "GPU offload shares VRAM with the image model. Stop this server before starting HiDream image generation."
}
Write-Host "Starting quantized Prompt-Refine at http://127.0.0.1:$Port/v1"
& $serverPath @serverArgs
exit $LASTEXITCODE
