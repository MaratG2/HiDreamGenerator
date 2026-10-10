<#
.SYNOPSIS
Starts the HiDream Prompt-Refine vLLM API in Docker Desktop on Windows.

.DESCRIPTION
vLLM requires Linux. This wrapper runs the official Linux vLLM image through
Docker Desktop and mounts a local Prompt-Refine checkpoint read-only.

.EXAMPLE
.\start_vllm_server.ps1 -DryRun

.EXAMPLE
.\start_vllm_server.ps1
#>

[CmdletBinding()]
param(
    [string]$ModelPath = $env:HIDREAM_REFINER_MODEL,
    [string]$ServedModelName = "HiDream-ai/Prompt-Refine",
    [string]$DockerImage = "vllm/vllm-openai:latest",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [ValidateRange(256, 131072)]
    [int]$MaxModelLength = 8192,
    [ValidateRange(1, 64)]
    [int]$TensorParallelSize = 1,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $PSScriptRoot "HiDream-ai\Prompt-Refine"
}

if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
    throw "Prompt-Refine model directory not found: $ModelPath"
}

$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$configPath = Join-Path $ModelPath "config.json"
$templatePath = Join-Path $ModelPath "tool_chat_template_gemma4.jinja"

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Model config not found: $configPath"
}
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
    throw "Gemma 4 chat template not found: $templatePath"
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install/start Docker Desktop, then retry."
}

if (-not $DryRun) {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not running. Start it, then retry."
    }

    $dockerRuntimes = docker info --format '{{json .Runtimes}}'
    if ($LASTEXITCODE -ne 0 -or $dockerRuntimes -notmatch 'nvidia') {
        throw "Docker Desktop does not expose the NVIDIA runtime. Enable WSL 2 GPU support in Docker Desktop."
    }
}

$weightBytes = (Get-ChildItem -LiteralPath $ModelPath -Filter "*.safetensors" -File |
    Measure-Object -Property Length -Sum).Sum
if ($weightBytes -gt 0) {
    $weightGiB = [math]::Round($weightBytes / 1GB, 1)
    Write-Warning "The local checkpoint contains $weightGiB GiB of weights. The BF16 31B model needs substantially more than 16 GiB VRAM; this machine will normally run out of memory unless you use a compatible quantized checkpoint or multiple GPUs."
}

$dockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all",
    "--ipc=host",
    "-p", "${Port}:${Port}",
    "-v", "${ModelPath}:/model:ro",
    $DockerImage,
    "--model", "/model",
    "--served-model-name", $ServedModelName,
    "--host", "0.0.0.0",
    "--port", $Port.ToString(),
    "--tensor-parallel-size", $TensorParallelSize.ToString(),
    "--max-model-len", $MaxModelLength.ToString(),
    "--enable-auto-tool-choice",
    "--reasoning-parser", "gemma4",
    "--tool-call-parser", "gemma4",
    "--chat-template", "/model/tool_chat_template_gemma4.jinja",
    "--mm-processor-kwargs", '{"max_soft_tokens": 1120}'
)

if ($AdditionalArguments) {
    $dockerArgs += $AdditionalArguments
}

if ($DryRun) {
    Write-Host "Docker command (arguments are shown one per line):"
    $dockerArgs | ForEach-Object { Write-Host "  $_" }
    exit 0
}

Write-Host "Starting Prompt-Refine at http://localhost:$Port/v1"
& docker @dockerArgs
exit $LASTEXITCODE
