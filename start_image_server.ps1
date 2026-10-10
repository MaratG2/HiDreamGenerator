<# Starts the memory-optimized HiDream web UI on Windows. #>

[CmdletBinding()]
param(
    [string]$ModelPath,
    [ValidateSet("full", "dev")]
    [string]$ModelType = "full",
    [ValidateSet("auto", "flash", "sdpa")]
    [string]$AttentionBackend = "auto",
    [switch]$SkipFrontendBuild,
    [switch]$DisableQuantizationCompile,
    [ValidateRange(1, 65535)]
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment not found. Install requirements-windows-cuda.txt first."
}
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $PSScriptRoot "HiDream-ai\HiDream-O1-Image-SDNQ-4bit-low-vram"
}
if (-not (Test-Path -LiteralPath (Join-Path $ModelPath "config.json") -PathType Leaf)) {
    throw "HiDream checkpoint not found: $ModelPath"
}
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path

$frontend = Join-Path $PSScriptRoot "frontend"
if (-not $SkipFrontendBuild -and (Test-Path -LiteralPath (Join-Path $frontend "package.json"))) {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) {
        throw "npm.cmd was not found. Install Node.js 22+ or use -SkipFrontendBuild."
    }
    Push-Location $frontend
    try {
        if (-not (Test-Path -LiteralPath "node_modules" -PathType Container)) {
            & $npm.Source ci
            if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
        }
        & $npm.Source run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend production build failed." }
    }
    finally {
        Pop-Location
    }
}

$env:PYTORCH_CUDA_ALLOC_CONF = ""
$cudaCheck = & $python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA-enabled PyTorch is not available in .venv."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HIDREAM_ATTENTION_BACKEND = $AttentionBackend
if ($DisableQuantizationCompile) {
    $env:SDNQ_USE_TORCH_COMPILE = "0"
    $env:SDNQ_USE_TRITON_MM = "0"
}

Write-Host "Starting HiDream at http://127.0.0.1:$Port (attention: $AttentionBackend)"
& $python (Join-Path $PSScriptRoot "app.py") `
    --model_path $ModelPath `
    --model_type $ModelType `
    --host 127.0.0.1 `
    --port $Port
exit $LASTEXITCODE
