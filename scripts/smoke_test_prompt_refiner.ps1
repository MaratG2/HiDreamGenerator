[CmdletBinding()]
param(
    [ValidateRange(0, 1000)]
    [int]$GpuLayers = 40,
    [ValidateRange(30, 600)]
    [int]$StartupTimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$model = (Resolve-Path (Join-Path $repoRoot "HiDream-ai\Prompt-Refine-GGUF\Prompt-Refine-Q4_K_M.gguf")).Path
$server = Get-ChildItem `
    -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
    -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $server) {
    throw "llama-server.exe was not found."
}

$stdout = Join-Path $env:TEMP "hidream-refiner-stdout.log"
$stderr = Join-Path $env:TEMP "hidream-refiner-stderr.log"
Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
$arguments = @(
    "--model", $model,
    "--alias", "HiDream-ai/Prompt-Refine",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--ctx-size", "4096",
    "--gpu-layers", $GpuLayers.ToString(),
    "--parallel", "1",
    "--reasoning-budget", "0",
    "--jinja"
)

$process = Start-Process `
    -FilePath $server.FullName `
    -ArgumentList $arguments `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

try {
    $ready = $false
    $attempts = [math]::Ceiling($StartupTimeoutSeconds / 2)
    for ($attempt = 0; $attempt -lt $attempts; $attempt++) {
        if ($process.HasExited) {
            break
        }
        try {
            $models = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8000/v1/models" `
                -TimeoutSec 3
            $ready = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    if (-not $ready) {
        Get-Content -LiteralPath $stderr -Tail 100 -ErrorAction SilentlyContinue
        throw "Prompt refiner did not become ready."
    }
    Write-Host "MODELS_ENDPOINT_OK: $($models.data[0].id)"

    $body = @{
        model = "HiDream-ai/Prompt-Refine"
        messages = @(@{
            role = "user"
            content = "Reply with exactly OK."
        })
        max_tokens = 16
        temperature = 0
    } | ConvertTo-Json -Depth 5
    $response = Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 300
    Write-Host "CHAT_COMPLETION_OK: $($response.choices[0].message.content)"
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}
