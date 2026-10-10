<#
.SYNOPSIS
Build and install the HiDream-specific FlashAttention wheel on Windows.

.DESCRIPTION
The upstream project does not publish Windows wheels. A standard source build
compiles 72 CUDA files, most of which HiDream inference cannot use. This script
builds FlashAttention 2 for BF16, head_dim=128, and sm_120 only. The resulting
wheel is for this repository's RTX 50-series inference path, not general use.
#>

[CmdletBinding()]
param(
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$verify = Join-Path $PSScriptRoot "verify_windows_acceleration.py"
$version = "2.8.3.post1"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment not found: $python"
}

if (-not $ForceRebuild) {
    & $python $verify
    if ($LASTEXITCODE -eq 0) {
        Write-Host "The HiDream FlashAttention build is already installed and valid."
        exit 0
    }
    Write-Host "Existing acceleration check failed; building the local wheel."
}

$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
    throw "CUDA nvcc was not found at $nvcc. Install CUDA Toolkit 13.0."
}

$vcRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvars = Join-Path $vcRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
    throw "VS 2022 C++ Build Tools were not found at $vcRoot."
}
$cl = Get-ChildItem -LiteralPath (Join-Path $vcRoot "VC\Tools\MSVC") `
    -Recurse -Filter cl.exe |
    Where-Object { $_.FullName -match "\\bin\\Hostx64\\x64\\cl\.exe$" } |
    Sort-Object FullName -Descending |
    Select-Object -First 1
if (-not $cl) {
    throw "The VS 2022 x64 compiler (cl.exe) was not found."
}

$buildRoot = Join-Path $root ".build\flash-attn-hidream-$version"
$downloadRoot = Join-Path $buildRoot "downloads"
$sourceParent = Join-Path $buildRoot "source"
$sourceRoot = Join-Path $sourceParent "flash_attn-$version"
$wheelRoot = Join-Path $buildRoot "wheels"
New-Item -ItemType Directory -Force -Path $downloadRoot, $sourceParent, $wheelRoot | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "setup.py") -PathType Leaf)) {
    & $python -m pip download "flash-attn==$version" --no-deps `
        --no-binary=:all: --dest $downloadRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not download the FlashAttention sdist." }
    $archive = Get-ChildItem -LiteralPath $downloadRoot -Filter "flash_attn-$version.tar.gz" |
        Select-Object -First 1
    if (-not $archive) { throw "Downloaded FlashAttention archive was not found." }
    & tar.exe -xf $archive.FullName -C $sourceParent
    if ($LASTEXITCODE -ne 0) { throw "Could not extract the FlashAttention sdist." }
}

$setupPath = Join-Path $sourceRoot "setup.py"
$switchPath = Join-Path $sourceRoot "csrc\flash_attn\src\static_switch.h"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$setup = [IO.File]::ReadAllText($setupPath).Replace("`r`n", "`n")

if (-not $setup.Contains("hidream_inference_sources")) {
    $setup = $setup.Replace("-std=c++17", "-std=c++20")
    $setup = $setup.Replace("/std:c++17", "/std:c++20")

    $sourceStart = @'
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
'@
    $sourceStart = $sourceStart.Replace("`r`n", "`n")
    $replacementStart = @'
    # HiDream casts Q/K/V to BF16 and its image checkpoint uses head_dim=128.
    def hidream_inference_sources(sources):
        required = {
            "csrc/flash_attn/flash_api.cpp",
            "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
            "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
        }
        return [source for source in sources if source in required]

    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=hidream_inference_sources([
'@
    $replacementStart = $replacementStart.Replace("`r`n", "`n")
    if (-not $setup.Contains($sourceStart)) { throw "FlashAttention setup.py layout changed." }
    $setup = $setup.Replace($sourceStart, $replacementStart)

    $sourceEnd = @'
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
'@
    $sourceEnd = $sourceEnd.Replace("`r`n", "`n")
    $replacementEnd = @'
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ]),
            extra_compile_args={
'@
    $replacementEnd = $replacementEnd.Replace("`r`n", "`n")
    if (-not $setup.Contains($sourceEnd)) { throw "FlashAttention source list changed." }
    $setup = $setup.Replace($sourceEnd, $replacementEnd)
    [IO.File]::WriteAllText($setupPath, $setup, $utf8NoBom)

    $switches = [IO.File]::ReadAllText($switchPath).Replace("`r`n", "`n")
    $macroIndex = $switches.IndexOf("#define FP16_SWITCH")
    if ($macroIndex -lt 0) { throw "FlashAttention dispatch macros were not found." }
    $minimalMacros = @'
#define FP16_SWITCH(COND, ...)                                      \
  [&] {                                                             \
    TORCH_CHECK(!(COND), "This HiDream build supports BF16 only"); \
    using elem_type = cutlass::bfloat16_t;                           \
    return __VA_ARGS__();                                           \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)                                      \
  [&] {                                                                  \
    TORCH_CHECK((HEADDIM) == 128, "This HiDream build supports head_dim=128 only"); \
    constexpr static int kHeadDim = 128;                                 \
    return __VA_ARGS__();                                                \
  }()
'@
    $minimalMacros = $minimalMacros.Replace("`r`n", "`n") + "`n"
    $switches = $switches.Substring(0, $macroIndex) + $minimalMacros
    [IO.File]::WriteAllText($switchPath, $switches, $utf8NoBom)
}

& $python -m pip install "ninja==1.13.0" packaging psutil
if ($LASTEXITCODE -ne 0) { throw "Could not install FlashAttention build dependencies." }

$environment = & cmd.exe /d /s /c "`"$vcvars`" >nul && set"
foreach ($line in $environment) {
    if ($line -match "^([^=]+)=(.*)$") {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}
$env:PATH = "$(Split-Path -Parent $cl.FullName);$(Split-Path -Parent $python);$(Join-Path $CudaRoot 'bin');$env:PATH"
$env:CC = $cl.FullName
$env:CXX = $cl.FullName
$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:FLASH_ATTENTION_FORCE_BUILD = "TRUE"
$env:FLASH_ATTN_CUDA_ARCHS = "120"
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:MAX_JOBS = "2"
$env:NVCC_THREADS = "1"
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"

& $python -m pip wheel $sourceRoot --no-build-isolation --no-deps --wheel-dir $wheelRoot
if ($LASTEXITCODE -ne 0) { throw "The specialized FlashAttention build failed." }
$wheel = Get-ChildItem -LiteralPath $wheelRoot -Filter "flash_attn-$version-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $wheel) { throw "The built FlashAttention wheel was not found." }
& $python -m pip install --force-reinstall --no-deps $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "The specialized FlashAttention wheel could not be installed." }

& $python $verify
exit $LASTEXITCODE
