# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 CLVwp. Apache-2.0 (see LICENSE).
# Automated bootstrap: SETUP.md steps 3, 4 and 5 in one idempotent script.
#
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1 [-Base C:\AI] [-Venv C:\AI\vllm-venv] [-SkipVllm] [-SkipKernels]
#
# Idempotent: safe to re-run. The venv, the vllm clone and the plugin install are
# guarded; the pip installs are declarative; the kernel build is incremental-friendly.
#
# Golden rule (issue #3): venv + caches + tmp live under an ACCENT-FREE path ($Base).
# A Windows profile with accented characters (C:\Users\René...) breaks tvm-ffi/triton.
param(
    [string]$Base = "C:\AI",            # accent-free root: tmp dirs (venv path = -Venv)
    [string]$Venv = "C:\AI\vllm-venv",
    [switch]$SkipVllm,                  # torch/deps only (steps 3)
    [switch]$SkipKernels                # NOT recommended: without it, GPTQ decode collapses
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

# ===== ASCII working dirs =====
New-Item -ItemType Directory -Force -Path $Base, "$Base\tmp" | Out-Null
$env:TMP = "$Base\tmp"
$env:TEMP = "$Base\tmp"

# ===== 1. venv (never the system Python) =====
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    Write-Host "=== Creating venv: $Venv ===" -ForegroundColor Cyan
    python -m venv $Venv
}
$py = "$Venv\Scripts\python.exe"
# pip/wheel: latest is fine. setuptools: held inside the range vLLM v0.19.1 declares
# in its pyproject (>=77.0.3,<81.0.0 for python>3.11). A blanket -U upgrade drifts an
# existing venv out of that range and pip's resolver then reports a permanent conflict;
# a too-new setuptools also sits under every future cpp_extension/HIP kernel build.
& $py -m pip install -U pip wheel
& $py -m pip install "setuptools>=77.0.3,<81.0.0"

# ===== 2. Pinned stack (requirements-win-rocm.txt, ~5 GB of downloads) =====
Write-Host "=== Installing pinned torch ROCm + dependencies ===" -ForegroundColor Cyan
& $py -m pip install -r "$repo\requirements-win-rocm.txt"

# GPU sanity check (device list: note the discrete card index, Ryzen iGPUs are device 0)
& $py -c "import torch; print(torch.__version__); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

if ($SkipVllm) {
    Write-Host "SETUP_OK (deps only; -SkipVllm was set, vLLM/plugin/kernels not installed)" -ForegroundColor Green
    exit 0
}

# ===== 3. vLLM v0.19.1 as a SIBLING of this repo (README contract) =====
$vllmDir = Join-Path (Split-Path -Parent $repo) "vllm"
if (-not (Test-Path $vllmDir)) {
    Write-Host "=== Cloning vLLM v0.19.1 -> $vllmDir ===" -ForegroundColor Cyan
    # longpaths: vLLM file names can exceed the 260-char Windows limit under a deep checkout path
    git -c core.longpaths=true clone --depth 1 --branch v0.19.1 https://github.com/vllm-project/vllm.git $vllmDir
}

Write-Host "=== vLLM (empty target) + plugin + patches ===" -ForegroundColor Cyan
Push-Location $vllmDir
try {
    $env:VLLM_TARGET_DEVICE = "empty"
    & $py -m pip install -e . --no-build-isolation
} finally { Pop-Location }
& $py -m pip install -e "$repo\windows_rocm_plugin"
& $py "$repo\tools\patch_vllm.py" $vllmDir

# ===== 4. Native HIP kernels (SETUP.md step 5, ~1 min) =====
if (-not $SkipKernels) {
    Write-Host "=== Building native kernels (experiments\vllm_c_ext) ===" -ForegroundColor Cyan
    $env:HIP_PATH = "$Venv\Lib\site-packages\_rocm_sdk_devel"
    $env:PATH = "$Venv\Scripts;" + $env:PATH
    Push-Location "$repo\experiments\vllm_c_ext"
    try { cmd /c build_run.bat } finally { Pop-Location }
}

Write-Host "SETUP_OK - try: powershell -ExecutionPolicy Bypass -File localserve.ps1" -ForegroundColor Green
