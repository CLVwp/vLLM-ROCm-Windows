# SPDX-License-Identifier: Apache-2.0
# Serve a local model on an AMD GPU (ROCm) + browser chat UI, one command.
# Usage: powershell -ExecutionPolicy Bypass -File localserve.ps1 [-Model "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"] [-Port 8000]
param(
    [string]$Model = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    [int]$Port = 8000,
    [string]$Venv = "C:\AI\vllm-venv"
)

$ErrorActionPreference = "Stop"

# ===== ASCII paths (mandatory: an accented Windows profile breaks tvm-ffi/triton, issue #3) =====
$env:HIP_VISIBLE_DEVICES = "1"   # device 0 = iGPU on most AMD APUs/desktops; the dGPU is 1
$env:ROCM_HOME = "$Venv\Lib\site-packages\_rocm_sdk_devel"
$env:TMP  = "C:\AI\tmp"
$env:TEMP = "C:\AI\tmp"

# Caches outside the user profile (accent-free paths)
$env:HF_HOME = "C:\AI\hf-home"
$env:VLLM_CACHE_ROOT = "C:\AI\vllm-cache"
$env:TRITON_CACHE_DIR = "C:\AI\triton-cache"
$env:TORCHINDUCTOR_CACHE_DIR = "C:\AI\inductor-cache"
$env:VLLM_WIN_BUILD_ROOT = "C:\AI\build"   # scratch dirs of the native kernel build (vw_cext_*)

foreach ($d in @($env:TMP, $env:HF_HOME, $env:VLLM_CACHE_ROOT, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# ===== vLLM behaviour on Windows ROCm (validated defaults, all overridable) =====
$env:VLLM_ENABLE_V1_MULTIPROCESSING = "0"
$env:VLLM_ROCM_USE_SKINNY_GEMM = "0"
$env:VLLM_ROCM_USE_AITER = "0"

Write-Host "=== vLLM serve: $Model on port $Port (GPU device $($env:HIP_VISIBLE_DEVICES)) ===" -ForegroundColor Cyan

$serve = Start-Process -FilePath "$Venv\Scripts\vllm.exe" -ArgumentList @(
    "serve", $Model, "--port", "$Port", "--dtype", "float16",
    "--enforce-eager", "--attention-backend", "TRITON_ATTN",
    "--gpu-memory-utilization", "0.6"
) -NoNewWindow -PassThru

# Wait for the server to answer (allow 10 min for the first load/download)
Write-Host "Waiting for the server (the first run may download the model)..." -NoNewline
$up = $false
foreach ($i in 1..120) {
    Start-Sleep -Seconds 5
    Write-Host "." -NoNewline
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
}
Write-Host ""
if (-not $up) {
    Write-Host "ERROR: the server did not answer within 10 minutes. See the logs above." -ForegroundColor Red
    exit 1
}
Write-Host "Server up on http://127.0.0.1:$Port" -ForegroundColor Green

# ===== UI =====
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot
try {
    & "$Venv\Scripts\python.exe" .\run\chat_ui.py
} finally {
    Pop-Location
    if (-not $serve.HasExited) { Stop-Process -Id $serve.Id -Force }
    Write-Host "Server stopped." -ForegroundColor Yellow
}
