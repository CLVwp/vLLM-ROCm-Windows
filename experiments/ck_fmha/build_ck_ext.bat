@echo off
call "E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set ROCM_HOME=C:\HIP-SDK
set HIP_PATH=C:\HIP-SDK
set ROCM_PATH=C:\HIP-SDK
if exist C:\vw_ckfmha_build rmdir /s /q C:\vw_ckfmha_build
python -u "C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\experiments\ck_fmha\build_ck_ext.py"
