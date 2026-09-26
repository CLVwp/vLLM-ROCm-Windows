# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Importing this module installs the single-process torch.distributed shim.

A .pth startup hook (`import vllm_windows_rocm.bootstrap`) makes this run before any
`import vllm`, so vLLM sees a working (single-process) torch.distributed from the start.
"""
import os

from .torchdist_shim import apply

# Keep the validated torch/Triton fallbacks as the default for Windows ROCm entrypoints.
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")

apply()
