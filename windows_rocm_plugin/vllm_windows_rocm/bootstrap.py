# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Importing this module installs the single-process torch.distributed shim.

A .pth startup hook (`import vllm_windows_rocm.bootstrap`) makes this run before any
`import vllm`, so vLLM sees a working (single-process) torch.distributed from the start.
"""
from .torchdist_shim import apply

apply()
