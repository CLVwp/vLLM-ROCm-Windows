# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
# Installed into the venv's site-packages by tools/patch_vllm.py. site.py auto-imports it at
# EVERY interpreter startup, including vLLM's model-inspection subprocess
# (`python -m vllm.model_executor.models.registry`, which does NOT load the platform plugin).
# Two independent fixes, each guarded so a non-vllm / non-tvm python never breaks startup.

# 1. Apply the single-process torch.distributed shim at startup, so model inspection of an
# un-cached architecture (e.g. ERNIE-4.5) does not fail with
# ModuleNotFoundError: torch._C._distributed_c10d on this USE_DISTRIBUTED=0 Windows torch.
try:
    import vllm_windows_rocm.bootstrap  # noqa: F401  (import triggers torchdist_shim.apply())
except Exception:
    pass

# 2. apache-tvm-ffi (used by xgrammar >= 0.2) loads DLLs through the ANSI API on Windows,
# which fails on non-ASCII install paths (e.g. C:\Users\PC-Clément\... -> "Failed to load
# dynamic shared library"). Convert paths to their 8.3 short form (pure ASCII) before
# tvm_ffi loads them. GetShortPathNameW yields an uppercase extension and tvm_ffi's loader
# registry is case-sensitive (.DLL != .dll), so lowercase it back.
try:
    import sys

    if sys.platform == "win32":
        from tvm_ffi import module as _tmod

        _orig_load_module = _tmod.load_module

        def _load_module_ascii(path, *args, **kwargs):
            try:
                import ctypes
                import os

                buf = ctypes.create_unicode_buffer(1024)
                if (ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024)
                        and buf.value.isascii()):
                    root, ext = os.path.splitext(buf.value)
                    path = root + ext.lower() if ext else buf.value
            except Exception:
                pass
            return _orig_load_module(path, *args, **kwargs)

        _tmod.load_module = _load_module_ascii
except Exception:
    pass
