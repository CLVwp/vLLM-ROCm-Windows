# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Insert the windows_rocm_rocm bootstrap import at the top of vLLM's package __init__,
so the Windows/ROCm compatibility shims are installed before any vLLM submodule loads
torch.distributed. Idempotent.

Also applies the repo's vLLM source patches (patches/vllm/*.patch) to the checkout when
git is available, so a fresh clone reaches the documented behaviour without manual
`git -C vllm apply` steps. Patching is skipped (with a notice) when the checkout is not
a git work tree; already-applied patches are detected and skipped, making re-runs safe.

Usage: python tools/patch_vllm.py [path-to-vllm-checkout]   (default: ./vllm)
"""
import subprocess
import sys
import os

MARK = "vllm_windows_rocm.bootstrap"
BLOCK = (
    "\n# --- vLLM-on-Windows-ROCm: install the single-process torch.distributed shim and\n"
    "# _C op fallbacks before any vllm submodule that imports torch.distributed is loaded.\n"
    "try:\n"
    "    import vllm_windows_rocm.bootstrap  # noqa: F401\n"
    "except Exception:\n"
    "    pass\n"
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_apply(vllm_root: str, patch_path: str, *extra: str) -> subprocess.CompletedProcess:
    """git apply with the flags that make Windows-checkout patches behave:
    --ignore-whitespace (CRLF vs LF mismatches between checkout and patch files)."""
    return subprocess.run(
        ["git", "-C", vllm_root, "apply", "--ignore-whitespace", "--whitespace=nowarn",
         *extra, patch_path],
        capture_output=True, text=True,
    )


def _patch_already_applied(vllm_root: str, patch_path: str) -> bool:
    """True when reverse-applying cleanly succeeds, i.e. the patch is already in the tree."""
    return _git_apply(vllm_root, patch_path, "--check", "-R").returncode == 0


def apply_source_patches(vllm_root: str) -> int:
    patch_dir = os.path.join(_repo_root(), "patches", "vllm")
    if not os.path.isdir(patch_dir):
        return 0
    patches = sorted(f for f in os.listdir(patch_dir) if f.endswith(".patch"))
    if not patches:
        return 0
    probe = subprocess.run(
        ["git", "-C", vllm_root, "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
    )
    if probe.returncode != 0:
        print(f"notice: {vllm_root} is not a git work tree; skipping "
              f"{len(patches)} source patch(es) — apply them manually per patches/README.md")
        return 0
    failures = 0
    for name in patches:
        p = os.path.join(patch_dir, name)
        if _patch_already_applied(vllm_root, p):
            print(f"already applied: {name}")
            continue
        r = _git_apply(vllm_root, p)
        if r.returncode == 0:
            print(f"applied: {name}")
        else:
            failures += 1
            print(f"FAILED to apply {name}: {r.stderr.strip()}", file=sys.stderr)
    return 1 if failures else 0


def install_sitecustomize(venv_site: str) -> None:
    """Copy sitecustomize.py into site-packages so EVERY interpreter start (including
    vLLM's model-inspection subprocess `python -m vllm.model_executor.models.registry`,
    which does not load the platform plugin) applies the torch.distributed shim.
    Without it, `vllm serve` fails on un-cached architectures with
    ModuleNotFoundError: torch._C._distributed_c10d. Idempotent by content hash."""
    src = os.path.join(_repo_root(), "windows_rocm_plugin", "sitecustomize.py")
    if not os.path.isfile(src):
        print(f"notice: {src} not found; skipping sitecustomize install")
        return
    dst = os.path.join(venv_site, "sitecustomize.py")
    try:
        same = os.path.isfile(dst) and open(src, "rb").read() == open(dst, "rb").read()
    except OSError:
        same = False
    if same:
        print("sitecustomize: already installed")
        return
    try:
        import shutil

        shutil.copyfile(src, dst)
        print(f"sitecustomize: installed to {dst}")
    except OSError as e:
        print(f"notice: could not install sitecustomize ({e}); `vllm serve` may fail on "
              "model-architecture inspection. Copy windows_rocm_plugin/sitecustomize.py "
              "to your venv's site-packages manually.")


def _venv_site_packages() -> str | None:
    """site-packages of the python running this script (venv or not)."""
    import sysconfig

    p = sysconfig.get_paths().get("purelib")
    return p


def main(argv):
    vllm_root = argv[0] if argv else "vllm"
    init_py = os.path.join(vllm_root, "vllm", "__init__.py")
    if not os.path.isfile(init_py):
        print(f"not found: {init_py}", file=sys.stderr)
        return 1
    site = _venv_site_packages()
    if site:
        install_sitecustomize(site)
    src = open(init_py, encoding="utf-8").read()
    if MARK in src:
        print("already patched")
    else:
        anchor = "from .version import __version__"
        idx = src.find(anchor)
        if idx == -1:
            # fall back: prepend
            new = BLOCK + src
        else:
            eol = src.find("\n", idx)
            new = src[: eol + 1] + BLOCK + src[eol + 1 :]
        open(init_py, "w", encoding="utf-8", newline="\n").write(new)
        print("patched", init_py)
    return apply_source_patches(vllm_root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
