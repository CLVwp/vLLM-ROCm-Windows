# CLAUDE.md — deterministic working rules for this repo

Native Windows + AMD ROCm (RDNA3) port of vLLM. vLLM is pinned at tag `v0.19.1`
(== commit `b1388b1`; pip reports it as `0.19.2.dev0+gb1388b1fb` — that is the post-tag
dev string, not a different base).

## Environment (non-negotiable)

- Python: the repo-root venv — `.venv/Scripts/python.exe`. The system python has no torch.
- **Run scripts from `run/`, never from the repo root**: the gitignored `vllm/` clone at the
  root shadows the installed `vllm` package as a namespace package otherwise.
- If the machine has an integrated GPU (most Ryzen desktop APUs), the iGPU is HIP device 0
  and the discrete card is device 1 → `HIP_VISIBLE_DEVICES=1` (or the dGPU index) for every
  GPU run. Symptom of a job landing on the iGPU: "device kernel image is invalid".
- Engine runs are in-process only: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (every `run/` script
  sets it itself). The plugin's config-time hooks require it.
- Non-ASCII Windows usernames (accents in `C:\Users\...`) break xgrammar/tvm_ffi (ANSI DLL
  loading). This is fixed by `windows_rocm_plugin/sitecustomize.py`, installed into the venv
  by `tools/patch_vllm.py` (torch.distributed shim + tvm_ffi 8.3-short-path fix). Never
  hand-edit the venv copy — it is managed and gets overwritten by patch_vllm.py. Keeping the
  venv and HF caches under an accent-free path is the belt-and-suspenders alternative.

## Where things live

- `windows_rocm_plugin/` — pip-installed platform plugin (monkey-patches; the normal way to
  change engine behaviour). Modules are wired in `vllm_windows_rocm/platform.py`.
- `vllm/` — gitignored clone at the pin. Direct edits there MUST be captured as
  `patches/vllm/*.patch` or they are lost on re-clone.
- `patches/vllm/` — applied by `python tools/patch_vllm.py vllm` in **alphabetical filename
  order**; if patch B depends on patch A's context, name B to sort after A. After editing the
  `vllm/` tree: regenerate the patch and verify it reverse-applies before committing.
- `run/` — all bench/probe drivers (see `run/README.md` table).
- `experiments/vllm_c_ext/` — native kernel builds (`vllm_win_cache_C.pyd`,
  `vllm_win_attn_C.pyd`), built by `build_run.bat` / `build_attn_c.py`.

## Behaviour-changing work

- fp8 KV cache requires the two-step calibration (probe → offsets file → `VLLM_WIN_KV_OFFSETS`);
  never bench fp8 without it. Do not combine `VLLM_WIN_KV_OFFSETS` with `VLLM_WIN_KV_KSCALE`.
- Every engine-behaviour change ships with an executable check: bit-exact kernel check
  (`run/kv_pc_kernel_check.py` style) for Triton kernels; `bench.py` / `precision_check.py`
  for engine paths.
- `agree_frac` is run-to-run noisy (GPTQ int4 GEMM nondeterminism, amplified by greedy
  cascades). Deterministic evidence = offline attention error metrics + text coherence.
  Never report agree_frac as primary proof.
- Docs stay in sync in the same change: README.md (features/status), SETUP.md (install),
  run/README.md (scripts), patches/README.md (patch set).

## Git

- Branches: numbered triage by creation order — `NNN/<type>/<slug>` with type in
  `bugfix|feature|docs|chore`. Check `git branch` for the next free number.
- Do not push to a branch that is the head of an open PR: new commits would join that PR.
- Commits: imperative subject, body explains why; when Claude co-authored, end with
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- PRs to the upstream repo: link the issue, include measured numbers (before/after), end
  description with the Claude Code attribution line.

## Tooling gotchas (Windows)

- Shell is Git Bash: forward slashes, POSIX syntax.
- `gh pr create/edit` with inline bodies: backticks execute as command substitution — always
  pass `--body-file` instead.
- `git -C vllm apply` needs `--ignore-whitespace` (CRLF vs LF), which patch_vllm.py already does.
