# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 CLVwp contributors
"""Measure real K/V amplitude per layer at several context lengths.

Purpose: decide the right STATIC k_scale/v_scale for fp8 KV cache on a model
whose checkpoint ships no scales (Qwen2.5-7B-Instruct-GPTQ-Int4). For each
attention layer we run one forward with a filler prompt of N tokens and
record max|K| / max|V| per layer, then compare against the KV fp8 e4m3
representable range (448) to derive a safe static scale per context budget.

Usage:
  python kv_scale_probe.py            # uses VLLM_KV_CTXS / VLLM_KV_MODEL
"""

import json
import math
import os

# MUST be set before importing vllm: keep the engine in-process so the
# module-level attention hooks below intercept the real forward path.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm import LLM, SamplingParams as sp

MODEL = os.environ.get("VLLM_KV_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
CTXS = [int(c) for c in os.environ.get("VLLM_KV_CTXS", "128,2048,8192,16384").split(",")]
OUT = os.environ.get("VLLM_KV_PROBE_OUT", "C:/AI/tmp/kv_scale_probe.json")

FILLER = (
    "The history of GPU computing begins in the late 1990s when fixed function "
    "graphics pipelines gave way to programmable shaders. Researchers noticed "
    "that vertex and pixel shaders could be tricked into running general "
    "arithmetic, and the GPGPU movement was born. In 2006 CUDA exposed a C "
    "language interface to the chip, and deep learning arrived a decade later "
    "to consume every flop the industry could produce. "
)
QUESTION = "\n\nSummarize the text above."


def main() -> None:
    llm = LLM(
        model=MODEL, dtype="float16", attention_backend="TRITON_ATTN",
        tensor_parallel_size=1, gpu_memory_utilization=0.6,
        max_model_len=max(CTXS) + 256, kv_cache_dtype="auto",
        enable_prefix_caching=False, enforce_eager=True, trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    results = {}
    for ctx in CTXS:
        reps = max(1, math.ceil(ctx / 78)) + 1
        text = FILLER * reps + QUESTION
        ids = tok.encode(text)
        if len(ids) > ctx:
            text = tok.decode(ids[: ctx - 8]) + QUESTION
        llm.generate([text], sp(max_tokens=1, ignore_eos=True))

        # walk the registered attention layers via the forward context hook:
        # after generate() the weights are in place; capture K/V by re-running
        # one layer forward is invasive, so instead hook the scale tensors the
        # same way calc_kv_scales does: monkey-patch once, rerun the prompt.
        kmax = {}

        # Hook the backend impl itself: TritonAttentionImpl.forward receives
        # the raw (q, k, v) for every layer, regardless of how the layer
        # dispatches (direct call vs opaque custom op).
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        orig_forward = TritonAttentionImpl.forward

        def patched_forward(self_impl, query, key, value, *a, **kw):
            if key is not None and value is not None:
                kmax[self_impl.layer_name] = (
                    key.abs().max().item(), value.abs().max().item())
            return orig_forward(self_impl, query, key, value, *a, **kw)

        TritonAttentionImpl.forward = patched_forward
        try:
            llm.generate([text], sp(max_tokens=1, ignore_eos=True))
        finally:
            TritonAttentionImpl.forward = orig_forward

        kmaxv = max(v[0] for v in kmax.values())
        vmaxv = max(v[1] for v in kmax.values())
        results[ctx] = {"k_max": kmaxv, "v_max": vmaxv,
                        "k_scale_safe": round(kmaxv / 448.0, 6),
                        "v_scale_safe": round(vmaxv / 448.0, 6)}
        print(f"ctx={ctx:>6}  max|K|={kmaxv:.3f}  max|V|={vmaxv:.3f}  "
              f"safe_scale k={results[ctx]['k_scale_safe']} "
              f"v={results[ctx]['v_scale_safe']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {OUT}")
    print("KV_SCALE_PROBE_OK")


if __name__ == "__main__":
    main()
