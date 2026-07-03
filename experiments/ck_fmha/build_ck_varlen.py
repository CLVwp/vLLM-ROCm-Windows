"""Build ck_fmha_varlen_C.pyd: ck_fmha_varlen(...) wrapping the CK ck_tile FMHA d128-fp16 GROUP-mode
(varlen) nlogits causal + non-causal instances for gfx1100. Same recipe as build_ck_ext.py, group instances."""
import os, shutil, time, torch
from torch.utils import cpp_extension
from torch.utils.hipify import hipify_python as _hp
_orig = _hp.hipify
def _no_none(*a, **k):
    r = _orig(*a, **k)
    try:
        for key, v in r.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception:
        pass
    return r
_hp.hipify = _no_none

CK = r"C:\Users\filip\Desktop\composable_kernel"
FMHA = os.path.join(CK, "example", "ck_tile", "01_fmha")
GEN = r"C:\Users\filip\Desktop\ckfmha_gen"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "build_src_varlen")
BUILD_DIR = r"C:\vw_ckvarlen_build"
os.makedirs(SRC, exist_ok=True); os.makedirs(BUILD_DIR, exist_ok=True)

P = "fmha_fwd_d128_{dt}_group_b128x64x32x128x32x128_r8x1x1_r8x1x1_w16x16x16_w16x16x16_o6_qr_hpad_vr_psskddv_nlogits_nbias_{m}_nlse_ndropout_nskip_nqscale_ntrload_nsink_gfx11.cpp"
INSTS = {
    "inst_c":    P.format(dt="fp16", m="mask"),
    "inst_nc":   P.format(dt="fp16", m="nmask"),
    "inst_c_bf": P.format(dt="bf16", m="mask"),
    "inst_nc_bf":P.format(dt="bf16", m="nmask"),
}
shutil.copy(os.path.join(HERE, "ck_fmha_varlen_ext.cpp"), os.path.join(SRC, "ck_fmha_varlen_ext.cu"))
for stem, fn in INSTS.items():
    shutil.copy(os.path.join(GEN, fn), os.path.join(SRC, stem + ".cu"))
sources = [os.path.join(SRC, "ck_fmha_varlen_ext.cu")] + [os.path.join(SRC, s + ".cu") for s in INSTS]

print("torch", torch.__version__, "hip", torch.version.hip)
t0 = time.perf_counter()
cpp_extension.load(
    name="ck_fmha_varlen_C", sources=sources, build_directory=BUILD_DIR,
    extra_include_paths=[os.path.join(CK, "include"), FMHA],
    extra_cuda_cflags=[
        "--offload-arch=gfx1100", "-std=c++17", "-O3",
        f"-I{os.path.join(CK, 'include')}", f"-I{FMHA}",
        "--rocm-path=C:\\HIP-SDK", "--rocm-device-lib-path=C:\\HIP-SDK\\lib\\llvm\\amdgcn\\bitcode",
        "-Wno-undefined-func-template", "-Wno-float-equal",
        "-DCK_TILE_FMHA_FWD_FAST_EXP2=1", "-fgpu-flush-denormals-to-zero",
        "-DCK_TILE_FMHA_FWD_SPLITKV_API=0", "-DCK_TILE_FMHA_FWD_APPENDKV_API=0",
        "-DCK_TILE_FMHA_FWD_PAGEDKV_API=0", "-DCK_TILE_FMHA_FWD_BATCH_PREFILL_API=0",
    ],
    extra_ldflags=["/LIBPATH:C:\\HIP-SDK\\lib", "amdhip64.lib"],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")
