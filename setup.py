import os

from setuptools import setup, find_packages

# Core dependencies shared between plotting and experiments
_core_deps = [
    "pyyaml>=6",
    "numpy>=1.23",
]

_plotting_deps = [
    "matplotlib>=3.6",
    "seaborn>=0.13",
    "pandas>=2.0",
    "statsmodels>=0.14",
    "scipy>=1.10",
]

_experiments_deps = [
    "transformers>=4.55.2,<5",
    "tokenizers>=0.22,<0.24",
    "vllm==0.11.0",
    "accelerate>=1.0",
    "hydra-core>=1.3,<2",
    "omegaconf>=2.3,<3",
    "wonderwords",
]

# Optional for the Dense/Quest/ShadowKV RULER baseline. vLLM's existing
# FlashAttention backend is the execution path; FlashInfer is only a
# performance optimization and must not force a vLLM replacement or
# CUDA-extension install during baseline setup.
_optional_full_deps = [
    "datasets>=3.0,<4",
    "flashinfer-python",
    "flashinfer-cubin",
]

# The upstream ShadowKV CPU-offload extension depends on CUTLASS and owns its
# cache layout.  This repository exposes only the two useful fused GPU
# primitives through a small optional extension, leaving the current vLLM
# environment untouched unless explicitly requested at build time.
_shadowkv_ext_modules = []
_shadowkv_cmdclass = {}
if os.getenv("SF_BUILD_SHADOWKV_CUDA", "0") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    _shadowkv_ext_modules.append(
        CUDAExtension(
            name="sparse_frontier._shadowkv_cuda",
            sources=[
                "sparse_frontier/modelling/attention/csrc/shadowkv_bindings.cpp",
                "sparse_frontier/modelling/attention/csrc/shadowkv_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    )
    _shadowkv_cmdclass["build_ext"] = BuildExtension

setup(
    name="sparse_frontier",
    version="1.0.0",
    description="Official implementation of the Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs",
    url="https://github.com/PiotrNawrot/sparse-frontier",
    packages=find_packages(include=['sparse_frontier', 'sparse_frontier.*']),
    ext_modules=_shadowkv_ext_modules,
    cmdclass=_shadowkv_cmdclass,
    entry_points={
        'vllm.general_plugins': [
            "swap_vllm_attention = sparse_frontier.modelling.models.vllm_model:swap_vllm_attention"
        ]
    },
    install_requires=_core_deps + _experiments_deps,
    extras_require={
        "plotting": _plotting_deps,
        "full": _optional_full_deps,
    },
    python_requires=">=3.10",
)
