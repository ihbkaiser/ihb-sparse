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
    "datasets>=3.0,<4",
    "wonderwords",
    "flashinfer-python",
    "flashinfer-cubin",
]

setup(
    name="sparse_frontier",
    version="1.0.0",
    description="Official implementation of the Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs",
    url="https://github.com/PiotrNawrot/sparse-frontier",
    packages=find_packages(include=['sparse_frontier', 'sparse_frontier.*']),
    entry_points={
        'vllm.general_plugins': [
            "swap_vllm_attention = sparse_frontier.modelling.models.vllm_model:swap_vllm_attention"
        ]
    },
    install_requires=_core_deps + _experiments_deps,
    extras_require={
        "plotting": _plotting_deps,
    },
    python_requires=">=3.10",
)
