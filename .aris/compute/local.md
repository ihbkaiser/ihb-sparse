# Local Query-Robust experiment environment

## Declarative specification

- Provider: local Vast container
- Workspace: `/workspace/sparse-frontier`
- Persistence: `/workspace` is not backed by a persistent volume
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, compute capability 8.6
- Host driver: 580.95.05 (CUDA maximum 13.0)
- System CUDA toolkit: 12.8
- Python: 3.12 at `/venv/main/bin/python`
- PyTorch: 2.8.0 CUDA 12.8 build
- torchvision: 0.23.0
- torchaudio: 2.8.0
- vLLM: 0.11.0
- transformers: 4.57.1
- Remaining packages: exact pins in `requirements.txt`
- Requirements SHA256: `c048cc133377b0098ae24875c450c88e8cf260d0d84ad269cf5385d90cdfa419`

## Model

- ID: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Revision: `d10aef7999a2b5ba950ab3974312feeedbfe0b77`
- Snapshot: `/workspace/.hf_home/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77`
- Config SHA256: `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`
- Index SHA256: `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b`
- Expected: BF16, 32 layers, 32 query heads, 8 KV heads, head dimension 128, Llama3 RoPE, no attention bias

## Ordered build

```bash
/venv/main/bin/python -m pip uninstall -y torch torchvision torchaudio torchcodec
uv pip install --python /venv/main/bin/python --no-cache torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /venv/main/bin/python --no-cache -r requirements.txt
uv pip install --python /venv/main/bin/python --no-cache -e .
```

If space is insufficient, `/root/.cache/pip` is replaceable. The model cache is not.

This container's base Torch tree is in a read-only overlay layer. Package wheels are staged with `TMPDIR=/dev/shm/query-robust-uv-tmp`; Torch must be installed into a clean package directory to prevent old lower-layer files from mixing with 2.8.0. The smallest model shard, prior-results archive, generated extension build, and RULER data are checksum-preserved in `/dev/shm` behind their original paths for this running instance. Restore or redownload them before a container restart.

## Witness commands

```bash
/venv/main/bin/python -m pytest -q
/venv/main/bin/python -c 'import torch; x=torch.arange(16,device="cuda",dtype=torch.float32).reshape(4,4); y=x@x.T; print(torch.__version__,torch.version.cuda,float(y.sum()))'
/venv/main/bin/python -c 'import torch; x=torch.tensor([1000.0,999.0],device="cuda",dtype=torch.bfloat16); y=torch.softmax(x.float(),-1); print(y.tolist(),bool(torch.isfinite(y).all()))'
```

Expected kernel sum: `3680.0`. Expected softmax: finite. The test count can increase as the feature lands, but failures and unexpected skips are not accepted.

## Experiment invocation

All GPU runs use the snapshot path above, `HF_HOME=/workspace/.hf_home`, `CUDA_VISIBLE_DEVICES=0`, fixed seeds, unique output directories, and `tee` logs. Run manifests record resolved packages, model/config hashes, git revision, GPU, attention scale, and artifact hashes.

Because the filesystem is ephemeral, commit source and small evidence before instance destruction. Raw checkpoints and tensor traces are not committed; checksum-bearing aggregates are preserved first.
