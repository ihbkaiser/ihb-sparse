# RULER README and Requirements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the README with a concise, reproducible RULER run guide and add an exact pinned dependency file.

**Architecture:** Document the repository's two supported execution paths: the mixed-task `ruler_pilot`/`ruler_runner` workflow for Dense, Quest, and ShadowKV, and the Hydra workflow for the remaining registered attention implementations. Pin the direct runtime dependencies in a root-level `requirements.txt`, including the CUDA 12.8 PyTorch index.

**Tech Stack:** Python 3.10+, PyTorch 2.8.0 CUDA 12.8, vLLM 0.11.0, Hugging Face Transformers, Hydra, pytest.

---

### Task 1: Map the executable RULER workflow

**Files:**
- Read: `sparse_frontier/ruler_pilot.py`
- Read: `sparse_frontier/ruler_runner.py`
- Read: `sparse_frontier/configs/model/*.yaml`
- Read: `sparse_frontier/configs/attention/*.yaml`

- [x] Confirm generator arguments, supported context lengths, model argument, method/budget matrix, output artifacts, and environment controls from source.

### Task 2: Write the reproducibility documentation

**Files:**
- Modify: `README.md`

- [ ] Document prerequisites, installation, model authentication/path selection, dataset generation for 8K/16K/32K, smoke/full matrix commands, single-method commands, optional ShadowKV settings, legacy Hydra commands, outputs, and troubleshooting constraints.

### Task 3: Pin install requirements

**Files:**
- Create: `requirements.txt`

- [ ] Pin every direct package required by the documented RULER workflow and include the CUDA 12.8 wheel index needed for PyTorch 2.8.0.

### Task 4: Verify before publishing

**Files:**
- Test: `tests/test_ruler_runner.py`, `tests/test_ruler_pilot.py`, `tests/test_evaluation_cli.py`

- [ ] Run focused and full tests, compile/import checks, README command/schema checks, inspect the diff, and verify no credential is present in tracked content.

### Task 5: Publish

**Files:**
- Git: commit README and requirements changes, then push `main` to `https://github.com/ihbkaiser/ihb-sparse` with `gh` authentication.

- [ ] Confirm the pushed commit and remote repository state without printing credentials.
