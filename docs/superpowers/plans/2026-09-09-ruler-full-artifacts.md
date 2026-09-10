# Full RULER Artifact Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate tokenizer-aligned full RULER JSONL artifacts at 32K, 64K, and 128K, upload them as separate Hugging Face dataset files, and let every Sparse-vLLM RULER method load the selected artifact by context length.

**Architecture:** Add one explicit preparation/upload CLI that invokes a pinned local NVIDIA RULER checkout for the official 13 synthetic tasks, converts each task's generated `test.jsonl` into a prompt-preserving row schema, validates target-length coverage and task/sample identities, and optionally uploads the resulting files. Extend the existing kvpress-compatible evaluator with a Hugging Face dataset-repository source whose filename template maps `--data-dir` to one JSONL file; local `--dataset-path` remains supported and method execution stays unchanged. The new schema stores upstream `input` as `prompt` and keeps `answer_prefix` separate so the evaluator can tokenize the exact generated prompt without guessing context/question boundaries.

**Tech Stack:** Python 3.12, Hugging Face `datasets`/`huggingface_hub`, NVIDIA RULER generator invoked as a subprocess, JSONL, pytest.

---

### Task 1: Define and test the full-artifact row contract

**Files:**
- Modify: `benchmark/kvpress_ruler/evaluate.py`
- Test: `tests/test_kvpress_ruler_max_new_tokens.py`

- [x] **Step 1: Write the failing tests**

Add tests for loading a local JSONL with either the legacy context/question fields or the new prompt-preserving fields, rejecting missing fields in any row, and mapping a selected context length to `ruler-{length}.jsonl` in a Hugging Face dataset repository through a mocked downloader. Add a prompt-tokenization test proving that the new path uses `prompt + answer_prefix` directly.

- [x] **Step 2: Run the focused tests and verify they fail for the missing source API**

Run: `pytest -q tests/test_kvpress_ruler_max_new_tokens.py`

Expected: FAIL because the new dataset-source configuration and resolver are not implemented.

- [x] **Step 3: Implement the smallest loader extension**

Add `dataset_repo_id`, `dataset_revision`, and `dataset_file_template` to `EvalConfig` and the CLI. Resolve exactly one remote filename from `data_dir`, download it with `hf_hub_download(repo_type="dataset")`, and route the downloaded path through the existing JSON/JSONL parser. Validate every row, not only the first row, against either `prompt`, `answer_prefix`, `answer`, `task`, and `max_new_tokens`, or the existing legacy `context`, `question`, `answer_prefix`, `answer`, `task`, and `max_new_tokens` schema. Add a shared row-to-prompt helper so all three sparse methods receive the same token IDs.

- [x] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_kvpress_ruler_max_new_tokens.py`

Expected: PASS, including explicit errors for missing fields and invalid dataset-source combinations.

### Task 2: Add the upstream RULER generation and Hugging Face upload CLI

**Files:**
- Create: `benchmark/kvpress_ruler/prepare_full_ruler.py`
- Test: `tests/test_prepare_full_ruler.py`

- [x] **Step 1: Write failing tests for deterministic conversion and validation**

Test conversion of one upstream task row into the prompt-preserving evaluator schema, preservation of list-valued answers, task names, answer prefixes, and row-local `max_new_tokens`. Test rejection of duplicate `(task, source_index)` identities, missing required upstream fields, and a generated row whose recorded token length exceeds its target length.

- [x] **Step 2: Run the new tests and verify the expected failure**

Run: `pytest -q tests/test_prepare_full_ruler.py`

Expected: FAIL because the preparation module does not exist.

- [x] **Step 3: Implement the preparation CLI**

Use explicit CLI parameters for the model/tokenizer path, matching RULER model-template type, local RULER checkout, output directory, three target lengths, sample count, seed, optional task list, optional HF repo ID, and optional revision. Invoke the upstream `scripts/data/prepare.py` once per task and length with `--subset test --tokenizer_type hf --model_template_type args.model_template_type`; fail immediately on a subprocess error. Read each task's generated `test.jsonl`, convert it to rows containing `prompt`, `answer_prefix`, `answer`, `task`, and `max_new_tokens`, add provenance fields (`context_length`, `seed`, `generator_commit`, `source_task`, `source_index`, `source_length`), validate all rows, write exactly `ruler-32768.jsonl`, `ruler-65536.jsonl`, and `ruler-131072.jsonl`, and upload only when `--hf-repo-id` is explicitly provided using `HfApi.upload_file`.

- [x] **Step 4: Run the new tests and verify they pass**

Run: `pytest -q tests/test_prepare_full_ruler.py`

Expected: PASS without invoking GPUs, network, or the upstream generator.

### Task 3: Document generation, upload, and all-method evaluation

**Files:**
- Modify: `docs/en/benchmarking/kvpress-ruler.md`
- Modify: `docs/zh/benchmarking/README.md`

- [x] **Step 1: Document the exact file naming and schema contract**

Document that the repository contains no generated 32K/64K/128K artifact, that generation must use the model tokenizer, that each file is one context length, and that the HF repository must contain the three exact JSONL names.

- [x] **Step 2: Document commands for preparation/upload and all three methods**

Provide commands that use explicit `MODEL_PATH`, `RULER_REPO`, `OUTPUT_DIR`, and `HF_REPO_ID` variables, run a smoke subset before the full 500-sample-per-task generation, upload the three files, and evaluate `quest`, `shadowkv`, and `query_robust` against the same HF repository with distinct output directories.

- [x] **Step 3: Run documentation command snippets at argument-validation level**

Run: `python benchmark/kvpress_ruler/prepare_full_ruler.py --help` and the evaluator parser tests.

Expected: Help lists the three target lengths, HF upload options, and the evaluator accepts `--dataset-repo-id` for every sparse method.

### Task 4: Verify the complete code path without expensive generation

**Files:**
- Modify: none
- Test: `tests/test_kvpress_ruler_max_new_tokens.py`, `tests/test_prepare_full_ruler.py`

- [x] **Step 1: Run focused tests**

Run: `pytest -q tests/test_kvpress_ruler_max_new_tokens.py tests/test_prepare_full_ruler.py`

Expected artifact: passing test output covering remote file resolution, full-row schema validation, conversion, and upload gating.

- [x] **Step 2: Run the broader RULER regression tests**

Run: `pytest -q tests/test_ruler_tasks.py tests/test_ruler_vt_regression.py tests/test_kvpress_ruler_max_new_tokens.py tests/test_prepare_full_ruler.py`

Expected artifact: all existing and new RULER tests pass; no GPU evaluation is started.

- [x] **Step 3: Report the remaining external prerequisites**

State that actual generation requires a local checkout of NVIDIA RULER plus its data dependencies and the target model tokenizer, while upload requires an authenticated Hugging Face session and an explicit dataset repository ID. Do not claim the three remote artifacts exist until the user runs the generation/upload command successfully.
