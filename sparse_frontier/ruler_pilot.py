"""Reproducible, mixed-task RULER pilot data for the vLLM experiments.

The five public names in this module are stable aliases for the corresponding
official RULER configurations:

* ``niah_single`` -> ``niah_single_1``
* ``niah_multikey`` -> ``niah_multikey_1``
* ``niah_multiquery`` -> ``niah_multiquery``
* ``vt`` -> ``vt``
* ``fwe`` -> ``fwe``

The generated JSONL intentionally keeps both this repository's fields
(``input_text``/``gold_answer``) and RULER's fields (``input``/``outputs``),
so it is standalone and can be audited without reconstructing task state.
Generation is deterministic and refuses to overwrite an existing file.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PILOT_TASKS = {
    "niah_single": {
        "ruler_task": "niah_single_1",
        "task": "niah",
        "args": {
            "type_haystack": "noise",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
        "tokens_to_generate": 128,
    },
    "niah_multikey": {
        "ruler_task": "niah_multikey_1",
        "task": "niah",
        "args": {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 4,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
        "tokens_to_generate": 128,
    },
    "niah_multiquery": {
        "ruler_task": "niah_multiquery",
        "task": "niah",
        "args": {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 1,
            "num_needle_v": 1,
            "num_needle_q": 4,
        },
        "tokens_to_generate": 128,
    },
    "vt": {
        "ruler_task": "vt",
        "task": "variable_tracking",
        "args": {"type_haystack": "noise", "num_chains": 1, "num_hops": 4},
        "tokens_to_generate": 30,
    },
    "fwe": {
        "ruler_task": "fwe",
        "task": "freq_words_extraction",
        "args": {"alpha": 2.0},
        "tokens_to_generate": 50,
    },
}

_NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the "
    "provided text?"
)
_VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\nQuestion: Find all variables that are assigned the value {query} in the text above."
)
_FWE_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. Find the three "
    "most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. "
    "Please ignore the dots '....'. What are the three most frequently appeared words in the "
    "above coded text?"
)

_NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again."
)
_FALLBACK_ESSAY = (
    "Paul Graham writes about making things, learning from mistakes, and choosing problems "
    "that matter. He describes how small teams discover ideas by observing the world closely. "
    "The essay returns to the value of patience, clear thinking, and useful work. "
)
_FALLBACK_WORDS = (
    "amber birch cobalt dahlia ember fjord granite hazel ivory juniper kettle linen maple "
    "nectar olive pebble quartz river saffron thistle umber velvet willow xenon yarrow zinc"
).split()


def _tokens(tokenizer: Any, text: str) -> list[Any]:
    return list(tokenizer.text_to_tokens(text))


def _generation_tokens(tokenizer: Any, text: str) -> list[Any]:
    encoded = tokenizer.encode_for_generation(text, return_tensors=False)
    return list(encoded["input_ids"])


def _number(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def _word(rng: random.Random, word_source: Sequence[str] | None) -> str:
    if word_source:
        return word_source[rng.randrange(len(word_source))]
    left = rng.choice(_FALLBACK_WORDS)
    right = rng.choice(_FALLBACK_WORDS)
    return f"{left}-{right}"


def _uuid_like(rng: random.Random) -> str:
    groups = ["".join(rng.choice("0123456789abcdef") for _ in range(n)) for n in (8, 4, 4, 4, 12)]
    return "-".join(groups)


def _needle(
    rng: random.Random,
    key_type: str,
    value_type: str,
    word_source: Sequence[str] | None,
) -> tuple[str, str, str]:
    key = _word(rng, word_source) if key_type == "words" else _uuid_like(rng)
    if value_type == "numbers":
        value = _number(rng)
    elif value_type == "words":
        value = _word(rng, word_source)
    elif value_type == "uuids":
        value = _uuid_like(rng)
    else:
        raise ValueError(f"Unsupported RULER needle type: {value_type}")
    singular = value_type[:-1] if value_type.endswith("s") else value_type
    sentence = f"One of the special magic {singular} for {key} is: {value}."
    return key, value, sentence


def _essay_sentences(essay_text: str) -> list[str]:
    # This is deliberately conservative: it avoids making NLTK a runtime
    # dependency for evaluation while retaining the official sentence insertion
    # behavior for ordinary prose.
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", essay_text) if part.strip()]
    return sentences or [essay_text.strip()]


def _build_niah(
    tokenizer: Any,
    args: Mapping[str, Any],
    rng: random.Random,
    max_prompt_tokens: int,
    essay_text: str | None,
    word_source: Sequence[str] | None,
) -> tuple[str, list[str], dict[str, Any]]:
    n_k = max(int(args["num_needle_k"]), int(args["num_needle_q"]))
    n_v = int(args["num_needle_v"])
    n_q = int(args["num_needle_q"])
    needles: list[tuple[str, str, str]] = []
    used_keys: set[str] = set()
    used_values: set[str] = set()
    for _ in range(n_k):
        while True:
            item = _needle(
                rng, args["type_needle_k"], args["type_needle_v"], word_source
            )
            if item[0] not in used_keys and item[1] not in used_values:
                used_keys.add(item[0])
                used_values.add(item[1])
                needles.append(item)
                break

    needle_sentences = [item[2] for item in needles for _ in range(n_v)]
    rng.shuffle(needle_sentences)
    if args["type_haystack"] == "essay":
        if not essay_text:
            raise ValueError(
                "Official RULER essay variants require --essay_path (PaulGrahamEssays.json); "
                "refusing to silently substitute a different haystack"
            )
        source = _essay_sentences(essay_text)
    elif args["type_haystack"] == "noise":
        context = _NOISE_SENTENCE
    else:
        raise ValueError(f"Unsupported official RULER NIAH haystack: {args['type_haystack']}")

    queries = rng.sample(needles, n_q)
    query = ", ".join(item[0] for item in queries[:-1])
    query = f"{query}, and {queries[-1][0]}" if n_q > 1 else queries[0][0]
    answers = [value for item in queries for value in [item[1]] * n_v]
    value_type = args["type_needle_v"]
    if n_q * n_v == 1:
        value_type = value_type[:-1]

    def render(current_context: str) -> str:
        return _NIAH_TEMPLATE.format(
            type_needle_v=value_type,
            context=current_context,
            query=query,
        )

    # Keep the prompt at the official maximum. The needle placement follows
    # RULER's depth/sample behavior, not an arbitrary global shuffle.
    placement_seed = rng.randrange(2**63)

    def insert_official(haystack_sentences: list[str]) -> list[str]:
        placement_rng = random.Random(placement_seed)
        if args["type_haystack"] == "essay":
            depths = [round(i * 100 / 39) for i in range(40)]
            chosen = sorted(placement_rng.sample(depths, len(needle_sentences)))
            insertion_positions = [0] + [int(len(haystack_sentences) * depth / 100) for depth in chosen] + [len(haystack_sentences)]
            combined: list[str] = []
            for segment in range(1, len(insertion_positions)):
                combined.extend(haystack_sentences[insertion_positions[segment - 1]:insertion_positions[segment]])
                if segment - 1 < len(needle_sentences):
                    combined.append(needle_sentences[segment - 1])
            return combined
        combined = list(haystack_sentences)
        if len(combined) < len(needle_sentences):
            return combined + list(needle_sentences)
        indexes = sorted(placement_rng.sample(range(len(combined)), len(needle_sentences)), reverse=True)
        for index, sentence in zip(indexes, needle_sentences):
            combined.insert(index, sentence)
        return combined

    if args["type_haystack"] == "essay":
        def candidate(count: int) -> list[str]:
            return (source * ((count + len(source) - 1) // len(source)))[:count]
        upper = max_prompt_tokens
    else:
        def candidate(count: int) -> list[str]:
            return [_NOISE_SENTENCE] * count
        upper = max_prompt_tokens // 4

    # Binary search instead of tokenizing after every sentence.  This matters
    # at 8K, where a naive implementation turns pilot generation into an
    # O(n^2) tokenizer workload.
    low, high, best = 0, upper, 0
    while low <= high:
        middle = (low + high) // 2
        current = candidate(middle)
        prompt_tokens = len(_generation_tokens(tokenizer, render(" ".join(insert_official(current)))))
        if prompt_tokens <= max_prompt_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    haystack_sentences = candidate(best)
    all_sentences = insert_official(haystack_sentences)
    prompt = render(" ".join(all_sentences))

    metadata = {"answers": answers, "query_keys": [item[0] for item in queries]}
    return prompt, answers, metadata


def _build_vt(
    tokenizer: Any,
    args: Mapping[str, Any],
    rng: random.Random,
    max_prompt_tokens: int,
) -> tuple[str, list[str], dict[str, Any]]:
    num_chains = int(args["num_chains"])
    num_hops = int(args["num_hops"])
    variables = []
    while len(variables) < num_chains * (num_hops + 1):
        value = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        if value not in variables:
            variables.append(value)
    chains = []
    for chain_idx in range(num_chains):
        chain_vars = variables[chain_idx * (num_hops + 1):(chain_idx + 1) * (num_hops + 1)]
        number = str(rng.randint(10_000, 99_999))
        chain = [f"VAR {chain_vars[0]} = {number}"]
        chain.extend(f"VAR {chain_vars[i + 1]} = VAR {chain_vars[i]}" for i in range(num_hops))
        chains.append(chain)
    assignment_statements = [statement for chain in chains for statement in chain]
    def noise_candidate(count: int) -> list[str]:
        return [_NOISE_SENTENCE] * count

    low, high, best = 0, max_prompt_tokens // 4, 0
    while low <= high:
        middle = (low + high) // 2
        candidate = noise_candidate(middle)
        candidate_prompt = _VT_TEMPLATE.format(
            context=" ".join(assignment_statements + candidate), query="0"
        )
        if len(_generation_tokens(tokenizer, candidate_prompt)) <= max_prompt_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    noise = noise_candidate(best)
    all_statements = assignment_statements + noise
    rng.shuffle(all_statements)
    target = chains[0][0].split("=")[-1].strip()
    prompt = _VT_TEMPLATE.format(context="\n".join(all_statements), query=target)
    if len(_generation_tokens(tokenizer, prompt)) > max_prompt_tokens:
        low, high, best = 0, len(noise), 0
        while low <= high:
            middle = (low + high) // 2
            candidate_noise = noise[:middle]
            candidate_statements = assignment_statements + candidate_noise
            rng_for_candidate = random.Random(rng.randrange(2**63))
            rng_for_candidate.shuffle(candidate_statements)
            candidate_prompt = _VT_TEMPLATE.format(
                context="\n".join(candidate_statements), query=target
            )
            if len(_generation_tokens(tokenizer, candidate_prompt)) <= max_prompt_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        noise = noise[:best]
        all_statements = assignment_statements + noise
        rng.shuffle(all_statements)
        prompt = _VT_TEMPLATE.format(context="\n".join(all_statements), query=target)
    return prompt, variables[: num_hops + 1], {"target_value": target, "target_vars": variables[: num_hops + 1]}


def _zeta_two() -> float:
    return math.pi**2 / 6.0


def _fwe_words(num_words: int, rng: random.Random, vocab_size: int, alpha: float) -> tuple[str, list[str]]:
    vocab: list[str] = []
    while len(vocab) < vocab_size:
        word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        if word not in vocab:
            vocab.append(word)
    vocab[0] = "..."
    normalizer = _zeta_two() if alpha == 2.0 else sum(k ** (-alpha) for k in range(1, vocab_size + 1))
    counts = [int(num_words * (k ** (-alpha)) / normalizer) for k in range(1, vocab_size + 1)]
    sampled = [word for word, count in zip(vocab, counts) for _ in range(count)]
    rng.shuffle(sampled)
    return " ".join(sampled), vocab[1:4]


def _build_fwe(
    tokenizer: Any,
    args: Mapping[str, Any],
    rng: random.Random,
    max_prompt_tokens: int,
) -> tuple[str, list[str], dict[str, Any]]:
    vocab_size = max(20, max_prompt_tokens // 50)
    alpha = float(args["alpha"])
    fwe_seed = rng.randrange(2**63)

    def render(num_words: int) -> tuple[str, list[str]]:
        # Candidate sizes are evaluated repeatedly during binary search. Use
        # the same per-size stream for the final render so token accounting is
        # invariant to how many search iterations preceded it.
        candidate_rng = random.Random(fwe_seed + num_words)
        context, answer = _fwe_words(num_words, candidate_rng, vocab_size, alpha)
        return _FWE_TEMPLATE.format(context=context), answer

    # Find the largest deterministic word budget that fits the prompt.
    low, high = 1, max_prompt_tokens * 2
    best = 1
    while low <= high:
        mid = (low + high) // 2
        candidate, _ = render(mid)
        if len(_generation_tokens(tokenizer, candidate)) <= max_prompt_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    prompt, answer = render(best)
    return prompt, answer, {"top_words": answer, "alpha": alpha, "num_words": best}


def build_pilot_rows(
    tokenizer: Any,
    samples_per_task: int = 50,
    max_seq_length: int = 8192,
    tokens_to_generate: int | None = None,
    seed: int = 20260831,
    essay_text: str | None = None,
    word_source: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the deterministic mixed-task pilot rows.

    ``max_seq_length`` follows RULER: it includes the expected generation
    allowance.  Therefore the prompt budget is task-specific
    ``max_seq_length - tokens_to_generate``.  Passing an explicit
    ``tokens_to_generate`` is useful for unit tests; production uses the
    official per-task values.
    """
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")
    if max_seq_length < 256:
        raise ValueError("max_seq_length is too small for the RULER prompt templates")
    rows: list[dict[str, Any]] = []
    for task_offset, (task_name, spec) in enumerate(PILOT_TASKS.items()):
        generation_tokens = int(tokens_to_generate or spec["tokens_to_generate"])
        prompt_budget = max_seq_length - generation_tokens
        if prompt_budget <= 0:
            raise ValueError(f"max_seq_length must exceed generation allowance for {task_name}")
        for task_index in range(samples_per_task):
            sample_seed = seed + task_offset * 1_000_003 + task_index
            rng = random.Random(sample_seed)
            if spec["task"] == "niah":
                prompt, answers, metadata = _build_niah(
                    tokenizer, spec["args"], rng, prompt_budget, essay_text, word_source
                )
            elif spec["task"] == "variable_tracking":
                prompt, answers, metadata = _build_vt(tokenizer, spec["args"], rng, prompt_budget)
            elif spec["task"] == "freq_words_extraction":
                prompt, answers, metadata = _build_fwe(tokenizer, spec["args"], rng, prompt_budget)
            else:  # pragma: no cover - guarded by the registry above
                raise AssertionError(f"No pilot builder for {spec['task']}")

            prompt_tokens = len(_generation_tokens(tokenizer, prompt))
            raw_tokens = len(_tokens(tokenizer, prompt))
            row = {
                "index": task_offset * samples_per_task + task_index,
                "task": task_name,
                "task_index": task_index,
                "ruler_task": spec["ruler_task"],
                "task_impl": spec["task"],
                "task_args": dict(spec["args"]),
                "context_length": max_seq_length,
                "tokens_to_generate": generation_tokens,
                "prompt_tokens": prompt_tokens,
                "input_text": prompt,
                "input": prompt,
                "gold_answer": list(answers),
                "outputs": list(answers),
                # RULER's length includes its task-specific generation budget;
                # retain both raw and chat-templated accounting for audits.
                "length": raw_tokens + generation_tokens,
                "length_w_model_template": prompt_tokens + generation_tokens,
                "random_seed": sample_seed,
            }
            row.update(metadata)
            rows.append(row)
    return rows


def write_pilot_once(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write a pilot JSONL and never overwrite it."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Pilot dataset already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_pilot_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Pilot dataset is empty: {path}")
    required = {"index", "task", "input_text", "gold_answer", "outputs", "length"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Pilot dataset is missing required fields: {missing}")
    indexes = [int(row["index"]) for row in rows]
    if indexes != list(range(len(rows))):
        raise ValueError("Pilot indexes must be contiguous and start at zero")
    counts = Counter(row["task"] for row in rows)
    expected = set(PILOT_TASKS)
    if set(counts) != expected:
        raise ValueError(f"Pilot tasks mismatch: expected {sorted(expected)}, got {sorted(counts)}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"Pilot task counts are not equal: {dict(counts)}")
    return rows


def evaluate_ruler_task(task: str, examples: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Official RULER synthetic string-match-all metric in [0, 1]."""
    if task not in PILOT_TASKS:
        raise ValueError(f"Unsupported pilot task {task!r}; expected one of {list(PILOT_TASKS)}")
    scores: list[float] = []
    nulls = 0
    for example in examples:
        prediction = str(example.get("pred", "")).strip()
        if not prediction:
            nulls += 1
        prediction = re.sub(r"[\x00-\x1f]", "\n", prediction).lower()
        references = example.get("outputs", example.get("gold_answer", []))
        if isinstance(references, str):
            references = [references]
        references = [str(reference).lower() for reference in references]
        score = sum(reference in prediction for reference in references) / len(references) if references else 0.0
        scores.append(float(score))
    mean = sum(scores) / len(scores) if scores else 0.0
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1) if len(scores) > 1 else 0.0
    return {
        "accuracy": mean,
        "accuracy_variance": variance,
        "null_predictions": nulls,
        "total_samples": len(examples),
    }


class RulerPilotTask:
    """Registry adapter for a single mixed-dataset task name."""

    @staticmethod
    def evaluate(examples: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        tasks = {str(example.get("task")) for example in examples}
        if len(tasks) != 1:
            raise ValueError(f"RulerPilotTask expects one task, got {sorted(tasks)}")
        return evaluate_ruler_task(next(iter(tasks)), examples)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create an immutable RULER pilot JSONL")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--essay_path", required=True, help="RULER PaulGrahamEssays.json")
    parser.add_argument("--samples_per_task", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    from sparse_frontier.modelling.tokenizer import Tokenizer

    with open(args.essay_path, encoding="utf-8") as handle:
        essay_text = json.load(handle)["text"]
    from wonderwords import random_word

    nouns = sorted(set(random_word._get_words_from_text_file("nounlist.txt")))
    adjectives = sorted(set(random_word._get_words_from_text_file("adjectivelist.txt")))

    class WordProduct(Sequence[str]):
        def __len__(self):
            return len(nouns) * len(adjectives)

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self[i] for i in range(*index.indices(len(self)))]
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            adjective_index, noun_index = divmod(index, len(nouns))
            return f"{adjectives[adjective_index]}-{nouns[noun_index]}"

    word_source = WordProduct()
    tokenizer = Tokenizer(args.model_path, device="cpu", thinking=False)
    rows = build_pilot_rows(
        tokenizer=tokenizer,
        samples_per_task=args.samples_per_task,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        essay_text=essay_text,
        word_source=word_source,
    )
    write_pilot_once(args.output, rows)
    print(f"Wrote {len(rows)} rows ({args.samples_per_task} per task) to {args.output}")


if __name__ == "__main__":
    main()
