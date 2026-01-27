# sparse_frontier/tasks/math/math_utils.py
from __future__ import annotations

import math
import re
from typing import Iterable, List, Dict, Optional

import numpy as np

# stable file reference from Lighteval
# https://github.com/huggingface/lighteval/blob/7c1cd62716b0a198a630c26d781430c54726cd02/src/lighteval/metrics/utils/extractive_match_utils.py
#
# release commit:
# https://github.com/huggingface/lighteval/tags

_ANSWER_BLOCK_RE = re.compile(r"(?is)<\s*answer\s*>(.*?)<\s*/\s*answer\s*>")
_ANSWER_TAG_RE = re.compile(r"(?i)\bANSWER\s*:\s*(.+)")
_FINAL_ANSWER_RE = re.compile(
    r"(?i)(?:therefore,?\s*)?(?:the\s*)?final\s*answer\s*is\s*:\s*(.+?)($|[\n\r\.])"
)

# Inline $...$ or $$...$$ on a single line
_DOLLAR_MATH_RE = re.compile(r"\${1,2}\s*([^$]+?)\s*\${1,2}")

# Multiline display math blocks (very common in model outputs)
_DISPLAY_MATH_BLOCK_RE = re.compile(r"(?is)\$\$\s*(.+?)\s*\$\$")
_BRACKET_MATH_BLOCK_RE = re.compile(r"(?is)\\\[\s*(.+?)\s*\\\]")
_BARE_DOLLAR_LINE_RE = re.compile(r"^\s*\${1,2}\s*$")

_WHITESPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
_FRAC_RE = re.compile(r"^\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*$")


# LaTeX helpers
_LATEX_TEXT_CMD_RE = re.compile(
    r"""\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|mbox)\s*\{([^{}]*)\}""",
    re.IGNORECASE,
)

_LATEX_SPACING_CMDS_RE = re.compile(r"""\\(?:,|;|:|!|\s)""")
_LEFT_RIGHT_RE = re.compile(r"""\\(?:left|right)\b""")

_LATEX_FRAC_RE = re.compile(
    r"""\\(?:dfrac|tfrac|frac)\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}"""
)

# Loose \frac variants seen in Math-500 golds: \frac43, \frac 34, \frac9{19}, \frac{270}7, etc.
_LATEX_FRAC_AB_RE = re.compile(r"""\\(?:dfrac|tfrac|frac)\s*([-+]?\d+)\s*([-+]?\d+)""")
_LATEX_FRAC_A_BRE = re.compile(r"""\\(?:dfrac|tfrac|frac)\s*([-+]?\d+)\s*\{\s*([^{}]+?)\s*\}""")
_LATEX_FRAC_BRA_B_RE = re.compile(r"""\\(?:dfrac|tfrac|frac)\s*\{\s*([^{}]+?)\s*\}\s*([-+]?\d+)""")

_DEGREE_RE = re.compile(
    r"""(?i)\b(\d+(?:\.\d+)?)\s*(?:\^\s*\{?\s*\\circ\s*\}?\s*|\\circ\b|\bdegrees?\b)"""
)

_LATEX_SQRT_RE = re.compile(r"""\\sqrt\s*\{\s*([^{}]+?)\s*\}""")
_LATEX_SQRT_BARE_RE = re.compile(r"""\\sqrt\s*([A-Za-z0-9]+)""")
_UNICODE_SQRT_BARE_RE = re.compile(r"""√\s*([A-Za-z0-9]+)""")

_TEXT_ONLY_RE = re.compile(r"^[A-Za-z]+(?:\s+[A-Za-z]+)*$")

# Ordinals like 12th / 12^{\mathrm{th}}
_ORDINAL_RE = re.compile(r"(?i)\b(\d+)\s*(st|nd|rd|th)\b")
_LATEX_ORDINAL_RE = re.compile(
    r"""(?i)\b(\d+)\s*\^\s*\{\s*\\(?:mathrm|text|mbox)\s*\{\s*(st|nd|rd|th)\s*\}\s*\}"""
)

# Single-letter choices wrapped in parentheses: (C) -> c
_SINGLE_LETTER_PAREN_RE = re.compile(r"^\(\s*([a-z])\s*\)$", re.IGNORECASE)

# x=5 / x \in [...] wrappers
_ASSIGNMENT_RE = re.compile(r"(?is)^\s*([a-z])\s*=\s*(.+?)\s*$")
_MEMBERSHIP_RE = re.compile(r"(?is)^\s*([a-z])\s*(?:\\in|∈|in)\s*(.+?)\s*$")

# Common symbol normalizations
_SYMBOL_REPLACEMENTS = [
    (re.compile(r"""\\pi\b"""), "pi"),
    (re.compile(r"""\\cdot\b"""), "*"),
    (re.compile(r"""\\times\b"""), "*"),
    (re.compile(r"""\\cup\b"""), "∪"),
    (re.compile(r"""\\infty\b"""), "infty"),
    (re.compile(r"""\\pm\b"""), "±"),
]


def _extract_last_math_block(text: str) -> Optional[str]:
    """Return content of the last $$...$$ or \\[...\\] block if present (non-empty)."""
    if not text:
        return None
    last: Optional[str] = None
    for m in _DISPLAY_MATH_BLOCK_RE.finditer(text):
        cand = (m.group(1) or "").strip()
        if cand:
            last = cand
    if last is not None:
        return last
    for m in _BRACKET_MATH_BLOCK_RE.finditer(text):
        cand = (m.group(1) or "").strip()
        if cand:
            last = cand
    return last


def _extract_last_balanced_arg(text: str, cmd: str) -> Optional[str]:
    """
    Extract the *last* balanced-brace argument of a LaTeX command like \\boxed{...}
    allowing nested braces.

    Example: "\\boxed{\\frac{14}{3}}" -> "\\frac{14}{3}"
    """
    needle = "\\" + cmd
    last = text.rfind(needle)
    if last == -1:
        return None

    i = last + len(needle)
    n = len(text)

    while i < n and text[i].isspace():
        i += 1

    if i >= n or text[i] not in "{(":
        return None

    open_ch = text[i]
    close_ch = "}" if open_ch == "{" else ")"
    i += 1

    depth = 1
    start = i
    while i < n:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _strip_latex_wrappers(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    # Prefer the last multiline display-math block if present
    block = _extract_last_math_block(s)
    if block is not None:
        s = block.strip()

    boxed = _extract_last_balanced_arg(s, "boxed")
    if boxed is not None:
        s = boxed.strip()

    # Inline $...$ / $$...$$ on one line
    m = _DOLLAR_MATH_RE.search(s)
    if m:
        s = m.group(1).strip()

    # \(...\), \[...\]
    s = re.sub(r"\\\(|\\\)", "", s)
    s = re.sub(r"\\\[|\\\]", "", s)
    s = _LEFT_RIGHT_RE.sub("", s)

    # Peel outer braces if balanced
    s = s.strip()
    while s.startswith("{") and s.endswith("}") and len(s) >= 2:
        inner = s[1:-1].strip()
        if inner.count("{") != inner.count("}"):
            break
        s = inner

    return s.strip()


def extract_tagged_response(text: str) -> str:
    """
    Extract an answer span from the model output.
    """
    if not text:
        return ""

    m = _ANSWER_BLOCK_RE.search(text)
    if m:
        body = m.group(1).strip()
        m2 = _ANSWER_TAG_RE.search(body)
        return _strip_latex_wrappers(m2.group(1).strip()) if m2 else _strip_latex_wrappers(body)

    m = _ANSWER_TAG_RE.search(text)
    if m:
        return _strip_latex_wrappers(m.group(1).strip())

    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return _strip_latex_wrappers(m.group(1).strip())

    boxed = _extract_last_balanced_arg(text, "boxed")
    if boxed is not None:
        return _strip_latex_wrappers(boxed.strip())

    # Critical: handle answers emitted as multiline $$ ... $$ blocks
    block = _extract_last_math_block(text)
    if block is not None:
        return _strip_latex_wrappers(block)

    # Fallback: last non-empty line, but skip bare "$$" / "$"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _BARE_DOLLAR_LINE_RE.fullmatch(ln):
            continue
        return _strip_latex_wrappers(ln)
    return ""


def _normalize_latex_text_cmds(s: str) -> str:
    while True:
        new_s = _LATEX_TEXT_CMD_RE.sub(r"\1", s)
        if new_s == s:
            return s
        s = new_s


def _normalize_latex_fracs_to_slash(s: str) -> str:
    # braced {a}{b}
    while True:
        new_s = _LATEX_FRAC_RE.sub(r"\1/\2", s)
        if new_s == s:
            break
        s = new_s

    # loose variants
    while True:
        new_s = _LATEX_FRAC_BRA_B_RE.sub(r"\1/\2", s)
        new_s = _LATEX_FRAC_A_BRE.sub(r"\1/\2", new_s)
        new_s = _LATEX_FRAC_AB_RE.sub(r"\1/\2", new_s)
        if new_s == s:
            break
        s = new_s

    return s


def _normalize_sqrts(s: str) -> str:
    # \sqrt{...} -> sqrt(...)
    while True:
        new_s = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s

    # \sqrt2 -> sqrt(2)
    while True:
        new_s = _LATEX_SQRT_BARE_RE.sub(r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s

    # √2 -> sqrt(2)
    while True:
        new_s = _UNICODE_SQRT_BARE_RE.sub(r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s

    return s


def normalize_answer(s: str) -> str:
    """
    Normalize answers for robust matching on Math-500.

    Handles:
    - answer blocks / tags / final answer lines
    - multiline $$...$$ extraction
    - \\boxed{...}
    - \\text{...} etc
    - \\frac variants (\\frac{a}{b}, \\frac ab, \\frac a{b}, \\frac{a}b)
    - sqrt variants (\\sqrt{2}, \\sqrt2, √2)
    - degrees (90^\\circ / 90 degrees -> 90)
    - thousands separators (10,080 -> 10080) while keeping tuple commas
    - removes spaces for "mathy" strings so "6 + 9i" == "6+9i"
    - strips x=..., x in ... wrappers
    """
    if s is None:
        return ""

    lower = s.lower()
    if "\n" in s or "answer:" in lower or "final answer" in lower or "<answer>" in lower:
        s = extract_tagged_response(s)

    s = _strip_latex_wrappers(s)
    s = re.sub(r"(?i)^\s*answer\s*:\s*", "", s).strip()

    # Unicode / common variants
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("π", "pi")  # unicode pi
    s = s.replace("\\$", "$")

    # Remove LaTeX spacing commands like \!, \, etc
    s = _LATEX_SPACING_CMDS_RE.sub("", s)

    # Text commands: \text{east} -> east, \mbox{ inches} ->  inches
    s = _normalize_latex_text_cmds(s)

    # Fractions: multiple forms -> a/b
    s = _normalize_latex_fracs_to_slash(s)

    # Sqrt: multiple forms -> sqrt(...)
    s = _normalize_sqrts(s)

    # Degrees -> just the number (works for simple numerals)
    s = _DEGREE_RE.sub(r"\1", s)

    # Symbols: \pi, \cdot, \times, \cup, \infty, \pm
    for rx, rep in _SYMBOL_REPLACEMENTS:
        s = rx.sub(rep, s)

    # \left \right
    s = _LEFT_RIGHT_RE.sub("", s)

    # Remove $ but KEEP commas (commas are meaningful in tuples/sets)
    s = s.replace("$", "")

    # Remove thousands separators only: 10,080 -> 10080 (but keep "(3,4)")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # Strip quotes
    s = s.strip().strip("\"'")

    # Strip trailing period unless it's part of "digit."
    if s.endswith(".") and not re.search(r"\d\.$", s):
        s = s[:-1].strip()

    # Ordinals: 12th / 12^{\mathrm{th}} -> 12
    s = _LATEX_ORDINAL_RE.sub(r"\1", s)
    s = _ORDINAL_RE.sub(r"\1", s)

    # Drop "degrees" word that can survive after fraction normalization (e.g. 270/7 degrees)
    s = re.sub(r"(?i)\bdegrees?\b", "", s).strip()

    # Strip x=... and x in ... wrappers (keep RHS)
    m = _ASSIGNMENT_RE.match(s)
    if m:
        s = m.group(2).strip()
    m = _MEMBERSHIP_RE.match(s)
    if m:
        s = m.group(2).strip()

    # Collapse whitespace
    s = _WHITESPACE_RE.sub(" ", s).strip()

    # (C) -> c (common multiple-choice formatting)
    m = _SINGLE_LETTER_PAREN_RE.match(s)
    if m:
        s = m.group(1).lower()

    # If it's "mathy", remove spaces entirely so "6 + 9i" == "6+9i"
    if s and not _TEXT_ONLY_RE.fullmatch(s):
        s = s.replace(" ", "")

    return s.lower()


def _try_parse_number(s: str) -> Optional[float]:
    """
    Parse numbers after normalization.
    Supports:
      - integers, decimals
      - a/b fractions
      - simple "number + unit" like "12cm" / "5.4cents"
    """
    s = normalize_answer(s)
    if not s:
        return None

    if _NUM_RE.fullmatch(s):
        try:
            return float(s)
        except Exception:
            return None

    m = _FRAC_RE.fullmatch(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            return num / den
        return None

    # number + unit (after normalize spaces may be gone): "12cm", "5.4cents", "864inches^2"
    m0 = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*[a-zA-Z][a-zA-Z0-9\^\s]*\s*$", s)
    if m0:
        try:
            return float(m0.group(1))
        except Exception:
            return None

    return None


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    for ch in s:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)

        if ch == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_top_level_pm(s: str) -> Optional[int]:
    """Find index of a top-level '±' (not inside (), [], {})."""
    depth_paren = depth_brack = depth_brace = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "±" and depth_paren == depth_brack == depth_brace == 0:
            return i
    return None


def _prepare_for_sympy(s: str) -> str:
    if s is None:
        return ""
    lower = s.lower()
    if "\n" in s or "answer:" in lower or "final answer" in lower or "<answer>" in lower:
        s = extract_tagged_response(s)

    s = _strip_latex_wrappers(s)
    s = re.sub(r"(?i)^\s*answer\s*:\s*", "", s).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("π", "pi").replace("\\$", "$")
    s = _LATEX_SPACING_CMDS_RE.sub("", s)

    s = _normalize_latex_text_cmds(s)

    # fractions -> (a)/(b)
    while True:
        new_s = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s

    while True:
        new_s = _LATEX_FRAC_BRA_B_RE.sub(r"(\1)/(\2)", s)
        new_s = _LATEX_FRAC_A_BRE.sub(r"(\1)/(\2)", new_s)
        new_s = _LATEX_FRAC_AB_RE.sub(r"(\1)/(\2)", new_s)
        if new_s == s:
            break
        s = new_s

    # sqrt
    while True:
        new_s = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s
    while True:
        new_s = _LATEX_SQRT_BARE_RE.sub(r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s
    while True:
        new_s = _UNICODE_SQRT_BARE_RE.sub(r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s

    s = _DEGREE_RE.sub(r"\1", s)

    for rx, rep in _SYMBOL_REPLACEMENTS:
        s = rx.sub(rep, s)

    s = _LEFT_RIGHT_RE.sub("", s)

    # strip x=..., x in ...
    m = _ASSIGNMENT_RE.match(s)
    if m:
        s = m.group(2).strip()
    m = _MEMBERSHIP_RE.match(s)
    if m:
        s = m.group(2).strip()

    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _try_parse_sympy(s: str):
    s = (s or "").strip()
    if not s:
        return None
    if _TEXT_ONLY_RE.fullmatch(s):
        return None

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
    except Exception:
        return None

    t = s

    # Expand "a ± b" at top level into (a+b, a-b)
    pm_idx = _find_top_level_pm(t)
    if pm_idx is not None:
        left = t[:pm_idx].strip()
        right = t[pm_idx + 1 :].strip()
        a = _try_parse_sympy(left)
        b = _try_parse_sympy(right)
        if a is not None and b is not None:
            return (a + b, a - b)

    # Tuples like "(1, 2)" or "1,2"
    if t.startswith("(") and t.endswith(")"):
        inner = t[1:-1].strip()
        parts = _split_top_level_commas(inner)
        if len(parts) > 1:
            objs = []
            for p in parts:
                o = _try_parse_sympy(p)
                if o is None:
                    return None
                objs.append(o)
            return tuple(objs)

    parts = _split_top_level_commas(t)
    if len(parts) > 1:
        objs = []
        for p in parts:
            o = _try_parse_sympy(p)
            if o is None:
                return None
            objs.append(o)
        return tuple(objs)

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {"pi": sp.pi, "sqrt": sp.sqrt, "infty": sp.oo}
    try:
        return parse_expr(t, transformations=transformations, local_dict=local_dict, evaluate=True)
    except Exception:
        return None


def _sympy_equal(a, b, atol: float = 1e-9) -> bool:
    try:
        import sympy as sp
    except Exception:
        return False

    if a is None or b is None:
        return False

    # Order-insensitive tuple compare (many answers are sets/solution lists where order shouldn't matter)
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        used = [False] * len(b)
        for x in a:
            found = False
            for j, y in enumerate(b):
                if used[j]:
                    continue
                if _sympy_equal(x, y, atol=atol):
                    used[j] = True
                    found = True
                    break
            if not found:
                return False
        return True

    try:
        if getattr(a, "is_number", False) and getattr(b, "is_number", False):
            av = float(sp.N(a))
            bv = float(sp.N(b))
            return math.isclose(av, bv, rel_tol=0.0, abs_tol=atol)
    except Exception:
        pass

    try:
        return sp.simplify(a - b) == 0
    except Exception:
        try:
            return bool(a.equals(b))
        except Exception:
            return False


def answers_equal(pred: str, golds: Iterable[str], atol: float = 1e-9) -> bool:
    """
    Check if predicted answer matches one of the gold answers.
    """
    pred_n = normalize_answer(pred)
    golds_list = list(golds)
    golds_n = [normalize_answer(g) for g in golds_list]

    if pred_n in golds_n:
        return True

    pnum = _try_parse_number(pred_n)
    if pnum is not None:
        for g in golds_n:
            gnum = _try_parse_number(g)
            if gnum is not None and math.isclose(pnum, gnum, rel_tol=0.0, abs_tol=atol):
                return True

    pred_s = _prepare_for_sympy(pred)
    pred_obj = _try_parse_sympy(pred_s)
    if pred_obj is None:
        return False

    for g in golds_list:
        g_s = _prepare_for_sympy(g)
        g_obj = _try_parse_sympy(g_s)
        if g_obj is not None and _sympy_equal(pred_obj, g_obj, atol=atol):
            return True

    return False


def avg_at_n(all_scores: List[int]) -> float:
    return float(np.mean(all_scores)) if all_scores else 0.0


def pass_at_k_estimator(c: int, n: int, k: int) -> float:
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    terms = 1.0 - (k / np.arange(n - c + 1, n + 1, dtype=np.float64))
    return float(1.0 - np.prod(terms))


def _comb(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def g_pass_at_k_table(n: int, c: int, k: int, thresholds: Iterable[float]) -> Dict[str, float]:
    def tail_prob_at_least(m: int) -> float:
        if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
            return 0.0
        num = 0
        for i in range(m, min(c, k) + 1):
            num += _comb(c, i) * _comb(n - c, k - i)
        den = _comb(n, k)
        return float(num / den) if den > 0 else 0.0

    metrics = {}
    for t in thresholds:
        m = max(int(math.ceil(k * float(t))), 1)
        metrics[f"g-pass@{k}_{t}"] = tail_prob_at_least(m)

    low = int(math.ceil(k * 0.5))
    mg = 0.0
    for i in range(low + 1, k + 1):
        mg += tail_prob_at_least(i)
    mg = 2.0 * mg / k if k > 0 else 0.0
    metrics[f"mg-pass@{k}"] = mg
    return metrics
