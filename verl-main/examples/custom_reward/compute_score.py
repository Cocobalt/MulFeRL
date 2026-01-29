import re
from typing import Any, Dict, Optional, Sequence

# -------------------- patterns --------------------
BOXED_PATTERN = re.compile(r"\\boxed\s*\{(.*?)\}", flags=re.S)

try:
    from math_verify import (  # type: ignore
        parse as mv_parse,
        verify as mv_verify,
        LatexExtractionConfig,
        ExprExtractionConfig,
        StringExtractionConfig,
    )
except Exception:
    from math_verify import parse as mv_parse, verify as mv_verify  # type: ignore

    try:
        from math_verify.parser import (  # type: ignore
            LatexExtractionConfig,
            ExprExtractionConfig,
            StringExtractionConfig,
        )
    except Exception:
        LatexExtractionConfig = None  # type: ignore
        ExprExtractionConfig = None  # type: ignore
        StringExtractionConfig = None  # type: ignore


# -------------------- helpers: format check --------------------
def check_format(solution_str: Optional[str]) -> float:
    """
    Loose format requirements:
      1) Starts with <thinking>
      2) Contains </thinking> and has trailing part
      3) Inside <thinking>, begins with <feedback>...</feedback>
      4) After </thinking>, contains at least one \\boxed{...}
    """
    if not solution_str:
        return 0.0

    stripped = solution_str.lstrip()
    if not stripped.startswith("<thinking>"):
        return 0.0

    m = re.match(r"<thinking>(.*)</thinking>(.*)", stripped, flags=re.DOTALL)
    if not m:
        return 0.0

    think_body, after_think = m.group(1), m.group(2)

    if not re.match(r"\s*<feedback>.*?</feedback>", think_body, flags=re.DOTALL):
        return 0.0

    if not BOXED_PATTERN.search(after_think):
        return 0.0

    return 1.0


def extract_after_thinking(solution_str: str) -> str:
    """
    If the completion follows <thinking>...</thinking>..., return the trailing part after </thinking>.
    Otherwise return original string.
    """
    m = re.match(r"\s*<thinking>.*?</thinking>(.*)", solution_str, flags=re.DOTALL)
    return m.group(1) if m else solution_str


_LATEX_ENV_RE = re.compile(r"(\$.*\$|\\\[.*\\\]|\\\(.*\\\)|\\begin\{.*?\}.*\\end\{.*?\})", re.S)


def wrap_latex_env_if_needed(s: str) -> str:
    """
    Math-Verify's LatexExtractionConfig expects LaTeX to appear in a LaTeX environment. :contentReference[oaicite:5]{index=5}
    So if we have backslash-y LaTeX but no env delimiters, wrap with $...$.
    """
    t = (s or "").strip()
    if not t:
        return t
    if _LATEX_ENV_RE.search(t):
        return t
    # Heuristic: if it looks like LaTeX, wrap.
    if "\\" in t or r"\frac" in t or r"\sqrt" in t or "^" in t or "_" in t:
        return f"${t}$"
    return t


# -------------------- helpers: math-verify compatibility wrappers --------------------
def _default_extraction_configs() -> Optional[Sequence[Any]]:
    """
    Return extraction configs if available. Otherwise None -> let Math-Verify default.
    README: default uses LatexExtractionConfig + ExprExtractionConfig. :contentReference[oaicite:6]{index=6}
    """
    cfgs = []
    if LatexExtractionConfig is not None:
        cfgs.append(LatexExtractionConfig())
    if ExprExtractionConfig is not None:
        cfgs.append(ExprExtractionConfig())
    if StringExtractionConfig is not None:
        cfgs.append(StringExtractionConfig())
    return cfgs or None


def mv_parse_robust(
    text: str,
    extraction_config: Optional[Sequence[Any]] = None,
    parsing_timeout: Optional[float] = None,
    raise_on_error: bool = False,
):
    """
    Call Math-Verify parse() with best-effort compatibility across versions.
    - parse supports extraction_config (README). :contentReference[oaicite:7]{index=7}
    - Newer versions add raise_on_error. :contentReference[oaicite:8]{index=8}
    - In threaded env, community suggests parsing_timeout=None. :contentReference[oaicite:9]{index=9}
    """
    kwargs = {}
    if extraction_config is not None:
        kwargs["extraction_config"] = extraction_config
    if parsing_timeout is not None or "parsing_timeout" in getattr(mv_parse, "__code__", ()).co_varnames:
        kwargs["parsing_timeout"] = parsing_timeout
    if "raise_on_error" in getattr(mv_parse, "__code__", ()).co_varnames:
        kwargs["raise_on_error"] = raise_on_error

    try:
        return mv_parse(text, **kwargs)
    except TypeError:
        # Fall back by dropping optional kwargs if signature differs
        try:
            return mv_parse(text, extraction_config=extraction_config) if extraction_config is not None else mv_parse(text)
        except TypeError:
            return mv_parse(text)


def mv_verify_robust(gold_parsed, answer_parsed, raise_on_error: bool = False) -> bool:
    """
    Call Math-Verify verify() with best-effort compatibility across versions.
    - verify(gold, answer) order matters (README). :contentReference[oaicite:10]{index=10}
    - Newer versions add raise_on_error. :contentReference[oaicite:11]{index=11}
    """
    try:
        if "raise_on_error" in getattr(mv_verify, "__code__", ()).co_varnames:
            return bool(mv_verify(gold_parsed, answer_parsed, raise_on_error=raise_on_error))
        return bool(mv_verify(gold_parsed, answer_parsed))
    except TypeError:
        return bool(mv_verify(gold_parsed, answer_parsed))


# -------------------- reward entrypoint --------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    veRL RewardManager entrypoint.

    Logic:
      1) Enforce your format constraints via check_format()
      2) Extract the final answer content (last \\boxed{...}) WITHOUT truncating to a number
      3) Use Math-Verify parse/verify for robust equivalence checking
      4) total_reward = correctness * format_ok
    """
    # kept for compatibility
    _ = data_source
    regen_prefix = ""
    if isinstance(extra_info, dict):
        regen_prefix = extra_info.get("regen_prefix", "") or ""

    full_text = (regen_prefix or "") + (solution_str or "")
    format_ok = check_format(full_text)  # 0.0 or 1.0
    if format_ok == 0.0:
        return 0.0


    extraction_cfg = _default_extraction_configs()

    parsing_timeout = 10
    raise_on_error = False

    try:
        gold_parsed = mv_parse_robust(
            ground_truth,
            extraction_config=extraction_cfg,
            parsing_timeout=parsing_timeout,
            raise_on_error=raise_on_error,
        )
        ans_parsed = mv_parse_robust(
            solution_str,
            extraction_config=extraction_cfg,
            parsing_timeout=parsing_timeout,
            raise_on_error=raise_on_error,
        )


        is_correct = mv_verify_robust(gold_parsed, ans_parsed, raise_on_error=raise_on_error)
        correctness_reward = 1.0 if is_correct else 0.0
    except Exception:

        correctness_reward = 0.0

    return float(correctness_reward * format_ok)
