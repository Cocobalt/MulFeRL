# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from typing import Any, Dict, List

import datasets


def build_system_prompt() -> str:
    """System prompt requiring <thinking> plus a structured [FEEDBACK] block, then \\boxed{answer}."""
    system_prompt = (
        "You are a reasoning assistant.\n"
        "Solve the problem step by step.\n\n"
        "You MUST follow this exact output format for every problem:\n"
        "1) Wrap ALL your reasoning inside a single <thinking>...</thinking> block.\n"
        "2) At the very start of <thinking>, output exactly ONE <feedback>...</feedback> block.\n"
        "   - The <feedback> must be concise and actionable, and use this structure:\n"
        "     <feedback>\n"
        "     Issue:\n"
        "     1. Likely pitfalls: ...\n"
        "     2. Step-by-step plan:\n"
        "        - (1) ...\n"
        "        - (2) ...\n"
        "        - (3) ...\n\n"
        "     Fix steps:\n"
        "     1. ...\n"
        "     2. ...\n"
        "     3. ...\n"
        "     </feedback>\n"
        "   - Give guidance / a repair plan, but do NOT give a full solution inside <feedback>.\n"
        "   - Do NOT output any expression inside <feedback> that directly equals the final result.\n"
        "     (e.g., do NOT write something like '... = <final answer>' inside <feedback>.)\n"
        "   - You MAY include tiny snippets (a short identity, a one-line correction),\n"
        "     but avoid long derivations or long equations in <feedback>.\n"
        "3) After </thinking>, on a new line, output the final numeric answer in the format:\n"
        "   \\boxed{answer}\n"
        "Do NOT add any extra text after the boxed answer.\n\n"
        "Example (format only):\n"
        "<thinking>\n"
        "<feedback>\n"
        "Issue:\n"
        "1. Likely pitfalls: Misreading quantities; forgetting to combine changes.\n"
        "2. Step-by-step plan:\n"
        "   - (1) Identify the initial quantity and each change.\n"
        "   - (2) Choose the correct operation (add/subtract/etc.).\n"
        "   - (3) Compute carefully.\n\n"
        "Fix steps:\n"
        "1. Extract numbers and what they represent.\n"
        "2. Write the operation clearly.\n"
        "3. Recompute the final arithmetic once.\n"
        "</feedback>\n"
        "Alice starts with 3 apples.\n"
        "She buys 2 more apples.\n"
        "Total apples = 3 + 2 = 5.\n"
        "</thinking>\n"
        "\\boxed{5}\n"
    )

    return system_prompt


def strip_user_instruction(user_content: str) -> str:
    """
    Remove instruction like:
      "Please reason step by step, and put your final answer within \\boxed{}."
    Robust to the instruction appearing as its own line or as a suffix on the same line.
    """
    if not isinstance(user_content, str):
        return ""

    triggers = [
        "Please reason step by step",
        "Please think step by step",
        "Please think",
    ]

    lines: List[str] = user_content.splitlines()
    kept: List[str] = []

    for line in lines:
        # If the trigger appears mid-line, cut off everything from trigger onward.
        cut_pos = None
        for t in triggers:
            p = line.find(t)
            if p != -1:
                cut_pos = p if cut_pos is None else min(cut_pos, p)
        if cut_pos is not None:
            prefix = line[:cut_pos].rstrip()
            if prefix:
                kept.append(prefix)
            # drop the rest (instruction)
            continue

        kept.append(line)

    text = "\n".join(kept).strip()

    # Also remove a couple of common exact suffix patterns if they survived.
    # (Keep this conservative to avoid deleting real problem text.)
    text = re.sub(
        r"\s*Please\s+reason\s+step\s+by\s+step,\s*and\s+put\s+your\s+final\s+answer\s+within\s+\\\\boxed\{\}\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"\s*Please\s+reason\s+step\s+by\s+step\s+and\s+put\s+your\s+final\s+answer\s+within\s+\\\\boxed\{\}\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text


def json_default(obj: Any):
    """Helper for jsonl dumping (handle numpy / arrow types)."""
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _normalize_extra_info(extra_info: Any) -> Dict[str, Any]:
    """Ensure extra_info is a dict. If it's a JSON string, try to parse it."""
    if extra_info is None:
        return {}
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            if isinstance(parsed, dict):
                return parsed
            return {"extra_info_raw": extra_info}
        except Exception:
            return {"extra_info_raw": extra_info}
    # fallback
    return {"extra_info_raw": str(extra_info)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_parquet",
        type=str,
        required=True,
        help="Path to the original parquet file.",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Path to save the rewritten parquet file.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to save the rewritten jsonl file.",
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=-1,
        help="If > 0, use only the first N samples from the input parquet.",
    )

    args = parser.parse_args()

    # 1) load parquet
    ds = datasets.load_dataset("parquet", data_files={"data": args.input_parquet})["data"]

    original_len = len(ds)
    if args.data_size is not None and args.data_size > 0:
        n = min(args.data_size, original_len)
        ds = ds.select(range(n))
        print(f"[data_size] Using {n} / {original_len} samples from {args.input_parquet}")
    else:
        print(f"[data_size] Using full dataset: {original_len} samples")

    new_system_prompt = build_system_prompt()

    # 2) map: update system prompt + clean user + add question_raw into extra_info
    def rewrite_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        prompt = example.get("prompt", [])
        if not isinstance(prompt, list) or len(prompt) == 0:
            return example

        # system at prompt[0]
        if isinstance(prompt[0], dict):
            prompt[0]["role"] = "system"
            prompt[0]["content"] = new_system_prompt

        old_user = ""
        new_user = ""

        # user at prompt[1]
        if len(prompt) >= 2 and isinstance(prompt[1], dict):
            old_user = prompt[1].get("content", "") or ""
            new_user = strip_user_instruction(old_user)
            prompt[1]["role"] = "user"
            prompt[1]["content"] = new_user

        example["prompt"] = prompt

        # add question_raw into extra_info
        extra_info = _normalize_extra_info(example.get("extra_info", None))
        if old_user:
            extra_info.setdefault("question_raw", old_user)
        example["extra_info"] = extra_info

        return example

    ds_new = ds.map(rewrite_fn, with_indices=True)

    # 3) save parquet
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    ds_new.to_parquet(args.output_parquet)

    # 4) save jsonl
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for ex in ds_new:
            f.write(json.dumps(ex, ensure_ascii=False, default=json_default) + "\n")

    print(f"Saved rewritten dataset to:\n  {args.output_parquet}\n  {args.output_jsonl}")


if __name__ == "__main__":
    main()