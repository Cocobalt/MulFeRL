# Copyright 2024+
#
# feedback_utils.py
#
# A utility module mirroring the structure of `search_r1_like_utils.py`,
# but for calling an OpenAI-compatible chat endpoint to generate
# per-sample feedback wrapped in <fd>...</fd>.
#
# Architecture kept the same: retry policy, batch-style entry function,
# tuple returns, rich metadata, unique request id logging.

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
import re
from typing import Any, Optional, Dict, List

import requests

DEFAULT_TIMEOUT = 30  # Default request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)


# ----------------------------- helpers ---------------------------------

def _extract_fd(text: str, fd_open: str, fd_close: str) -> str:
    m = re.search(re.escape(fd_open) + r"(.*?)" + re.escape(fd_close), text, re.S)
    return m.group(1).strip() if m else text.strip()


def _build_feedback_prompt(prompt: str, answer_full: str, fd_open: str, fd_close: str) -> str:
    # Keep the prompt concise and deterministic; callers can override at higher level if needed.
    return (
        "You are a strict reviewer.\n"
        f"Critique the FULL solution process below, and output ONLY {fd_open}...{fd_close}.\n"
        "Focus on reasoning quality:\n"
        "1) Identify incorrect steps, leaps, unsupported claims.\n"
        "2) Point out missing sub-steps, edge cases, or wrong equations.\n"
        "3) Suggest concrete corrections.\n"
        "Do NOT provide a final answer.\n\n"
        f"Question:\n{prompt}\n\n"
        f"Solution (full trace):\n{answer_full}\n\n"
        f"Return only {fd_open}...{fd_close}."
    )


# ------------------------- low-level caller -----------------------------

def call_feedback_api(
    api_base: str,
    api_key: str,
    model: str,
    samples: List[Dict[str, str]],  # each: {"prompt": str, "answer_full": str}
    fd_open: str = "<fd>",
    fd_close: str = "</fd>",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Call an OpenAI-compatible /chat/completions endpoint to generate feedback for each sample.

    This mirrors `call_search_api` in spirit: returns (response_json, error_message).
    On success, response_json = {"result": [{"feedback": str, "raw": str, "prompt": str}]}
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Feedback Request ID: {request_id}] "

    url = "http://0.0.0.0:8100/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # We'll call per-sample (most chat servers don't batch multiple prompts into one call reliably).
    results: List[Dict[str, Any]] = []

    for idx, s in enumerate(samples):
        last_error = None
        prompt = s.get("prompt", "")
        answer_full = s.get("answer_full", "")
        user_content = _build_feedback_prompt(prompt, answer_full, fd_open, fd_close)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"{log_prefix}Sample {idx} attempt {attempt+1}/{MAX_RETRIES}: POST {url}"
                )
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)

                if resp.status_code in [500, 502, 503, 504]:
                    last_error = (
                        f"{log_prefix}Server Error ({resp.status_code}) for sample {idx} on attempt {attempt+1}"
                    )
                    logger.warning(last_error)
                    if attempt < MAX_RETRIES - 1:
                        delay = INITIAL_RETRY_DELAY * (attempt + 1)
                        logger.info(f"{log_prefix}Retrying after {delay}s...")
                        time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                raw = (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
                inner = _extract_fd(raw, fd_open, fd_close)
                wrapped = f"{fd_open}{inner}{fd_close}" if inner else f"{fd_open}{fd_close}"

                results.append({
                    "feedback": wrapped,
                    "raw": raw,
                    "prompt": prompt,
                    "answer_len": len(answer_full),
                })
                logger.info(f"{log_prefix}Sample {idx} success on attempt {attempt+1}")
                break

            except requests.exceptions.ConnectionError as e:
                last_error = f"{log_prefix}Connection Error for sample {idx}: {e}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay}s...")
                    time.sleep(delay)
                continue
            except requests.exceptions.Timeout as e:
                last_error = f"{log_prefix}Timeout Error for sample {idx}: {e}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay}s...")
                    time.sleep(delay)
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"{log_prefix}API Request Error for sample {idx}: {e}"
                break
            except json.JSONDecodeError as e:
                raw_text = resp.text if "resp" in locals() else "N/A"
                last_error = f"{log_prefix}JSON Decode Error for sample {idx}: {e}, Response: {raw_text[:200]}"
                break
            except Exception as e:
                last_error = f"{log_prefix}Unexpected Error for sample {idx}: {e}"
                break
        else:
            # Exhausted retries without `break`
            pass

        if last_error and (len(results) <= idx):
            # push a placeholder failure record to keep alignment
            results.append({
                "feedback": f"{fd_open}Feedback unavailable: {last_error.replace(log_prefix, '')}{fd_close}",
                "raw": None,
                "prompt": prompt,
                "answer_len": len(answer_full),
                "error": last_error.replace(log_prefix, "")
            })

    # Wrap as a single JSON like search util does
    response_json = {"result": results}
    return response_json, None


# ----------------------- pretty / batch wrapper -------------------------

def _feedbacks2string(result_items: List[Dict[str, Any]]) -> str:
    """Convert feedback list to a human-readable block (for logging / debugging)."""
    blocks = []
    for i, item in enumerate(result_items):
        fb = item.get("feedback", "")
        pr = (item.get("prompt", "") or "").splitlines()[0][:80]
        blocks.append(f"Sample {i+1} (Prompt head: {pr})\n{fb}")
    return "\n\n".join(blocks).strip()


def perform_single_feedback_batch(
    api_base: str,
    api_key: str,
    model: str,
    samples: List[Dict[str, str]],  # each: {"prompt": str, "answer_full": str}
    fd_open: str = "<fd>",
    fd_close: str = "</fd>",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """
    Batch-style feedback generation, mirroring perform_single_search_batch:

    Returns (result_text, metadata), where result_text is a JSON string.
    metadata contains useful counters and the pretty-printed string for quick viewing.
    """
    logger.info(f"Starting batch feedback for {len(samples)} samples.")

    api_response: Optional[dict[str, Any]] = None
    error_msg: Optional[str] = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_feedback_api(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    samples=samples,
                    fd_open=fd_open,
                    fd_close=fd_close,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_feedback_api(
                api_base=api_base,
                api_key=api_key,
                model=model,
                samples=samples,
                fd_open=fd_open,
                fd_close=fd_close,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch feedback: {e}"
        logger.error(f"Batch feedback: {error_msg}")
        traceback.print_exc()

    # Assemble metadata
    metadata: dict[str, Any] = {
        "sample_count": len(samples),
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "formatted_result": None,
    }

    result_text = json.dumps({"result": "Feedback request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Feedback error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Batch feedback: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch feedback: API Response: {api_response}")
        metadata["api_response"] = api_response
        try:
            items = api_response.get("result", []) or []
            pretty = _feedbacks2string(items)
            result_text = json.dumps({"result": pretty}, ensure_ascii=False)
            metadata["status"] = "success"
            metadata["formatted_result"] = pretty
            logger.info(f"Batch feedback: Successful, got {len(items)} items")
        except Exception as e:
            error_msg = f"Error processing feedback results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Batch feedback: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error("Batch feedback: Unknown API state.")

    return result_text, metadata
