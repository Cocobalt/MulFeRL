"""Metric aggregation helpers."""

from typing import Any

import numpy as np


def _to_scalar(x: Any) -> float:
    import torch

    if x is None:
        return np.nan

    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return np.nan
        return float(x.detach().float().mean().cpu().item())

    if isinstance(x, (np.ndarray, np.generic)):
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return np.nan
        return float(arr.mean())

    if isinstance(x, (list, tuple)):
        flat = []
        stack = list(x)
        while stack:
            value = stack.pop()
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                stack.extend(list(value))
            elif isinstance(value, (np.ndarray, np.generic)):
                flat.extend(np.asarray(value).reshape(-1).tolist())
            else:
                flat.append(value)

        vals = []
        for value in flat:
            try:
                vals.append(float(value))
            except Exception:
                continue
        return float(np.mean(vals)) if vals else np.nan

    try:
        return float(x)
    except Exception:
        return np.nan


def sanitize_metrics(metrics: dict[str, list[Any]]) -> dict[str, list[float]]:
    import torch

    out: dict[str, list[float]] = {}
    for key, vals in metrics.items():
        out[key] = []
        for value in vals or []:
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                print(f"[WARN] metric {key} has non-scalar tensor {tuple(value.shape)}")
            if isinstance(value, np.ndarray) and value.size != 1:
                print(f"[WARN] metric {key} has non-scalar ndarray {value.shape}")
            out[key].append(_to_scalar(value))
    return out


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    metrics = sanitize_metrics(metrics)

    reduced = {}
    for key, val in metrics.items():
        if not val:
            reduced[key] = np.nan
            continue

        if "max" in key:
            reduced[key] = float(np.max(val))
        elif "min" in key:
            reduced[key] = float(np.min(val))
        else:
            reduced[key] = float(np.mean(val))

    return reduced
