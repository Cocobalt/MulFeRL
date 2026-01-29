# # Copyright 2025 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Metrics utils.
# """

# from typing import Any

# import numpy as np


# def _to_scalar(x):
#     import numpy as np
#     import torch
#     if isinstance(x, torch.Tensor):
#         return float(x.detach().float().mean().cpu().item())
#     if isinstance(x, np.ndarray):
#         return float(np.asarray(x, dtype=np.float32).mean())
#     if isinstance(x, (list, tuple)):
#         return float(np.asarray(x, dtype=np.float32).mean())  # 若 ragged 仍可能报错
#     return float(x)

# def sanitize_metrics(metrics):
#     out = {}
#     for k, vals in metrics.items():
#         out[k] = [_to_scalar(v) for v in vals]
#     return out


# def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
#     """
#     Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
#     The reduce operation is determined by the key name:
#     - If the key contains "max", np.max is used
#     - If the key contains "min", np.min is used
#     - Otherwise, np.mean is used

#     Args:
#         metrics: A dictionary mapping metric names to lists of metric values.

#     Returns:
#         A dictionary with the same keys but with each list replaced by its reduced value.

#     Example:
#         >>> metrics = {
#         ...     "loss": [1.0, 2.0, 3.0],
#         ...     "accuracy": [0.8, 0.9, 0.7],
#         ...     "max_reward": [5.0, 8.0, 6.0],
#         ...     "min_error": [0.1, 0.05, 0.2]
#         ... }
#         >>> reduce_metrics(metrics)
#         {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
#     """
#     metrics = sanitize_metrics(metrics)
#     for key, val in metrics.items():
#         if "max" in key:
#             metrics[key] = np.max(val)
#         elif "min" in key:
#             metrics[key] = np.min(val)
#         else:
#             metrics[key] = np.mean(val)
#     return metrics

from typing import Any
import numpy as np

def _to_scalar(x):
    import torch

    # None 防御
    if x is None:
        return np.nan

    # torch tensor
    if isinstance(x, torch.Tensor):
        # mean over all elements -> scalar
        return float(x.detach().float().mean().cpu().item())

    # numpy array / numpy scalar
    if isinstance(x, (np.ndarray, np.generic)):
        arr = np.asarray(x, dtype=np.float32)
        return float(arr.mean())

    # python list/tuple：先转 object array，再 flatten 收集可转 float 的元素
    if isinstance(x, (list, tuple)):
        # 尽量把嵌套展开成一维标量列表
        flat = []
        stack = list(x)
        while stack:
            v = stack.pop()
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                stack.extend(list(v))
            elif isinstance(v, (np.ndarray, np.generic)):
                flat.extend(np.asarray(v).reshape(-1).tolist())
            else:
                flat.append(v)
        if len(flat) == 0:
            return np.nan
        # 过滤掉不能转 float 的
        vals = []
        for v in flat:
            try:
                vals.append(float(v))
            except Exception:
                pass
        return float(np.mean(vals)) if len(vals) > 0 else np.nan

    # 普通数字
    try:
        return float(x)
    except Exception:
        return np.nan


def sanitize_metrics(metrics):
    import torch
    import numpy as np
    out = {}
    for k, vals in metrics.items():
        out[k] = []
        for v in (vals or []):
            if isinstance(v, torch.Tensor) and v.numel() != 1:
                print(f"[WARN] metric {k} has non-scalar tensor {tuple(v.shape)}")
            if isinstance(v, np.ndarray) and v.size != 1:
                print(f"[WARN] metric {k} has non-scalar ndarray {v.shape}")
            out[k].append(_to_scalar(v))
    return out



def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    metrics = sanitize_metrics(metrics)

    reduced = {}
    for key, val in metrics.items():
        # 空 list 防御：直接跳过或返回 nan（你二选一）
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
