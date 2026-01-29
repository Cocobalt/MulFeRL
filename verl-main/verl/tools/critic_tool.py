# Copyright 2024+
import json
import logging
import os
import re
from typing import Any, Optional, Tuple
from uuid import uuid4

import requests
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
# 复用官方 search_tool 的并发/限流实现
from .search_tool import init_search_execution_pool, PoolMode

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _chat_once(base_url: str, api_key: str, model: str, user_content: str, timeout_s: float) -> str:
    """调用 OpenAI 兼容 /chat/completions 的最小包装。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""


class ExternalCriticTool(BaseTool):
    """
    Critic 工具：对“完整解题过程（思路+答案）”给出简洁、可执行的反馈，反馈需包在 <fd>...</fd> 中。
    不直接产生 RL 奖励（step reward 与 final reward 均为 0）。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # 基本配置（与 SearchTool 风格一致）
        self.num_workers = int(config.get("num_workers", 120))
        self.rate_limit = int(config.get("rate_limit", 120))
        self.timeout = int(config.get("timeout", 30))
        self.enable_global_rate_limit = bool(config.get("enable_global_rate_limit", True))
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # 外部 LLM 接口
        self.api_base = config.get("api_base", "http://127.0.0.1:8100/v1")
        self.api_key = config.get("api_key", "empty")
        self.model = config.get("model", "qwen3")

        # 模板与标签
        self.fd_open = config.get("fd_open", "<fd>")
        self.fd_close = config.get("fd_close", "</fd>")
        self.feedback_prompt_tmpl = config.get(
            "feedback_prompt_tmpl",
            "You are a strict reviewer.\n"
            f"Critique the FULL solution process below, and output ONLY {self.fd_open}...{self.fd_close}.\n"
            "Focus on reasoning quality:\n"
            "1) Identify incorrect steps, leaps, unsupported claims.\n"
            "2) Point out missing sub-steps, edge cases, or wrong equations.\n"
            "3) Suggest concrete corrections.\n"
            "Do NOT provide a final answer.\n\n"
            "Question:\n{prompt}\n\n"
            "Solution (full trace):\n{answer_full}\n\n"
            f"Return only {self.fd_open}...{self.fd_close}."
        )

        logger.info(f"Initialized ExternalCriticTool with config: "
                    f"num_workers={self.num_workers}, rate_limit={self.rate_limit}, timeout={self.timeout}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """与 SearchTool 对齐：返回 (instance_id, ToolResponse)。"""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "history": [],
            "last_feedback": "",
        }
        return instance_id, ToolResponse()

    # —— 与 SearchTool 对齐：把真正的外部调用逻辑抽成一个方法，并由 execution_pool 调用 ——
    def execute_feedback(
        self,
        instance_id: str,
        prompt: str,
        answer_full: str,
        timeout: int,
        fd_open: str,
        fd_close: str,
    ) -> tuple[str, dict]:
        """构造输入 → 调用外部 LLM → 提取 <fd>...。</fd>；返回 (wrapped_feedback, metadata)。"""
        user_msg = self.feedback_prompt_tmpl.format(prompt=prompt, answer_full=answer_full)
        raw = _chat_once(self.api_base, self.api_key, self.model, user_msg, timeout)
        inner = self._extract_fd(raw, fd_open, fd_close).strip()
        wrapped = f"{fd_open}{inner}{fd_close}" if inner else f"{fd_open}{fd_close}"

        # 记录便于 calc_reward 或调试
        self._instance_dict[instance_id]["last_feedback"] = wrapped
        meta = {
            "status": "ok",
            "chars_in": len(user_msg),
            "chars_out": len(wrapped),
        }
        return wrapped, meta

    def _extract_fd(self, text: str, fd_open: str, fd_close: str) -> str:
        m = re.search(re.escape(fd_open) + r"(.*?)" + re.escape(fd_close), text, re.S)
        return m.group(1).strip() if m else text.strip()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        """
        与 SearchTool.execute 相同的三返回：ToolResponse, step_reward(float), metrics(dict)。
        期望 parameters 至少包含：
          - prompt: str
          - 二选一：answer_full: str 或 messages: list[{role, content}]
        """
        timeout = self.timeout
        prompt = parameters.get("prompt")
        answer_full = parameters.get("answer_full")
        messages = parameters.get("messages")

        # 允许 messages → 拼成 answer_full
        if messages and not answer_full:
            try:
                answer_full = "\n".join(f"[{m.get('role','')}] {m.get('content','')}" for m in messages).strip()
            except Exception:
                answer_full = None

        if not isinstance(prompt, str) or not isinstance(answer_full, str) or not answer_full:
            error_msg = "Error: 'prompt' and either 'answer_full' or 'messages' are required."
            logger.error(f"[ExternalCriticTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {"status": "bad_params"}

        # 通过 execution_pool 执行，结构与 SearchTool 完全一致
        try:
            wrapped_feedback, meta = await self.execution_pool.execute.remote(
                self.execute_feedback, instance_id, prompt, answer_full, timeout, self.fd_open, self.fd_close
            )
            metrics = {
                "status": meta.get("status", "ok"),
                "chars_in": meta.get("chars_in", 0),
                "chars_out": meta.get("chars_out", 0),
            }
            return ToolResponse(text=wrapped_feedback), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Critic execution failed: {e}"})
            logger.error(f"[ExternalCriticTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """不做奖励塑形，返回 0.0（与官方 GSM8K 工具可选奖励不同，这里明确为 0）。"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
