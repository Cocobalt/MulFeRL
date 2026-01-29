# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4
import httpx
from openai import AsyncOpenAI

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from openai import APIStatusError, APIConnectionError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MulFeRLTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        base_url = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        self.judge_stop = ["</feedback>"]
        self.judge_model = config.get("api_model")

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        question: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        if question is None:
            question = kwargs.get("question", None)

        self._instance_dict[instance_id] = {
            "question": question,
            "ground_truth": ground_truth,
            "response": "",
            "reward": 0.0,
            "feedback": "",
        }
        return instance_id, ToolResponse()

    async def _call_judge(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> tuple[float, str, dict]:

        extra_metrics: dict[str, Any] = {}


        try:
            kwargs = {
                "model": self.judge_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 1024,
                "stop": self.judge_stop,
            }


            resp = await self._client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content or ""
            extra_metrics["judge_raw_output"] = content


            content_wo_think = re.sub(r"<think>(.*?)</think>", r"\1", content, flags=re.S | re.I)
            content_wo_think = re.sub(r"</?think>", "", content_wo_think, flags=re.I).strip()

            m_thinking = re.search(r"<thinking>(.*?)</thinking>", content_wo_think, flags=re.S | re.I)
            search_text = m_thinking.group(1) if m_thinking else content_wo_think

            def _extract_between(
                text: str,
                open_pat: str,
                close_pat: str,
                tag_prefix: str,
            ) -> tuple[Optional[str], Optional[str]]:
                m = re.search(open_pat + r"(.*?)" + close_pat, text, flags=re.S | re.I)
                if m:
                    return m.group(1).strip(), f"{tag_prefix}_both"

                start = re.search(open_pat, text, flags=re.S | re.I)
                end = re.search(close_pat, text, flags=re.S | re.I)

                if start and not end:
                    out = text[start.end():].strip()
                    return out, f"{tag_prefix}_start_only"

                if end and not start:
                    out = text[:end.start()].strip()
                    return out, f"{tag_prefix}_end_only"

                if start and end and end.start() > start.end():
                    out = text[start.end():end.start()].strip()
                    return out, f"{tag_prefix}_both_fallback"

                return None, None

            feedback_text, parse_mode = _extract_between(search_text, r"<feedback>", r"</feedback>", "xml")
            if feedback_text is not None:
                extra_metrics["parse_mode"] = parse_mode
                return 0.0, feedback_text, extra_metrics


            extra_metrics["parse_mode"] = "raw"
            return 0.0, feedback_text, extra_metrics

        except Exception as e:
            logger.exception("Error calling  feedback via OpenAI API: %s", e)
            feedback_text = (
                "Judging tool failed to generate feedback. "
                "Please carefully re-check your solution."
            )
            extra_metrics["judge_error"] = str(e)
        return 0.0, feedback_text, extra_metrics

    # -------------------- shorter prompts (cheaper input tokens) --------------------
    def _system_prompt_group(self) -> str:

        return (
            "You are a strict reviewer of an incorrect solution.\n"
            "Please briefly explain (step-by-step) the solution's mistakes and provide your correction suggestions.\n\n"
            "Task:\n"
            "- Identify the earliest/root mistake (the first wrong step).\n"
            "- Explain the errors step-by-step (concise, specific).\n"
            "- Give concrete correction suggestions (how to fix the process).\n\n"
            "Rules:\n"
            "- Do NOT provide a full correct solution.\n"
            "- Do NOT provide or reveal the final numeric answer.\n"
            "- Avoid generic advice; point to specific steps or lines.\n\n"
            "Output ONLY:\n"
            "<feedback>\n"
            "Issue:\n"
            "1. Earliest/root mistake: ...\n"
            "2. Where it first goes wrong (quote 1–2 lines): \"...\"\n"
            "3. Step-by-step errors:\n"
            "   - (1) ...\n"
            "   - (2) ...\n"
            "   - (3) ...\n\n"
            "Fix steps:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "</feedback>"
        )


    def _system_prompt_merge(self) -> str:
        return (
            "You merge multiple feedback comments on an incorrect solution.\n\n"
            "Task:\n"
            "- Combine the feedback into ONE concise, actionable feedback.\n"
            "- Deduplicate repeated points.\n"
            "- Keep ONLY the earliest/root failure modes (the first wrong step and its consequences).\n\n"
            "Rules:\n"
            "- Do NOT provide a full correct solution.\n"
            "- Do NOT provide or reveal the final numeric answer.\n"
            "- Keep it short and specific.\n\n"
            "Output ONLY:\n"
            "<feedback>\n"
            "Issue:\n"
            "1. Earliest/root mistake: ...\n"
            "2. Where it first goes wrong (quote 1–2 lines): \"...\"\n"
            "3. Step-by-step errors:\n"
            "   - (1) ...\n"
            "   - (2) ...\n"
            "   - (3) ...\n\n"
            "Fix steps:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "</feedback>"
        )


    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        inst = self._instance_dict.get(instance_id)
        if inst is None:
            inst = {
                "question": parameters.get("question", None),
                "ground_truth": None,
                "response": "",
                "reward": 0.0,
                "feedback": "",
            }
            self._instance_dict[instance_id] = inst

        mode = parameters.get("mode", "group")
        question = parameters.get("question", "") or inst.get("question") or ""
        extra_metrics: dict[str, Any] = {}
        feedback_text = ""

        if mode == "group":
            solutions = parameters.get("solutions", [])
            if isinstance(solutions, str):
                solutions = [solutions]
            if not isinstance(solutions, (list, tuple)) or len(solutions) == 0:
                solutions = [""]

            solutions_block = "\n\n".join(f"Solution {i + 1}:\n{sol}" for i, sol in enumerate(solutions))

            system_prompt = self._system_prompt_group()
            user_prompt = (
                "Question:\n"
                f"{question}\n\n"
                "Student solutions:\n"
                f"{solutions_block}\n\n"
                "Write feedback in the required format."
            )

            _ignored_score, feedback_text, judge_metrics = await self._call_judge(
                system_prompt, user_prompt
            )
            extra_metrics.update(judge_metrics)

            inst["response"] = "\n\n".join(f"#### {i}: {s}" for i, s in enumerate(solutions))
            inst["reward"] = 0.0
            inst["feedback"] = feedback_text

        elif mode == "merge":
            feedbacks = parameters.get("feedbacks", [])
            if isinstance(feedbacks, str):
                feedbacks = [feedbacks]
            if not isinstance(feedbacks, (list, tuple)) or len(feedbacks) == 0:
                feedbacks = [""]

            cleaned = [fb for fb in feedbacks if isinstance(fb, str) and fb.strip()]
            if len(cleaned) <= 1:
                feedback_text = cleaned[0] if cleaned else ""
            else:
                feedbacks_block = "\n\n".join(f"Feedback {i + 1}:\n{fb}" for i, fb in enumerate(cleaned))
                system_prompt = self._system_prompt_merge()
                user_prompt = (
                    "Question:\n"
                    f"{question}\n\n"
                    "Feedback comments:\n"
                    f"{feedbacks_block}\n\n"
                    "Merge them into ONE feedback in the required format."
                )

                _ignored_score, feedback_text, judge_metrics = await self._call_judge(
                    system_prompt, user_prompt
                )
                extra_metrics.update(judge_metrics)

            inst["merged_feedbacks"] = feedbacks
            inst["final_feedback"] = feedback_text

        else:
            feedback_text = f"Invalid mode '{mode}'. Expected 'group' or 'merge'."
            extra_metrics["error"] = "invalid_mode"

        tool_reward = 0.0
        return ToolResponse(text=feedback_text), tool_reward, extra_metrics
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
