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
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import gsm8k
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from openai import AsyncOpenAI
import json
import re
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Gsm8kTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        base_url = os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8100/v1")
        api_key = os.getenv("OPENAI_API_KEY", "empty")

        # 判题使用的模型名（你部署的是 qwen3）
        self.judge_model = config.get("api_model", "qwen3")

        # Async OpenAI client
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._instance_dict = {}


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema


    async def create(
        self, instance_id: Optional[str] = None, question: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
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

    # @rollout_trace_op
    # async def execute(
    #     self,
    #     instance_id: str,
    #     parameters: dict[str, Any],
    #     **kwargs,
    # ) -> tuple[ToolResponse, float, dict]:
    #     inst = self._instance_dict.get(instance_id)

    #     mode = parameters.get("mode", "group")  # 默认 group

    #     question = parameters.get("question", "") or inst.get("question") or ""
    #     # print("mode in tool feedback group:", mode)
    #     # print("question in tool feedback group:", question)
    #     ground_truth = inst.get("ground_truth") or ""

    #     extra_metrics: dict[str, Any] = {}
    #     feedback_text = ""
    #     score = 0.0

    #     if mode == "group":
    #         # ========= 第一阶段：一题多解 =========
    #         solutions = parameters.get("solutions", [])
    #         if isinstance(solutions, str):
    #             solutions = [solutions]
    #         if not isinstance(solutions, (list, tuple)) or len(solutions) == 0:
    #             solutions = [""]  # 至少给一个防止后面挂掉

    #         # 这里你可以：
    #         # - 对每个 solution 单独打分再平均
    #         # - 或者选最好的，或返回一个总体 feedback
    #         # 下面先用一个极简 placeholder：
    #         # print("question in tool feedback group:", question)
    #         feedback_text = (
    #             # parameters.get("question", "NO QUESTION PROVIDED")
    #             f"Question: {question}\n"
    #             f"Ground Truth: {ground_truth}\n"
    #             f"Number of Solutions: {len(solutions)}"
    #         )
    #         score = 0.0  # 可以取 max / mean 等

    #         # 例如把所有 solution 存一下，方便 calc_reward 用
    #         inst["response"] = "\n\n".join(f"#### {i}: {s}" for i, s in enumerate(solutions))
    #         inst["reward"] = score
    #         inst["feedback"] = feedback_text

    #     elif mode == "merge":
    #         # ========= 第二阶段：一题多 feedback 合并 =========
    #         feedbacks = parameters.get("feedbacks", [])
    #         if isinstance(feedbacks, str):
    #             feedbacks = [feedbacks]
    #         if not isinstance(feedbacks, (list, tuple)) or len(feedbacks) == 0:
    #             feedbacks = [""]

    #         # TODO: 这里写你真正的 merge 逻辑，
    #         # 比如调用 judge_model 再总结一次，或者简单拼接：
    #         feedback_text = "Merged FEEDABCK"
    #         feedback_text += " hahaha ".join(feedbacks)
    #         # merge 阶段一般不再改 score，可以保持 0 或从 kwargs 里拿
    #         score = 0.0

    #         inst["merged_feedbacks"] = feedbacks
    #         inst["final_feedback"] = feedback_text

    #     else:
    #         # 意料之外的 mode，当作错误处理
    #         feedback_text = f"Invalid mode '{mode}'. Expected 'group' or 'merge'."
    #         score = 0.0
    #         extra_metrics["error"] = "invalid_mode"

    #     tool_reward = 0.0
    #     return ToolResponse(text=feedback_text), tool_reward, extra_metrics
    
    
    
    # async def _call_judge(
    #     self,
    #     system_prompt: str,
    #     user_prompt: str,
    #     max_tokens: Optional[int] = None,
    # ) -> tuple[float, str, dict]:
    #     """调用后端模型并解析 JSON，只返回 feedback；score 一律 0.0。"""
    #     extra_metrics: dict[str, Any] = {}
    #     feedback_text = ""
    #     score = 0.0  # 统一保持 0

    #     try:
    #         kwargs = {
    #             "model": self.judge_model,
    #             "messages": [
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt},
    #             ],
    #             "temperature": 0,
    #             "max_tokens":4096
    #         }
    #         # 在这里硬控长度 ✅
    #         if max_tokens is not None:
    #             kwargs["max_tokens"] = int(max_tokens)

    #         resp = await self._client.chat.completions.create(**kwargs)

    #         content = resp.choices[0].message.content or ""
    #         logger.info(f"[Gsm8kTool] judge raw content: {repr(content)}")

    #         extra_metrics["judge_raw_output"] = content
    #         data = None

    #         # 1) 直接整体解析
    #         try:
    #             data = json.loads(content.strip())
    #         except json.JSONDecodeError:
    #             # 2) 从 ```json ... ``` 代码块里抽
    #             fence = re.search(
    #                 r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S | re.I
    #             )
    #             candidates = []
    #             if fence:
    #                 candidates.append(fence.group(1))

    #             # 3) 抓所有 {...}，逐个尝试
    #             candidates.extend(re.findall(r"\{.*?\}", content, re.S))

    #             for cand in candidates:
    #                 try:
    #                     data = json.loads(cand)
    #                     break
    #                 except json.JSONDecodeError:
    #                     continue

    #             if data is None:
    #                 raise ValueError(
    #                     f"Cannot parse JSON from model output: {repr(content)}"
    #                 )

    #         # data 一定是 dict；只关心 feedback，score 忽略
    #         feedback_text = str(data.get("feedback", ""))

    #         extra_metrics["judge_parsed"] = data

    #     except Exception as e:
    #         logger.exception("Error calling GSM8K judge via OpenAI API: %s", e)
    #         feedback_text = (
    #             "Judging tool failed to parse structured feedback from the judge model. "
    #             "Please carefully re-check your solution."
    #         )
    #         score = 0.0
    #         extra_metrics["judge_error"] = str(e)

    #     # 对外统一返回 score=0.0，由外部 calc_reward 决定真正奖励
    #     return 0.0, feedback_text, extra_metrics



    # @rollout_trace_op
    # async def execute(
    #     self,
    #     instance_id: str,
    #     parameters: dict[str, Any],
    #     **kwargs,
    # ) -> tuple[ToolResponse, float, dict]:
    #     inst = self._instance_dict.get(instance_id)
    #     if inst is None:
    #         # 正常流程先 create 再 execute，这里兜一下
    #         inst = {
    #             "question": parameters.get("question", None),
    #             "ground_truth": None,
    #             "response": "",
    #             "reward": 0.0,
    #             "feedback": "",
    #         }
    #         self._instance_dict[instance_id] = inst

    #     mode = parameters.get("mode", "group")  # 默认 group
    #     feedback_length = parameters.get("feedback_length", None)

    #     # 只用 question 构造 prompt，ground_truth 不再发给判题模型
    #     question = parameters.get("question", "") or inst.get("question") or ""

    #     extra_metrics: dict[str, Any] = {}
    #     feedback_text = ""
    #     score = 0.0  # 这个 score 不再作为 RL 奖励用

    #     if mode == "group":
    #         # ========= 第一阶段：一题多解 =========
    #         solutions = parameters.get("solutions", [])
    #         if isinstance(solutions, str):
    #             solutions = [solutions]
    #         if not isinstance(solutions, (list, tuple)) or len(solutions) == 0:
    #             solutions = [""]  # 至少给一个，防止 prompt 完全为空

    #         solutions_block = "\n\n".join(
    #             f"Solution {i + 1}:\n{sol}" for i, sol in enumerate(solutions)
    #         )

    #         system_prompt = (
    #             "You are a strict math tutor for GSM8K-style math word problems. "
    #             "Given a problem and several student solutions, you must provide feedback "
    #             "that helps the student understand mistakes and improve.\n"
    #             "You MUST output a single JSON object with exactly these fields:\n"
    #             '  - \"feedback\": a short explanation in English or Chinese.\n'
    #             "Do not output anything except that JSON object."
    #         )

    #         user_prompt = (
    #             "Question:\n"
    #             f"{question}\n\n"
    #             "Student solutions:\n"
    #             f"{solutions_block}\n\n"
    #             "Now analyze these solutions and return the JSON."
    #         )

    #         # 不再传 ground_truth 给判题模型
    #         _ignored_score, feedback_text, judge_metrics = await self._call_judge(
    #             system_prompt, user_prompt
    #         )
    #         extra_metrics.update(judge_metrics)

    #         # 记录到实例里，方便后续 calc_reward / logging（ground_truth 仍然只留在 inst 里）
    #         inst["response"] = "\n\n".join(
    #             f"#### {i}: {s}" for i, s in enumerate(solutions)
    #         )
    #         inst["reward"] = 0.0
    #         inst["feedback"] = feedback_text

    #     elif mode == "merge":
    #         # ========= 第二阶段：一题多 feedback 合并 =========
    #         feedbacks = parameters.get("feedbacks", [])
    #         if isinstance(feedbacks, str):
    #             feedbacks = [feedbacks]
    #         if not isinstance(feedbacks, (list, tuple)) or len(feedbacks) == 0:
    #             feedbacks = [""]

    #         feedbacks_block = "\n\n".join(
    #             f"Feedback {i + 1}:\n{fb}" for i, fb in enumerate(feedbacks)
    #         )

    #         system_prompt = (
    #             "You are an expert tutor. You will be given a math question and several "
    #             "feedback comments about a student's solution. Your task is to merge these "
    #             "comments into a single, clear feedback for the student.\n"
    #             "You MUST output a single JSON object with exactly these fields:\n"
    #             '  - \"feedback\": the merged feedback string.\n'
    #             "Do not output anything except that JSON object."
    #         )

    #         user_prompt = (
    #             "Question:\n"
    #             f"{question}\n\n"
    #             "Feedback comments:\n"
    #             f"{feedbacks_block}\n\n"
    #             "Now merge these comments into one concise feedback and return the JSON."
    #         )

    #         _ignored_score, feedback_text, judge_metrics = await self._call_judge(
    #             system_prompt, user_prompt
    #         )
    #         extra_metrics.update(judge_metrics)

    #         inst["merged_feedbacks"] = feedbacks
    #         inst["final_feedback"] = feedback_text

    #     else:
    #         # 意料之外的 mode，当作错误处理
    #         feedback_text = f"Invalid mode '{mode}'. Expected 'group' or 'merge'."
    #         score = 0.0
    #         extra_metrics["error"] = "invalid_mode"

    #     # 工具本身不直接给 PPO 奖励，统一设为 0，由外部 rm / adv_estimator 决定
    #     tool_reward = 0.0
    #     return ToolResponse(text=feedback_text), tool_reward, extra_metrics
    
    async def _call_judge( 
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> tuple[float, str, dict]:
        """
        调用后端模型并解析文本格式：

        期望格式优先级：
        1. [FEEDBACK] ... [/FEEDBACK]
        2. [FEEDBACK] ...    （只有开标签）
        3.      ... [/FEEDBACK]（只有闭标签）
        4. 都没有 → 整段当 feedback
        """
        extra_metrics: dict[str, Any] = {}
        feedback_text = ""
        score = 0.0  # 统一保持 0

        try:
            kwargs = {
                "model": self.judge_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 2048,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

            resp = await self._client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content or ""
            logger.info(f"[Gsm8kTool] judge raw content: {repr(content)}")
            extra_metrics["judge_raw_output"] = content

            # 一些 reasoner 模型会在前面带 <think>...</think>，先撸掉
            content_wo_think = re.sub(
                r"<think>.*?</think>", "", content, flags=re.S | re.I
            ).strip()

            # 优先：成对 [FEEDBACK] ... [/FEEDBACK]
            m = re.search(
                r"\[FEEDBACK\](.*)\[/FEEDBACK\]",
                content_wo_think,
                flags=re.S | re.I,
            )

            if m:
                feedback_text = m.group(1).strip()
                extra_metrics["parse_mode"] = "tag_both"
            else:
                # 只有开标签 / 只有闭标签 / 都没有 的情况
                start = re.search(r"\[FEEDBACK\]", content_wo_think, flags=re.S | re.I)
                end = re.search(r"\[/FEEDBACK\]", content_wo_think, flags=re.S | re.I)

                if start and end and end.start() > start.end():
                    # 理论上不会走到这里，因为上面的成对 regex 已经覆盖了，
                    # 这里当作兜底
                    feedback_text = content_wo_think[start.end():end.start()].strip()
                    extra_metrics["parse_mode"] = "tag_both_fallback"
                elif start and not end:
                    # 只有 [FEEDBACK] → 取到结尾
                    feedback_text = content_wo_think[start.end():].strip()
                    extra_metrics["parse_mode"] = "tag_start_only"
                elif end and not start:
                    # 只有 [/FEEDBACK] → 取从开头到它之前
                    feedback_text = content_wo_think[: end.start()].strip()
                    extra_metrics["parse_mode"] = "tag_end_only"
                else:
                    # 完全没有 tag → 整段当 feedback
                    feedback_text = content_wo_think.strip()
                    # feedback_text = content.strip()
                    extra_metrics["parse_mode"] = "raw"

        except Exception as e:
            logger.exception("Error calling GSM8K judge via OpenAI API: %s", e)
            feedback_text = (
                "Judging tool failed to generate feedback. "
                "Please carefully re-check your solution."
            )
            score = 0.0
            extra_metrics["judge_error"] = str(e)

        return 0.0, feedback_text, extra_metrics

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
        feedback_length = parameters.get("feedback_length", None)

        question = parameters.get("question", "") or inst.get("question") or ""

        extra_metrics: dict[str, Any] = {}
        feedback_text = ""
        score = 0.0

        if mode == "group":
            # ========= 第一阶段：一题多解 =========
            solutions = parameters.get("solutions", [])
            if isinstance(solutions, str):
                solutions = [solutions]
            if not isinstance(solutions, (list, tuple)) or len(solutions) == 0:
                solutions = [""]

            solutions_block = "\n\n".join(
                f"Solution {i + 1}:\n{sol}" for i, sol in enumerate(solutions)
            )

            system_prompt = (
                "You are a strict math tutor for GSM8K-style word problems. "
                "Given a problem and several student solutions, you should analyze common mistakes "
                "and give constructive feedback.\n\n"
                "Return the feedback in the following format:\n"
                "[FEEDBACK]\n"
                "<your feedback text here>\n"
                "[/FEEDBACK]\n\n"
                "Do NOT output anything outside these [FEEDBACK] tags."
            )

            user_prompt = (
                "Question:\n"
                f"{question}\n\n"
                "Student solutions:\n"
                f"{solutions_block}\n\n"
                "Now analyze these solutions and write feedback in the specified format."
            )

            _ignored_score, feedback_text, judge_metrics = await self._call_judge(
                system_prompt, user_prompt
            )
            extra_metrics.update(judge_metrics)

            inst["response"] = "\n\n".join(
                f"#### {i}: {s}" for i, s in enumerate(solutions)
            )
            inst["reward"] = 0.0
            inst["feedback"] = feedback_text

        elif mode == "merge":
            # ========= 第二阶段：一题多 feedback 合并 =========
            feedbacks = parameters.get("feedbacks", [])
            if isinstance(feedbacks, str):
                feedbacks = [feedbacks]
            if not isinstance(feedbacks, (list, tuple)) or len(feedbacks) == 0:
                feedbacks = [""]

            feedbacks_block = "\n\n".join(
                f"Feedback {i + 1}:\n{fb}" for i, fb in enumerate(feedbacks)
            )

            system_prompt = (
                "You are an expert tutor. You will be given a math question and several feedback "
                "comments. Your task is to merge these comments into a single, clear feedback.\n\n"
                "Return the merged feedback in the following format:\n"
                "[FEEDBACK]\n"
                "<your merged feedback here>\n"
                "[/FEEDBACK]\n\n"
                "Do NOT output anything outside these [FEEDBACK] tags."
            )

            user_prompt = (
                "Question:\n"
                f"{question}\n\n"
                "Feedback comments:\n"
                f"{feedbacks_block}\n\n"
                "Now merge these comments and write the final feedback in the specified format."
            )

            _ignored_score, feedback_text, judge_metrics = await self._call_judge(
                system_prompt, user_prompt
            )
            extra_metrics.update(judge_metrics)

            inst["merged_feedbacks"] = feedbacks
            inst["final_feedback"] = feedback_text
        else:
            # 意料之外的 mode，当作错误处理
            feedback_text = f"Invalid mode '{mode}'. Expected 'group' or 'merge'."
            score = 0.0
            extra_metrics["error"] = "invalid_mode"
        tool_reward = 0.0
        return ToolResponse(text=feedback_text), tool_reward, extra_metrics
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
