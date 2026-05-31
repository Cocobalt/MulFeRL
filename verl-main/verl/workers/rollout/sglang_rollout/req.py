# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Generator, Optional
from uuid import uuid4

import numpy as np
import ray
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.device import get_visible_devices_keyword
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj, get_named_tensor_buckets
from verl.workers.rollout.utils import is_valid_ipv6_address

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool

# compatible with sglang 0.5.3
try:
    from sglang.srt.utils import get_ip
except ImportError:
    from sglang.srt.utils import get_local_ip_auto as get_ip

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def _async_rollout_a_request(
    self,
    req: AsyncRolloutRequest,
    do_sample: bool = True,
    is_validate: bool = False,
    **kwargs,
) -> AsyncRolloutRequest:
    assert self._tp_rank == 0, "only the master process can call this function"
    _req = deepcopy(req)
    finish_reason_type = None
    output = None

    current_turns = 0
    user_turns = 0
    user_turn_rewards = []

    # Create request-level sampling parameters
    request_sampling_params = self.sampling_params.copy()
    if not do_sample:
        request_sampling_params.update(
            {
                "n": 1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "ignore_eos": False,
                "min_new_tokens": 0,
                "max_new_tokens": self.config.response_length,
                "skip_special_tokens": True,
                "spaces_between_special_tokens": True,
            }
        )
    elif is_validate:
        request_sampling_params.update(
            {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        )

    # Update with any additional kwargs
    request_sampling_params.update(kwargs)

    while current_turns < self.config.multi_turn.max_assistant_turns:
        if _req.state == AsyncRolloutRequestStateEnum.PENDING:
            await self._handle_pending_state(_req)
            _req.state = AsyncRolloutRequestStateEnum.RUNNING
        elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
            if _req.messages[-1].tool_calls is not None:
                parsed_tool_calls = _req.messages[-1].tool_calls
                if self.config.skip_tokenizer_init:
                    _req.messages[-1].tool_calls = None
                tool_call_results = await asyncio.gather(
                    *[
                        self._tool_map[tool_call.function.name].execute(
                            _req.request_id,
                            tool_call.function.arguments,
                            **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {}),
                        )
                        for tool_call in parsed_tool_calls
                    ]
                )
                _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                    _req.update_metrics(metrics, tool_call.function.name)
                if _req.input_ids.size(-1) >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    break
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            else:
                raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
        elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
            # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
            # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
            # token accounts for the EOS token).
            prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))

            if prompt_length + 1 >= self.config.max_model_len:
                finish_reason_type = FinishReasonTypeEnum.LENGTH
                break

            # Video support is not implemented yet
            image_data = (
                _req.multi_modal_data["image"]
                if _req.multi_modal_data and "image" in _req.multi_modal_data
                else None
            )
            video_data = (
                _req.multi_modal_data["video"]
                if _req.multi_modal_data and "video" in _req.multi_modal_data
                else None
            )
            if video_data:
                logger.warning(
                    "video support is not implemented yet, current length of video data is %d", len(video_data)
                )

            output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
            if self.config.skip_tokenizer_init:
                content_ids = output["output_ids"]
                content = self.processing_class.decode(content_ids, skip_special_tokens=True)
                content_ids = torch.tensor(
                    content_ids, dtype=_req.input_ids.dtype, device=_req.input_ids.device
                ).unsqueeze(0)
            else:
                content_ids = None
                content = output["text"]

            finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
            current_turns += 1
            if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                break
            else:
                if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                    finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                    try:
                        normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                    except JSONDecodeError:
                        normed_content = content
                        tool_calls = []
                    except AttributeError:
                        normed_content = content
                        tool_calls = []
                    parsed_tool_calls = []
                    for tool_call in tool_calls:
                        function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                            OpenAIFunctionParsedSchema(
                                name=tool_call.name,
                                arguments=tool_call.parameters,
                            )
                        )
                        # Drop the tool call if its arguments has decode error
                        if has_decode_error:
                            continue
                        parsed_tool_calls.append(
                            OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index),
                                function=function,
                            )
                        )
                    if len(parsed_tool_calls) > 0:
                        _req.add_assistant_message(
                            # since the content is updated, we just pass the content not content_ids
                            self.processing_class,
                            content=normed_content,
                            tool_calls=parsed_tool_calls,
                        )
                    else:
                        _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                        break
                else:
                    _req.add_assistant_message(
                        self.processing_class,
                        content=content,
                        content_ids=content_ids,
                    )
                    if (
                        _req.interaction_kwargs
                        and self.interaction_map
                        and user_turns < self.config.multi_turn.max_user_turns
                        and current_turns < self.config.multi_turn.max_assistant_turns
                    ):
                        _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                    else:
                        # Add ending condition
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                        break
        elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
            user_turns += 1
            messages = [{"role": x.role, "content": x.content} for x in _req.messages]

            # Get interaction by name from interaction_kwargs
            interaction_name = _req.interaction_kwargs.get(
                "name", "gsm8k"
            )  # Default to gsm8k for backward compatibility
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )

            interaction = self.interaction_map[interaction_name]

            should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                _req.request_id, messages, **_req.interaction_kwargs
            )
            user_turn_rewards.append(reward)
            # Add turn check
            if (
                should_terminate_sequence
                or user_turns > self.config.multi_turn.max_user_turns
                or current_turns > self.config.multi_turn.max_assistant_turns
            ):
                finish_reason_type = FinishReasonTypeEnum.STOP
                _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                break
            else:
                _req.add_user_message(self.processing_class, content)
                if _req.input_ids.size(-1) >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    break
                else:
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING

    if current_turns >= self.config.multi_turn.max_assistant_turns:
        finish_reason_type = FinishReasonTypeEnum.STOP

    # Calculate the reward for each tool
    async def calc_reward_and_release_fn(name: str, tool: BaseTool):
        reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
        await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
        return name, reward

    tool_reward_tasks = []
    for name in _req.tools_kwargs.keys():
        tool = self._tool_map[name]
        tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
    tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
    tool_reward_scores = dict(tool_reward_scores)
    all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
    _req.finalize(self.processing_class, all_rewards, finish_reason_type)

    if self.config.calculate_log_probs:
        debug_sampling_params = {**self.sampling_params}
        debug_sampling_params["max_new_tokens"] = 0
        output = await self._engine.async_generate(
            prompt=None,
            input_ids=_req.input_ids,
            sampling_params=debug_sampling_params,
            return_logprob=True,
            logprob_start_len=0,
        )
        # len(input_token_logprobs) = len(input_tokens)-1，because logprob of 1st token is None
        _req.output_token_ids, _req.rollout_log_probs = _extract_logprob_from_output(output)
    return _req

async def _handle_engine_call(
    self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
) -> dict:
    generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
    return await self._handle_engine_generate(generation_prompt_ids, sampling_params, image_data)

async def _handle_engine_generate(
    self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None
) -> dict:
    max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)

    kwargs = sampling_params.copy()
    kwargs["max_new_tokens"] = max_new_tokens
    kwargs["n"] = 1  # group size is supported in preprocess
    return_logprob = kwargs.pop("logprobs", False)

    output = await self._engine.async_generate(
        input_ids=generation_prompt_ids,
        sampling_params=kwargs,
        return_logprob=return_logprob,
        image_data=image_data,
    )
    return output

async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
    if _req.tool_schemas is not None:
        tool_creation_coroutines = []
        for tool_schema in _req.tool_schemas:
            tool = self._tool_map[tool_schema.function.name]
            create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
            tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
        tool_creation_results = await asyncio.gather(*tool_creation_coroutines)
        _req.add_tool_response_messages(
            self.processing_class, [tool_result for _, tool_result in tool_creation_results]
        )
    if _req.interaction_kwargs and self.interaction_map:
        interaction_kwargs = _req.interaction_kwargs
        # Get interaction by name from interaction_kwargs
        interaction_name = interaction_kwargs.get("name", "gsm8k")  # Default to gsm8k for backward compatibility
        if interaction_name not in self.interaction_map:
            raise ValueError(
                f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                f"{list(self.interaction_map.keys())}"
            )

        interaction = self.interaction_map[interaction_name]
        await interaction.start_interaction(_req.request_id, **interaction_kwargs)

@GPUMemoryLogger(role="sglang rollout", logger=logger)
@torch.no_grad()
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    """Generates multi-turn sequences for a batch of prompts.
    For multi-turn generation, each prompt is processed separately via
    `_req_level_generate_sequences` for better tool calling control.
    Note that in multi-turn generation, we repeat the prompts for rollout.n times in ray_trainer.
    Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
    """
    # Async rollout with tools support
    do_sample = prompts.meta_info.get("do_sample", True)
    is_validate = prompts.meta_info.get("validate", False)
    tgt_device = prompts.batch["input_ids"].device

    if self._tp_rank == 0:
        print("_req_level_generate_sequences called!!!!!!!! and tp_rank is 0")
        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
        )

        # distinguish training and validation
        if is_validate:
            # Validation mode: process all requests without abort
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                )
            )
        else:
            # add progress monitoring and abort function
            total_requests = len(req_list)
            target_completion = int(total_requests * (1 - self.config.get("over_sample_rate", 0.0)))
            # abort when target_completion of requests are completed

            completed_count = 0
            aborted_requests = []
            all_tasks = []

            async def rollout_a_request_with_cancellation_handler(req):
                try:
                    result = await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)
                    return result
                except asyncio.CancelledError:
                    # request is cancelled, return padding
                    logger.info(f"Request {req.request_id} was cancelled, creating padding")
                    aborted_requests.append(req.request_id)
                    return self._create_padding_request(req)

            async def run_with_cancellation():
                nonlocal all_tasks
                nonlocal completed_count
                all_tasks = [
                    asyncio.create_task(rollout_a_request_with_cancellation_handler(req)) for req in req_list
                ]

                # Wait for target_completion tasks to complete
                try:
                    for completed_task in asyncio.as_completed(all_tasks):
                        await completed_task
                        completed_count += 1
                        if completed_count >= target_completion:
                            break
                finally:
                    # Cancel remaining tasks
                    for t in all_tasks:
                        if not t.done():
                            t.cancel()

                    # Wait for all tasks to finish (including cancelled ones)
                    final_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                    # Abort all requests in SGLang engine
                    await self._engine.abort_request(abort_all=True)
                return final_results

            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(run_with_cancellation())

        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
    else:
        print("IS_NONE")
        sorted_output_req_list = None

    dist.barrier()
    [sorted_output_req_list] = broadcast_pyobj(
        data=[sorted_output_req_list],
        rank=self._rank,
        dist_group=self._device_mesh_cpu["tp"].get_group(),
        src=self._device_mesh_cpu["tp"].mesh[0].item(),
        force_cpu_device=False,
    )
    # Construct the batch data
    prompt_ids, response_ids = [], []
    prompt_attention_mask, response_attention_mask = [], []
    prompt_position_ids, response_position_ids = [], []
    response_loss_mask = []
    messages = []
    reward_scores = []
    multi_modal_inputs = []
    request_ids = []
    if self.config.calculate_log_probs:
        output_logprobs = []
        rollout_output_token_ids = []

    for req in sorted_output_req_list:
        assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
        assert (
            req.input_ids.shape[-1]
            == req.attention_mask.shape[-1]
            == req.position_ids.shape[-1]
            == req.loss_mask.shape[-1]
        ), f"""Request {req.request_id} has different length of 
            {req.input_ids.shape[-1]=}, {req.attention_mask.shape[-1]=}, 
            {req.position_ids.shape[-1]=}, {req.loss_mask.shape[-1]=}"""
        error_message_lines = [
            f"""Request {req.request_id} has input_ids length {req.input_ids.shape[-1]}
                greater than max_model_len {self.config.max_model_len}""",
            f"Decoded input_ids: {self.processing_class.decode(req.input_ids.squeeze(0))}",
            f"Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids.squeeze(0))}",
            f"Decoded response_ids: {self.processing_class.decode(req.response_ids.squeeze(0))}",
            f"Messages: {req.messages}",
            f"Max model length: {req.max_model_len}",
        ]
        error_message = "\n".join(error_message_lines)
        assert req.input_ids.shape[-1] <= self.config.max_model_len, error_message

        prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
        response_ids.append(req.response_ids.to(tgt_device).squeeze(0))
        if req.response_ids.shape[-1] > self.config.response_length:
            logger.warning(
                f"""{req.request_id=} has response_ids length {req.response_ids.shape[-1]} 
                greater than max_response_len {self.config.response_length},\n{req=}"""
            )
        prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
        response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
        prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
        response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
        response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
        messages.append({"messages": req.messages})
        reward_scores.append(req.reward_scores)
        multi_modal_inputs.append(req.multi_modal_inputs)
        request_ids.append(req.request_id)
        if self.config.calculate_log_probs:
            # extract output log_probs
            output_logprobs.append(req.rollout_log_probs[-len(req.response_ids) :])
            rollout_output_token_ids.append(req.output_token_ids[-len(req.response_ids) :])

    prompt_ids = pad_sequence(
        prompt_ids,
        batch_first=True,
        padding_value=self.pad_token_id,
        padding_side="left",
    )
    if prompt_ids.shape[-1] < self.config.prompt_length:
        prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
    response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
    if response_ids.shape[-1] < self.config.response_length:
        response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
    prompt_attention_mask = pad_sequence(
        prompt_attention_mask,
        batch_first=True,
        padding_value=0,
        padding_side="left",
    )
    if prompt_attention_mask.shape[-1] < self.config.prompt_length:
        prompt_attention_mask = pad_sequence_to_length(
            prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
        )
    response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
    if response_attention_mask.shape[-1] < self.config.response_length:
        response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

    # padding prompt_position_ids
    if prompt_position_ids[0].dim() == 2:
        # if prompt_position_ids is a 2D tensor
        # e.g. from qwen2vl, prompt_position_ids.shape = (3, seq_len)
        transposed_prompt_position_ids = [p.transpose(0, 1) for p in prompt_position_ids]
        prompt_position_ids = pad_sequence(
            transposed_prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
        )
        prompt_position_ids = prompt_position_ids.transpose(1, 2)
    else:
        prompt_position_ids = pad_sequence(
            prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
        )
    if prompt_position_ids.shape[-1] < self.config.prompt_length:
        prompt_position_ids = pad_sequence_to_length(
            prompt_position_ids, self.config.prompt_length, 0, left_pad=True
        )

    # padding response_position_ids
    if response_position_ids[0].dim() == 2:
        # if response_position_ids is a 2D tensor
        # e.g. from qwen2vl, response_position_ids.shape = (3, seq_len)
        transposed_response_position_ids = [p.transpose(0, 1) for p in response_position_ids]
        response_position_ids = pad_sequence(
            transposed_response_position_ids, batch_first=True, padding_value=0, padding_side="left"
        )
        response_position_ids = response_position_ids.transpose(1, 2)
    else:
        response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
    if response_position_ids.shape[-1] < self.config.response_length:
        response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)

    response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
    if response_loss_mask.shape[1] < self.config.response_length:
        response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
    if self.config.calculate_log_probs:
        output_logprobs = pad_sequence(output_logprobs, padding_value=0.0, batch_first=True)
        output_logprobs = pad_sequence_to_length(
            output_logprobs, pad_token_id=0.0, max_seq_len=response_ids.shape[-1]
        ).to(tgt_device)
        rollout_output_token_ids = pad_sequence(
            rollout_output_token_ids, padding_value=self.pad_token_id, batch_first=True
        )
        rollout_output_token_ids = pad_sequence_to_length(
            rollout_output_token_ids, pad_token_id=self.pad_token_id, max_seq_len=response_ids.shape[-1]
        ).to(tgt_device)

    input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
    position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

    # Construct the batch data
    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "response_mask": response_loss_mask,
            "input_ids": input_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=len(sorted_output_req_list),
    )
    if self.config.calculate_log_probs:
        batch["rollout_log_probs"] = output_logprobs
        batch["rollout_output_token_ids"] = rollout_output_token_ids

    # free cache engine
    if self._engine is not None and self._tp_rank == 0:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._engine.flush_cache())

    non_tensor_batch = {
        "messages": np.array(messages),
        "reward_scores": np.array(reward_scores),
        "request_id": np.array(request_ids),
    }

    is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
        hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
    )

    if is_multimodal:
        non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
    )

