# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_dpo_loss_from_micro(
    log_prob: torch.Tensor,              # (bs, resp_len), requires grad
    response_mask: torch.Tensor,         # (bs, resp_len), 0/1 or bool
    dpo_pair_id: torch.Tensor,           # (bs,), long; -1 for non-dpo rows
    dpo_is_chosen: torch.Tensor,         # (bs,), bool
    beta: float = 0.1,
    ref_log_prob=None,                  # (bs, resp_len) optional
):
    """
    Compute DPO logistic loss within ONE micro-batch.

    Notes:
      - Only rows with dpo_pair_id >= 0 are used.
      - A pair contributes only if BOTH chosen and rejected for the same pair_id exist in this micro-batch.
        (So trainer should interleave [c0,r0,c1,r1,...] and keep micro-batch size even.)
      - If ref_log_prob is None, we do reference-free DPO (ref=0).

    Returns:
      loss: scalar tensor or None if no complete pairs
      delta: detached tensor [K] or None
      num_pairs: int K
    """
    if dpo_pair_id is None or dpo_is_chosen is None:
        return None, None, 0

    mask = dpo_pair_id >= 0
    if not torch.any(mask):
        return None, None, 0

    pid = dpo_pair_id[mask].to(torch.long)
    is_c = dpo_is_chosen[mask].bool()
    lp_tok = log_prob[mask]
    rm = response_mask[mask].bool()

    seq_lp = (lp_tok * rm.to(lp_tok.dtype)).sum(dim=-1)  # [Ndpo]

    if ref_log_prob is not None:
        ref_tok = ref_log_prob[mask]
        seq_ref = (ref_tok * rm.to(ref_tok.dtype)).sum(dim=-1)
    else:
        seq_ref = torch.zeros_like(seq_lp)

    chosen_mask = is_c
    reject_mask = ~is_c
    if (not torch.any(chosen_mask)) or (not torch.any(reject_mask)):
        return None, None, 0

    pid_c = pid[chosen_mask]
    pid_r = pid[reject_mask]
    lp_c = seq_lp[chosen_mask]
    lp_r = seq_lp[reject_mask]
    ref_c = seq_ref[chosen_mask]
    ref_r = seq_ref[reject_mask]

    pid_c_sorted, c_ord = torch.sort(pid_c)
    pid_r_sorted, r_ord = torch.sort(pid_r)

    lp_c = lp_c[c_ord]
    ref_c = ref_c[c_ord]
    lp_r = lp_r[r_ord]
    ref_r = ref_r[r_ord]

    pos = torch.searchsorted(pid_r_sorted, pid_c_sorted)
    ok = (pos < pid_r_sorted.numel()) & (pid_r_sorted[pos] == pid_c_sorted)
    if not torch.any(ok):
        return None, None, 0

    lp_c = lp_c[ok]
    ref_c = ref_c[ok]
    lp_r = lp_r[pos[ok]]
    ref_r = ref_r[pos[ok]]

    delta = (lp_c - ref_c) - (lp_r - ref_r)  # [K]
    loss = -F.logsigmoid(beta * delta).mean()
    return loss, delta.detach(), int(delta.numel())



class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # ---- loss routing: "grpo"(default) | "dpo" | "mixed" ----
        loss_type = data.meta_info.get("loss_type", "grpo")
        dpo_beta = float(data.meta_info.get("dpo_beta", 0.1))
        dpo_lambda = float(data.meta_info.get("dpo_lambda", 1.0))

        # DPO needs pair alignment inside micro-batches; dynamic bsz may reorder/split pairs.
        if loss_type in ("dpo", "mixed"):
            if self.config.use_dynamic_bsz:
                raise ValueError("DPO/mixed requires use_dynamic_bsz=False (pair alignment may break).")
            if (self.config.ppo_micro_batch_size_per_gpu % 2) != 0:
                raise ValueError(
                    "For DPO/mixed, ppo_micro_batch_size_per_gpu must be even to keep pairs in the same micro-batch."
                )

        # ---- select keys (minimal change to original) ----
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        if loss_type in ("grpo", "mixed"):
            select_keys += ["old_log_probs", "advantages"]
        if loss_type in ("dpo", "mixed"):
            select_keys += ["dpo_pair_id", "dpo_is_chosen"]

        if self.config.use_kl_loss and ("ref_log_prob" in data.batch.keys()):
            # KL is only applied to GRPO rows (see below). For DPO, ref_log_prob is optional.
            select_keys.append("ref_log_prob")
        elif (loss_type in ("dpo", "mixed")) and ("ref_log_prob" in data.batch.keys()):
            # Use as reference term in DPO if provided (reference-free otherwise).
            select_keys.append("ref_log_prob")

        # Include pre-computed IS weights if present in batch
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if has_multi_modal_inputs and loss_type in ("dpo", "mixed"):
            raise ValueError("DPO/mixed with multi_modal_inputs is not supported in this implementation.")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    response_mask = model_inputs["response_mask"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = (entropy_coeff != 0)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob_full = model_inputs.get("old_log_probs", None)
                    else:
                        if on_policy:
                            old_log_prob_full = log_prob.detach()
                        else:
                            old_log_prob_full = model_inputs.get("old_log_probs", None)

                    # Extract pre-computed rollout importance sampling weights if present
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # ===== GRPO/PPO loss (only for GRPO rows; in mixed, dpo_pair_id < 0) =====
                    grpo_loss = None
                    pg_loss = None
                    pg_clipfrac = None
                    ppo_kl = None
                    pg_clipfrac_lower = None

                    if loss_type in ("grpo", "mixed"):
                        if loss_type == "mixed":
                            grpo_rows = (model_inputs["dpo_pair_id"] < 0)
                        else:
                            grpo_rows = None  # all rows

                        if grpo_rows is None or torch.any(grpo_rows):
                            _log_prob = log_prob if grpo_rows is None else log_prob[grpo_rows]
                            _resp_mask = response_mask if grpo_rows is None else response_mask[grpo_rows]
                            _advantages = model_inputs["advantages"] if grpo_rows is None else model_inputs["advantages"][grpo_rows]
                            _old_log_prob = old_log_prob_full if grpo_rows is None else old_log_prob_full[grpo_rows]
                            _isw = rollout_is_weights if (rollout_is_weights is None or grpo_rows is None) else rollout_is_weights[grpo_rows]

                            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                            policy_loss_fn = get_policy_loss_fn(loss_mode)

                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob=_old_log_prob,
                                log_prob=_log_prob,
                                advantages=_advantages,
                                response_mask=_resp_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=_isw,
                            )

                            if entropy_coeff != 0 and entropy is not None:
                                _ent = entropy if grpo_rows is None else entropy[grpo_rows]
                                entropy_loss = agg_loss(loss_mat=_ent, loss_mask=_resp_mask, loss_agg_mode=loss_agg_mode)
                                policy_loss = pg_loss - entropy_loss * entropy_coeff
                            else:
                                policy_loss = pg_loss

                            # KL only for GRPO rows (avoid double-counting with DPO reference term)
                            if self.config.use_kl_loss and ("ref_log_prob" in model_inputs):
                                _ref = model_inputs["ref_log_prob"] if grpo_rows is None else model_inputs["ref_log_prob"][grpo_rows]
                                kld = kl_penalty(
                                    logprob=_log_prob, ref_logprob=_ref, kl_penalty=self.config.kl_loss_type
                                )
                                kl_loss = agg_loss(loss_mat=kld, loss_mask=_resp_mask, loss_agg_mode=loss_agg_mode)

                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                micro_batch_metrics["actor/kl_loss"] = float(kl_loss.detach().float().mean().item()) * loss_scale_factor
                                micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                            grpo_loss = policy_loss

                    # ===== DPO loss (only for DPO rows; pairs must be inside micro-batch) =====
                    dpo_loss = None
                    dpo_delta = None
                    dpo_pairs = 0
                    if loss_type in ("dpo", "mixed"):
                        ref_lp = model_inputs.get("ref_log_prob", None)
                        dpo_loss, dpo_delta, dpo_pairs = compute_dpo_loss_from_micro(
                            log_prob=log_prob,
                            response_mask=response_mask,
                            dpo_pair_id=model_inputs["dpo_pair_id"],
                            dpo_is_chosen=model_inputs["dpo_is_chosen"],
                            beta=dpo_beta,
                            ref_log_prob=ref_lp,
                        )
                        if dpo_loss is not None:
                            dpo_loss = dpo_loss * dpo_lambda
                        else:
                            if loss_type == "dpo":
                                raise ValueError(
                                    "DPO loss has no complete pairs in this micro-batch. "
                                    "Ensure trainer interleaves [c0,r0,c1,r1,...] and micro-batch size is even."
                                )

                    # ===== Combine =====
                    if loss_type == "grpo":
                        total_policy_loss = grpo_loss
                    elif loss_type == "dpo":
                        total_policy_loss = dpo_loss
                    else:  # mixed
                        if grpo_loss is None and dpo_loss is None:
                            continue
                        total_policy_loss = 0.0
                        if grpo_loss is not None:
                            total_policy_loss = total_policy_loss + grpo_loss
                        if dpo_loss is not None:
                            total_policy_loss = total_policy_loss + dpo_loss

                    loss = total_policy_loss * loss_scale_factor
                    loss.backward()

                    # ===== Metrics (keep original keys) =====
                    if pg_loss is not None:
                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                                "actor/pg_clipfrac": float(pg_clipfrac.detach().float().mean().item()) if pg_clipfrac is not None else 0.0,
                                "actor/ppo_kl": float(ppo_kl.detach().float().mean().item()) if ppo_kl is not None else 0.0,
                                "actor/pg_clipfrac_lower": float(pg_clipfrac_lower.detach().float().mean().item()) if pg_clipfrac_lower is not None else 0.0,
                            }
                        )
                    if dpo_loss is not None:
                        micro_batch_metrics["dpo/loss"] = dpo_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["dpo/pairs"] = float(dpo_pairs)
                        if dpo_delta is not None and dpo_delta.numel() > 0:
                            micro_batch_metrics["dpo/delta_mean"] = float(dpo_delta.mean().item())
                            micro_batch_metrics["dpo/delta_min"] = float(dpo_delta.min().item())
                            micro_batch_metrics["dpo/delta_max"] = float(dpo_delta.max().item())

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
