# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import torch
import torch.nn.functional as F

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_sppo_loss(
    old_log_prob: torch.Tensor,  # (bs, seq_len)
    log_prob: torch.Tensor,  # (bs, seq_len)
    rewards: torch.Tensor,  # (bs,)
    response_mask: torch.Tensor,  # (bs, seq_len)
    eta: float = 1.0,
    loss_agg_mode: str = "token-mean",
):
    """SPPO Loss computation."""
    # Compute log-ratios over masked tokens
    log_prob_sum = (log_prob * response_mask).sum(dim=1)  # (bs,)
    old_log_prob_sum = (old_log_prob * response_mask).sum(dim=1)  # (bs,)
    log_ratios = log_prob_sum - old_log_prob_sum  # (bs,)

    scaled_rewards = eta * (rewards)
    loss_vec = (log_ratios - scaled_rewards) ** 2  # (bs,)

    # token-mean: average over valid samples (per-sample loss is already scalar)
    sample_mask = response_mask.any(dim=1).float()  # (bs,)
    loss = verl_F.masked_mean(loss_vec, sample_mask)

    return loss, log_ratios, scaled_rewards


def compute_dpo_loss_from_micro(
    log_prob: torch.Tensor,              # (bs, seq_len), requires grad
    response_mask: torch.Tensor,         # (bs, seq_len), 0/1 or bool
    dpo_pair_id: torch.Tensor,           # (bs,), long; -1 for non-dpo rows
    dpo_is_chosen: torch.Tensor,         # (bs,), bool
    beta: float = 0.1,
    ref_log_prob: torch.Tensor | None = None,  # (bs, seq_len) optional
):
    """Compute DPO logistic loss within ONE micro-batch.

    We assume each valid pair_id has exactly one chosen and one rejected *within the same micro-batch*.
    If pairs are split across micro-batches, those pairs won't contribute to loss (num_pairs will be 0).

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

    # Align chosen and rejected by pair_id
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


class DataParallelSPPOActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        print("w@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Updating SPPO Actor policy...")
        print("w@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        # loss routing: "grpo" (default) | "dpo" | "mixed"
        loss_type = data.meta_info.get("loss_type", "grpo")
        dpo_beta = float(data.meta_info.get("dpo_beta", 0.1))
        dpo_lambda = float(data.meta_info.get("dpo_lambda", 1.0))

        # ---- guards for DPO: pairs must stay inside the same micro-batch ----
        if loss_type in ("dpo", "mixed"):
            if self.config.use_dynamic_bsz:
                raise ValueError("DPO/mixed currently requires use_dynamic_bsz=False (pair alignment may break).")

            if (self.config.ppo_micro_batch_size_per_gpu % 2) != 0:
                raise ValueError(
                    "For DPO/mixed, ppo_micro_batch_size_per_gpu must be even to keep pairs together."
                )

        # ---- select keys depending on loss_type ----
        available_keys = set(data.batch.keys())
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if multi_turn:
            select_keys.append("loss_mask")

        if loss_type in ("grpo", "mixed"):
            print("LOSS TYPE:", loss_type)
            select_keys += ["old_log_probs", "seq_level_rewards"]

        if loss_type in ("dpo", "mixed"):
            print("LOSS TYPE:", loss_type)
            if "dpo_pair_id" not in available_keys or "dpo_is_chosen" not in available_keys:
                raise ValueError("loss_type is dpo/mixed but dpo_pair_id/dpo_is_chosen not found in batch.")
            select_keys += ["dpo_pair_id", "dpo_is_chosen"]

        if "ref_log_prob" in available_keys and (self.config.use_kl_loss or loss_type in ("dpo", "mixed")):
            select_keys.append("ref_log_prob")

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        if has_multi_modal_inputs:
            if loss_type in ("dpo", "mixed"):
                raise ValueError("DPO/mixed with multi-modal inputs is not supported yet.")
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data_mb in enumerate(dataloader):
                mini_batch = data_mb

                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data_mb.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())

                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    eta = self.config.get("sppo_eta", 1.0)

                    # forward
                    calculate_entropy = (entropy_coeff != 0)
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # ---------------- GRPO/SPPO loss ----------------
                    grpo_loss = None
                    log_ratios = None
                    preference = None

                    if loss_type in ("grpo", "mixed"):
                        if loss_type == "mixed":
                            grpo_rows = (data["dpo_pair_id"] < 0)
                        else:
                            grpo_rows = None

                        if grpo_rows is None or torch.any(grpo_rows):
                            _old = data["old_log_probs"] if grpo_rows is None else data["old_log_probs"][grpo_rows]
                            _rw = data["seq_level_rewards"] if grpo_rows is None else data["seq_level_rewards"][grpo_rows]
                            _lp = log_prob if grpo_rows is None else log_prob[grpo_rows]
                            _rm = response_mask if grpo_rows is None else response_mask[grpo_rows]
                            _ent = entropy if grpo_rows is None else (entropy[grpo_rows] if entropy is not None else None)

                            pg_loss, log_ratios, preference = compute_sppo_loss(
                                old_log_prob=_old,
                                log_prob=_lp,
                                rewards=_rw,
                                response_mask=_rm,
                                eta=eta,
                                loss_agg_mode=loss_agg_mode,
                            )

                            if entropy_coeff != 0 and _ent is not None:
                                entropy_loss = agg_loss(loss_mat=_ent, loss_mask=_rm, loss_agg_mode=loss_agg_mode)
                                policy_loss_grpo = pg_loss - entropy_loss * entropy_coeff
                            else:
                                policy_loss_grpo = pg_loss

                            # KL only for GRPO rows (avoid double-count with DPO reference term)
                            if self.config.use_kl_loss and ("ref_log_prob" in data):
                                _ref = data["ref_log_prob"] if grpo_rows is None else data["ref_log_prob"][grpo_rows]
                                kld = kl_penalty(
                                    logprob=_lp, ref_logprob=_ref, kl_penalty=self.config.kl_loss_type
                                )
                                kl_loss = agg_loss(loss_mat=kld, loss_mask=_rm, loss_agg_mode=self.config.loss_agg_mode)
                                policy_loss_grpo = policy_loss_grpo + kl_loss * self.config.kl_loss_coef
                                metrics["actor/kl_loss"] = float(kl_loss.detach().item())
                                metrics["actor/kl_coef"] = float(self.config.kl_loss_coef)

                            grpo_loss = policy_loss_grpo

                    # ---------------- DPO loss ----------------
                    dpo_loss = None
                    dpo_delta = None
                    dpo_pairs = 0
                    if loss_type in ("dpo", "mixed"):
                        ref_lp = data.get("ref_log_prob", None)
                        dpo_loss, dpo_delta, dpo_pairs = compute_dpo_loss_from_micro(
                            log_prob=log_prob,
                            response_mask=response_mask,
                            dpo_pair_id=data["dpo_pair_id"],
                            dpo_is_chosen=data["dpo_is_chosen"],
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

                    # ---------------- total loss ----------------
                    if loss_type == "grpo":
                        total_policy_loss = grpo_loss
                    elif loss_type == "dpo":
                        total_policy_loss = dpo_loss
                    else:
                        if grpo_loss is None and dpo_loss is None:
                            continue
                        total_policy_loss = 0.0
                        if grpo_loss is not None:
                            total_policy_loss = total_policy_loss + grpo_loss
                        if dpo_loss is not None:
                            total_policy_loss = total_policy_loss + dpo_loss

                    # scale for grad accumulation
                    loss = total_policy_loss / self.gradient_accumulation
                    loss.backward()

                    m = {"actor/loss": float(loss.detach().item())}
                    if log_ratios is not None:
                        m["actor/log_ratio_mean"] = float(log_ratios.mean().detach().item())
                    if preference is not None:
                        m["actor/preference_mean"] = float(preference.mean().detach().item())
                    if dpo_loss is not None:
                        m["dpo/loss"] = float(dpo_loss.detach().item())
                        m["dpo/pairs"] = float(dpo_pairs)
                        if dpo_delta is not None and dpo_delta.numel() > 0:
                            m["dpo/delta_mean"] = float(dpo_delta.mean().item())
                            m["dpo/delta_min"] = float(dpo_delta.min().item())
                            m["dpo/delta_max"] = float(dpo_delta.max().item())
                    append_to_dict(metrics, m)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": float(grad_norm.detach().item())})

        self.actor_optimizer.zero_grad()
        return metrics
