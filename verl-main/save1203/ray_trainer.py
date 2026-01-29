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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Construct contiguous position ids given an attention mask with padding."""

    if attention_mask.dim() != 2:
        raise ValueError(
            f"build_position_ids expects a 2D attention mask, got shape {tuple(attention_mask.shape)}"
        )

    mask = attention_mask
    if mask.dtype != torch.long:
        mask = mask.to(dtype=torch.long)
    position_ids = compute_position_id_with_mask(mask)
    # Ensure pad tokens keep zero so regenerated prompts stay aligned with model expectations.
    return position_ids * mask.ne(0)


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.need_feedback = self.config.actor_rollout_ref.rollout.get("feedback")
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.latest_all_negative_batch: Optional[DataProto] = None
        self.latest_all_negative_prompts: Optional[DataProto] = None

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )


    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute IS weights and apply rejection sampling for rollout-training mismatch.

        Computes importance sampling weights to correct for distribution mismatch between
        rollout and training policies. Applies rejection sampling (mask mode/veto) by
        modifying response_mask. Always updates response_mask; conditionally adds IS weights.

        Key behavior:
        - response_mask: ALWAYS updated with rejection (mask mode + veto excluded from training)
        - rollout_is_weights: Added to batch ONLY if config.algorithm.rollout_is=True

        This separation ensures:
        - Rejection works even when IS weights are disabled (rollout_is=False)
        - Metrics can be monitored before enabling IS weight application

        Args:
            batch: DataProto with old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if rollout_is=True)
                metrics: Dict of IS and mismatch metrics, all with "mismatch/" prefix
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch (None = disabled, float = enabled)
        rollout_is_threshold = self.config.algorithm.get("rollout_is_threshold", None)
        if rollout_is_threshold is not None and rollout_is_threshold > 0 and "rollout_log_probs" in batch.batch:
            # Compute IS weights and get modified response_mask
            rollout_is_weights, modified_response_mask, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.get("rollout_is_threshold_lower", None),
                rollout_is_veto_threshold=self.config.algorithm.get("rollout_is_veto_threshold", None),
            )

            # ALWAYS update response_mask with rejection (even if rollout_is=False)
            # - Mask mode: tokens with outlier IS ratios excluded
            # - Veto: sequences with catastrophic tokens excluded
            # This ensures correct loss normalization (rejected samples not in denominator)
            batch.batch["response_mask"] = modified_response_mask

            # Conditionally add IS weights based on rollout_is config flag
            # - rollout_is=True: Enable IS weight correction in policy loss
            # - rollout_is=False: Metrics-only mode (rejection still applied via mask)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights (safety-bounded, mode-processed) to enable weight correction
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def _collect_all_negative_rollout_prompts(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        repeat_times: int,
    ) -> tuple[Optional[DataProto], Optional[DataProto], dict]:
        """Collect prompts whose rollout generations are all non-positive.

        Returns tuple of (failed_rollout_batch, failed_prompt_batch, metrics).
        failed_rollout_batch contains all rollout copies, failed_prompt_batch keeps
        one representative sample per prompt for easier inspection.
        """

        def _strip_rollout_outputs(prompt_batch: DataProto) -> None:
            """Remove rollout-specific tensors so only the original inputs remain."""

            tensor_keys_to_remove = [
                "responses",
                "response_mask",
                "rollout_log_probs",
                "rollout_is_weights",
                "old_log_probs",
                "ref_log_probs",
                "token_level_scores",
                "token_level_rewards",
                "values",
            ]

            for key in tensor_keys_to_remove:
                if key in prompt_batch.batch:
                    prompt_batch.batch.pop(key)

        def _decode_token_sequences(
            token_tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor] = None
        ) -> list[str]:
            """Convert token tensors (optionally masked) back to text for inspection."""

            if token_tensor is None:
                return []

            token_tensor = token_tensor.detach().cpu()
            mask_tensor = mask_tensor.detach().cpu() if mask_tensor is not None else None
            decoded_sequences: list[str] = []
            for row_idx in range(token_tensor.size(0)):
                token_row = token_tensor[row_idx]
                if mask_tensor is not None:
                    mask_row = mask_tensor[row_idx].bool()
                    if mask_row.any():
                        token_row = token_row[mask_row]
                    else:
                        token_row = token_row[:0]
                decoded_sequences.append(self.tokenizer.decode(token_row.tolist(), skip_special_tokens=True))
            return decoded_sequences

        

        if reward_tensor is None or repeat_times <= 1:
            self.latest_all_negative_batch = None
            self.latest_all_negative_prompts = None
            return None, None, {}

        total_rewards = reward_tensor.sum(dim=-1)
        batch_size = total_rewards.shape[0]
        if batch_size % repeat_times != 0:
            self.latest_all_negative_batch = None
            self.latest_all_negative_prompts = None
            return None, None, {}

        rewards_per_prompt = total_rewards.view(batch_size // repeat_times, repeat_times)
        all_negative_mask = torch.all(rewards_per_prompt <= 0, dim=1)
        if not torch.any(all_negative_mask):
            self.latest_all_negative_batch = None
            self.latest_all_negative_prompts = None
            return None, None, {}

        prompt_indices = torch.arange(batch_size, device=reward_tensor.device).view(-1, repeat_times)
        selected_indices = prompt_indices[all_negative_mask].reshape(-1).cpu()
        failed_batch = batch.select_idxs(selected_indices)

        prompt_anchor_indices = prompt_indices[all_negative_mask][:, 0].cpu()
        failed_prompts = batch.select_idxs(prompt_anchor_indices)

        # Decode prompt/response texts to make downstream tooling easier to implement.
        prompt_texts_per_prompt = []
        if "prompts" in failed_prompts.batch:
            prompt_texts_per_prompt = _decode_token_sequences(failed_prompts.batch["prompts"])
        elif "input_ids" in failed_prompts.batch:
            prompt_texts_per_prompt = _decode_token_sequences(
                failed_prompts.batch["input_ids"], failed_prompts.batch.get("attention_mask")
            )

        prompt_texts_repeated: list[str] = []
        failed_rollout_size = 0
        for tensor in failed_batch.batch.values():
            if isinstance(tensor, torch.Tensor):
                failed_rollout_size = tensor.shape[0]
                break
        if prompt_texts_per_prompt:
            for text in prompt_texts_per_prompt:
                prompt_texts_repeated.extend([text] * repeat_times)
            # Match actual failed rollout count in case of padding/partial batches.
            if failed_rollout_size > 0:
                prompt_texts_repeated = prompt_texts_repeated[:failed_rollout_size]

        rollout_texts = []
        if "responses" in failed_batch.batch:
            rollout_texts = _decode_token_sequences(
                failed_batch.batch["responses"], failed_batch.batch.get("response_mask")
            )

        if prompt_texts_repeated:
            failed_batch.non_tensor_batch["prompt_text"] = np.array(prompt_texts_repeated, dtype=object)
        if rollout_texts:
            failed_batch.non_tensor_batch["rollout_text"] = np.array(rollout_texts, dtype=object)
        if prompt_texts_repeated and rollout_texts:
            paired = [
                {"prompt": p, "rollout": r}
                for p, r in zip(prompt_texts_repeated, rollout_texts)
            ]
            failed_batch.non_tensor_batch["prompt_rollout_pairs"] = np.array(paired, dtype=object)

        if prompt_texts_per_prompt:
            failed_prompts.non_tensor_batch["prompt_text"] = np.array(prompt_texts_per_prompt, dtype=object)

        _strip_rollout_outputs(failed_prompts)

        failed_batch.meta_info = dict(failed_batch.meta_info)
        failed_prompt_count = int(all_negative_mask.sum().item())
        failed_batch.meta_info["failed_rollout_indices"] = selected_indices.tolist()
        failed_batch.meta_info["failed_prompt_count"] = failed_prompt_count
        failed_batch.meta_info["repeat_times"] = repeat_times

        failed_prompts.meta_info = dict(failed_prompts.meta_info)
        failed_prompts.meta_info["failed_prompt_count"] = failed_prompt_count
        failed_prompts.meta_info["repeat_times"] = repeat_times

        uid_array = batch.non_tensor_batch.get("uid")
        if uid_array is not None and uid_array.shape[0] == batch_size:
            try:
                uid_matrix = uid_array.reshape(-1, repeat_times)
                failed_uids = uid_matrix[all_negative_mask.cpu().numpy(), 0].tolist()
                failed_batch.meta_info["failed_prompt_uids"] = failed_uids
                failed_prompts.meta_info["failed_prompt_uids"] = failed_uids
            except Exception:
                pass

        self.latest_all_negative_batch = failed_batch
        self.latest_all_negative_prompts = failed_prompts
        metrics = {
            "rollout/all_negative_prompt_cnt": failed_prompt_count,
            "rollout/all_negative_sample_cnt": failed_prompt_count * repeat_times,
        }
        return failed_batch, failed_prompts, metrics

    # def _collect_all_negative_rollout_prompts_by_group(
    #     self,
    #     batch: DataProto,
    #     reward_tensor: torch.Tensor,
    #     repeat_times: int,
    # ) -> tuple[Optional[DataProto], Optional[DataProto], dict]:
    #     """Collect prompts whose rollout generations are all non-positive.

    #     Returns tuple of (failed_rollout_batch, failed_prompt_batch, metrics).
    #     failed_rollout_batch contains all rollout copies, failed_prompt_batch keeps
    #     one representative sample per prompt for easier inspection.
    #     """

    #     def _strip_rollout_outputs(prompt_batch: DataProto) -> None:
    #         """Remove rollout-specific tensors so only the original inputs remain."""

    #         tensor_keys_to_remove = [
    #             "responses",
    #             "response_mask",
    #             "rollout_log_probs",
    #             "rollout_is_weights",
    #             "old_log_probs",
    #             "ref_log_probs",
    #             "token_level_scores",
    #             "token_level_rewards",
    #             "values",
    #         ]

    #         for key in tensor_keys_to_remove:
    #             if key in prompt_batch.batch:
    #                 prompt_batch.batch.pop(key)

    #     def _decode_token_sequences(
    #         token_tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor] = None
    #     ) -> list[str]:
    #         """Convert token tensors (optionally masked) back to text for inspection."""

    #         if token_tensor is None:
    #             return []

    #         token_tensor = token_tensor.detach().cpu()
    #         mask_tensor = mask_tensor.detach().cpu() if mask_tensor is not None else None
    #         decoded_sequences: list[str] = []
    #         for row_idx in range(token_tensor.size(0)):
    #             token_row = token_tensor[row_idx]
    #             if mask_tensor is not None:
    #                 mask_row = mask_tensor[row_idx].bool()
    #                 if mask_row.any():
    #                     token_row = token_row[mask_row]
    #                 else:
    #                     token_row = token_row[:0]
    #             decoded_sequences.append(self.tokenizer.decode(token_row.tolist(), skip_special_tokens=True))
    #         return decoded_sequences

    #     # 如果没有重复 rollout 或 reward 为空，直接返回
    #     if reward_tensor is None or repeat_times <= 1:
    #         self.latest_all_negative_batch = None
    #         self.latest_all_negative_prompts = None
    #         return None, None, {}

    #     total_rewards = reward_tensor.sum(dim=-1)
    #     batch_size = total_rewards.shape[0]
    #     if batch_size % repeat_times != 0:
    #         self.latest_all_negative_batch = None
    #         self.latest_all_negative_prompts = None
    #         return None, None, {}

    #     # [num_prompts, repeat_times]
    #     rewards_per_prompt = total_rewards.view(batch_size // repeat_times, repeat_times)
    #     all_negative_mask = torch.all(rewards_per_prompt <= 0, dim=1)
    #     if not torch.any(all_negative_mask):
    #         self.latest_all_negative_batch = None
    #         self.latest_all_negative_prompts = None
    #         return None, None, {}

    #     # prompt_indices[i] = 这一条 prompt 在 batch 里的所有 rollout 的行号
    #     prompt_indices = torch.arange(batch_size, device=reward_tensor.device).view(-1, repeat_times)
    #     selected_indices = prompt_indices[all_negative_mask].reshape(-1).cpu()
    #     failed_batch = batch.select_idxs(selected_indices)

    #     # anchor prompt（每个 failed prompt 取第一条 rollout 作为代表）
    #     prompt_anchor_indices = prompt_indices[all_negative_mask][:, 0].cpu()
    #     failed_prompts = batch.select_idxs(prompt_anchor_indices)

    #     # Decode prompt/response texts to make downstream tooling easier to implement.
    #     prompt_texts_per_prompt = []
    #     if "prompts" in failed_prompts.batch:
    #         prompt_texts_per_prompt = _decode_token_sequences(failed_prompts.batch["prompts"])
    #     elif "input_ids" in failed_prompts.batch:
    #         prompt_texts_per_prompt = _decode_token_sequences(
    #             failed_prompts.batch["input_ids"], failed_prompts.batch.get("attention_mask")
    #         )

    #     prompt_texts_repeated: list[str] = []
    #     failed_rollout_size = 0
    #     for tensor in failed_batch.batch.values():
    #         if isinstance(tensor, torch.Tensor):
    #             failed_rollout_size = tensor.shape[0]
    #             break
    #     if prompt_texts_per_prompt:
    #         for text in prompt_texts_per_prompt:
    #             prompt_texts_repeated.extend([text] * repeat_times)
    #         # Match actual failed rollout count in case of padding/partial batches.
    #         if failed_rollout_size > 0:
    #             prompt_texts_repeated = prompt_texts_repeated[:failed_rollout_size]

    #     rollout_texts = []
    #     if "responses" in failed_batch.batch:
    #         rollout_texts = _decode_token_sequences(
    #             failed_batch.batch["responses"], failed_batch.batch.get("response_mask")
    #         )

    #     if prompt_texts_repeated:
    #         failed_batch.non_tensor_batch["prompt_text"] = np.array(prompt_texts_repeated, dtype=object)
    #     if rollout_texts:
    #         failed_batch.non_tensor_batch["rollout_text"] = np.array(rollout_texts, dtype=object)
    #     if prompt_texts_repeated and rollout_texts:
    #         paired = [
    #             {"prompt": p, "rollout": r}
    #             for p, r in zip(prompt_texts_repeated, rollout_texts)
    #         ]
    #         failed_batch.non_tensor_batch["prompt_rollout_pairs"] = np.array(paired, dtype=object)

    #     if prompt_texts_per_prompt:
    #         failed_prompts.non_tensor_batch["prompt_text"] = np.array(prompt_texts_per_prompt, dtype=object)

    #     _strip_rollout_outputs(failed_prompts)

    #     failed_batch.meta_info = dict(failed_batch.meta_info)
    #     failed_prompt_count = int(all_negative_mask.sum().item())
    #     failed_batch.meta_info["failed_rollout_indices"] = selected_indices.tolist()
    #     failed_batch.meta_info["failed_prompt_count"] = failed_prompt_count
    #     failed_batch.meta_info["repeat_times"] = repeat_times

    #     failed_prompts.meta_info = dict(failed_prompts.meta_info)
    #     failed_prompts.meta_info["failed_prompt_count"] = failed_prompt_count
    #     failed_prompts.meta_info["repeat_times"] = repeat_times

    #     # uid 信息
    #     failed_uids = None
    #     uid_array = batch.non_tensor_batch.get("uid")
    #     if uid_array is not None and uid_array.shape[0] == batch_size:
    #         try:
    #             uid_matrix = uid_array.reshape(-1, repeat_times)
    #             failed_uids = uid_matrix[all_negative_mask.cpu().numpy(), 0].tolist()
    #             failed_batch.meta_info["failed_prompt_uids"] = failed_uids
    #             failed_prompts.meta_info["failed_prompt_uids"] = failed_uids
    #         except Exception:
    #             # uid reshape 出错就当没有 uid
    #             failed_uids = None
    #             pass

    #     # ==== feedback 分组逻辑开始 ====
    #     # k（每个 feedback 用多少条 rollout）：
    #     # 建议你在 __init__ 里设 self.feedback_group_size = config.algorithm.feedback_group_size 之类的
    #     feedback_group_size = int(getattr(self, "feedback_group_size", 2) or 2)
    #     if feedback_group_size > 0 and failed_rollout_size > 0:
    #         # local_index_matrix[p, :] = 这个 failed prompt 在 failed_batch 里的 rollout 行号
    #         total_failed_rollouts = failed_prompt_count * repeat_times
    #         assert total_failed_rollouts == failed_rollout_size, (
    #             f"total_failed_rollouts={total_failed_rollouts}, "
    #             f"failed_rollout_size={failed_rollout_size}"
    #         )

    #         local_index_matrix = torch.arange(total_failed_rollouts).view(failed_prompt_count, repeat_times)

    #         # 每一行 rollout 对应的 group_id（长度 == failed_rollout_size）
    #         group_ids_per_row: list[int] = [-1] * total_failed_rollouts
    #         # 每个 group 对应哪个 failed prompt（下标是 failed_prompts 里的 index）
    #         group_prompt_indices: list[int] = []
    #         # 如果有 uid，顺便记录每个 group 的 prompt uid，方便 merge
    #         group_prompt_uids: list[Any] = []
    #         # 每个 failed prompt 有多少个 group
    #         num_groups_per_prompt: list[int] = []

    #         for p_idx in range(failed_prompt_count):
    #             # 这个 failed prompt 在 failed_batch 里的 rollout 行号（长度 repeat_times）
    #             row_indices = local_index_matrix[p_idx]

    #             # 打乱一下，实现「随机分组」
    #             perm = torch.randperm(row_indices.numel(), device=row_indices.device)
    #             shuffled = row_indices[perm]

    #             # 以 k 为单位切块：[k, k, k, ...]，最后不够 k 也单独成一组
    #             start = 0
    #             this_prompt_group_count = 0
    #             while start < shuffled.numel():
    #                 chunk = shuffled[start : start + feedback_group_size]
    #                 if chunk.numel() == 0:
    #                     break
    #                 group_id = len(group_prompt_indices)

    #                 for idx in chunk.tolist():
    #                     group_ids_per_row[idx] = group_id

    #                 group_prompt_indices.append(p_idx)
    #                 if failed_uids is not None and len(failed_uids) == failed_prompt_count:
    #                     group_prompt_uids.append(failed_uids[p_idx])
    #                 this_prompt_group_count += 1
    #                 start += feedback_group_size

    #             num_groups_per_prompt.append(this_prompt_group_count)

    #         # 写回 meta_info，后面 generate_feedbacks 可以直接用这些信息来按 group 聚合
    #         failed_batch.meta_info["feedback_group_size"] = feedback_group_size
    #         failed_batch.meta_info["feedback_group_ids"] = group_ids_per_row
    #         failed_batch.meta_info["feedback_group_prompt_indices"] = group_prompt_indices
    #         if group_prompt_uids:
    #             failed_batch.meta_info["feedback_group_prompt_uids"] = group_prompt_uids

    #         failed_prompts.meta_info["feedback_num_groups_per_prompt"] = num_groups_per_prompt
    #     # ==== feedback 分组逻辑结束 ====

    #     self.latest_all_negative_batch = failed_batch
    #     self.latest_all_negative_prompts = failed_prompts
    #     metrics = {
    #         "rollout/all_negative_prompt_cnt": failed_prompt_count,
    #         "rollout/all_negative_sample_cnt": failed_prompt_count * repeat_times,
    #     }
    #     return failed_batch, failed_prompts, metrics

    def _collect_all_negative_rollout_prompts_by_group(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        repeat_times: int,
    ) -> tuple[Optional[DataProto], Optional[DataProto], dict]:
        """Collect prompts whose rollout generations are all non-positive, with grouped feedback info.

        Returns tuple of (failed_rollout_batch, failed_prompt_batch, metrics).
        failed_rollout_batch contains all rollout copies, failed_prompt_batch keeps
        one representative sample per prompt for easier inspection.
        """
        import numpy as np

        def _strip_rollout_outputs(prompt_batch: DataProto) -> None:
            """Remove rollout-specific tensors so only the original inputs remain."""

            tensor_keys_to_remove = [
                "responses",
                "response_mask",
                "rollout_log_probs",
                "rollout_is_weights",
                "old_log_probs",
                "ref_log_probs",
                "token_level_scores",
                "token_level_rewards",
                "values",
            ]

            for key in tensor_keys_to_remove:
                if key in prompt_batch.batch:
                    prompt_batch.batch.pop(key)

        def _decode_token_sequences(
            token_tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor] = None
        ) -> list[str]:
            """Convert token tensors (optionally masked) back to text for inspection."""

            if token_tensor is None:
                return []

            token_tensor = token_tensor.detach().cpu()
            mask_tensor = mask_tensor.detach().cpu() if mask_tensor is not None else None
            decoded_sequences: list[str] = []
            for row_idx in range(token_tensor.size(0)):
                token_row = token_tensor[row_idx]
                if mask_tensor is not None:
                    mask_row = mask_tensor[row_idx].bool()
                    if mask_row.any():
                        token_row = token_row[mask_row]
                    else:
                        token_row = token_row[:0]
                decoded_sequences.append(
                    self.tokenizer.decode(token_row.tolist(), skip_special_tokens=True)
                )
            return decoded_sequences

        # 如果没有重复 rollout 或 reward 为空，直接返回
        if reward_tensor is None or repeat_times <= 1:
            self.latest_all_negative_batch = None
            self.latest_all_negative_prompts = None
            return None, None, {}

        total_rewards = reward_tensor.sum(dim=-1)
        batch_size = total_rewards.shape[0]
        if batch_size % repeat_times != 0:
            self.latest_all_negative_batch = None
            self.latest_all_negative_prompts = None
            return None, None, {}

        # [num_prompts, repeat_times]
        rewards_per_prompt = total_rewards.view(batch_size // repeat_times, repeat_times)
        all_negative_mask = torch.all(rewards_per_prompt <= 0, dim=1)
        if not torch.any(all_negative_mask):
            self.latest_all_negative_batch = None
            self.latest_all_negative_prompts = None
            return None, None, {}

        # prompt_indices[i] = 这一条 prompt 在 batch 里的所有 rollout 的行号
        prompt_indices = torch.arange(batch_size, device=reward_tensor.device).view(
            -1, repeat_times
        )
        selected_indices = prompt_indices[all_negative_mask].reshape(-1).cpu()
        failed_batch = batch.select_idxs(selected_indices)

        # anchor prompt（每个 failed prompt 取第一条 rollout 作为代表）
        prompt_anchor_indices = prompt_indices[all_negative_mask][:, 0].cpu()
        failed_prompts = batch.select_idxs(prompt_anchor_indices)

        # Decode prompt/response texts to make downstream tooling easier to implement.
        failed_batch.non_tensor_batch = dict(getattr(failed_batch, "non_tensor_batch", {}) or {})
        failed_prompts.non_tensor_batch = dict(getattr(failed_prompts, "non_tensor_batch", {}) or {})

        prompt_texts_per_prompt: list[str] = []
        if "prompts" in failed_prompts.batch:
            prompt_texts_per_prompt = _decode_token_sequences(failed_prompts.batch["prompts"])
        elif "input_ids" in failed_prompts.batch:
            prompt_texts_per_prompt = _decode_token_sequences(
                failed_prompts.batch["input_ids"],
                failed_prompts.batch.get("attention_mask"),
            )

        prompt_texts_repeated: list[str] = []
        failed_rollout_size = 0
        for tensor in failed_batch.batch.values():
            if isinstance(tensor, torch.Tensor):
                failed_rollout_size = tensor.shape[0]
                break

        failed_prompt_count = int(all_negative_mask.sum().item())

        if prompt_texts_per_prompt:
            for text in prompt_texts_per_prompt:
                prompt_texts_repeated.extend([text] * repeat_times)
            # Match actual failed rollout count in case of padding/partial batches.
            if failed_rollout_size > 0:
                prompt_texts_repeated = prompt_texts_repeated[:failed_rollout_size]

        rollout_texts: list[str] = []
        if "responses" in failed_batch.batch:
            rollout_texts = _decode_token_sequences(
                failed_batch.batch["responses"],
                failed_batch.batch.get("response_mask"),
            )

        if prompt_texts_repeated:
            failed_batch.non_tensor_batch["prompt_text"] = np.array(
                prompt_texts_repeated, dtype=object
            )
        if rollout_texts:
            failed_batch.non_tensor_batch["rollout_text"] = np.array(
                rollout_texts, dtype=object
            )
        if prompt_texts_repeated and rollout_texts:
            paired = [
                {"prompt": p, "rollout": r}
                for p, r in zip(prompt_texts_repeated, rollout_texts)
            ]
            failed_batch.non_tensor_batch["prompt_rollout_pairs"] = np.array(
                paired, dtype=object
            )

        if prompt_texts_per_prompt:
            failed_prompts.non_tensor_batch["prompt_text"] = np.array(
                prompt_texts_per_prompt, dtype=object
            )

        _strip_rollout_outputs(failed_prompts)

        failed_batch.meta_info = dict(failed_batch.meta_info)
        failed_batch.meta_info["failed_rollout_indices"] = selected_indices.tolist()
        failed_batch.meta_info["failed_prompt_count"] = failed_prompt_count
        failed_batch.meta_info["repeat_times"] = repeat_times

        failed_prompts.meta_info = dict(failed_prompts.meta_info)
        failed_prompts.meta_info["failed_prompt_count"] = failed_prompt_count
        failed_prompts.meta_info["repeat_times"] = repeat_times

        # uid 信息
        failed_uids = None
        uid_array = batch.non_tensor_batch.get("uid")
        if uid_array is not None and uid_array.shape[0] == batch_size:
            try:
                uid_matrix = uid_array.reshape(-1, repeat_times)
                failed_uids = uid_matrix[
                    all_negative_mask.cpu().numpy(), 0
                ].tolist()
                failed_batch.meta_info["failed_prompt_uids"] = failed_uids
                failed_prompts.meta_info["failed_prompt_uids"] = failed_uids
            except Exception:
                failed_uids = None
                pass

        # ==== feedback 分组逻辑开始 ====
        feedback_group_size = int(getattr(self, "feedback_group_size", 2) or 2)
        total_failed_rollouts = failed_prompt_count * repeat_times

        if (
            feedback_group_size > 0
            and failed_rollout_size > 0
            and total_failed_rollouts == failed_rollout_size
        ):
            # local_index_matrix[p, :] = 这个 failed prompt 在 failed_batch 里的 rollout 行号
            local_index_matrix = torch.arange(total_failed_rollouts).view(
                failed_prompt_count, repeat_times
            )

            # 每一行 rollout 对应的 group_id（长度 == failed_rollout_size）
            group_ids_per_row: list[int] = [-1] * total_failed_rollouts
            # 每个 group 对应哪个 failed prompt（下标是 failed_prompts 里的 index）
            group_prompt_indices: list[int] = []
            # 如果有 uid，顺便记录每个 group 的 prompt uid
            group_prompt_uids: list[Any] = []
            # 每个 failed prompt 有多少个 group
            num_groups_per_prompt: list[int] = []

            for p_idx in range(failed_prompt_count):
                # 这个 failed prompt 在 failed_batch 里的 rollout 行号（长度 repeat_times）
                row_indices = local_index_matrix[p_idx]

                # 打乱一下，实现「随机分组」
                perm = torch.randperm(
                    row_indices.numel(), device=row_indices.device
                )
                shuffled = row_indices[perm]

                # 以 k 为单位切块：[k, k, k, ...]，最后不够 k 也单独成一组
                start = 0
                this_prompt_group_count = 0
                while start < shuffled.numel():
                    chunk = shuffled[start : start + feedback_group_size]
                    if chunk.numel() == 0:
                        break
                    group_id = len(group_prompt_indices)

                    for idx in chunk.tolist():
                        group_ids_per_row[idx] = group_id

                    group_prompt_indices.append(p_idx)
                    if (
                        failed_uids is not None
                        and len(failed_uids) == failed_prompt_count
                    ):
                        group_prompt_uids.append(failed_uids[p_idx])
                    this_prompt_group_count += 1
                    start += feedback_group_size

                num_groups_per_prompt.append(this_prompt_group_count)

            # ====== 关键改动：按行对齐的 group_ids 放 non_tensor_batch，保证多卡 shard 安全 ======
            failed_batch.non_tensor_batch["feedback_group_ids"] = np.array(
                group_ids_per_row, dtype=object
            )

            # 这些是“全局级别”信息，放 meta_info 即可
            failed_batch.meta_info["feedback_group_size"] = feedback_group_size
            failed_batch.meta_info["feedback_num_groups"] = len(group_prompt_indices)
            failed_batch.meta_info["feedback_group_prompt_indices"] = group_prompt_indices
            if group_prompt_uids:
                failed_batch.meta_info["feedback_group_prompt_uids"] = group_prompt_uids

            failed_prompts.meta_info["feedback_num_groups_per_prompt"] = num_groups_per_prompt
        # ==== feedback 分组逻辑结束 ====

        self.latest_all_negative_batch = failed_batch
        self.latest_all_negative_prompts = failed_prompts
        metrics = {
            "rollout/all_negative_prompt_cnt": failed_prompt_count,
            "rollout/all_negative_sample_cnt": failed_prompt_count * repeat_times,
        }
        return failed_batch, failed_prompts, metrics


    def _build_regen_batch(self, failed_prompts: DataProto, failed_feedback_batch: DataProto) -> DataProto:
        """基于原始 prompt + feedback，构造第二轮重生成用的 batch。"""

        prompt_texts = failed_prompts.non_tensor_batch.get("prompt_text")

        # 2. feedback 文本：来自 feedback 流
        feedback_texts = failed_feedback_batch.non_tensor_batch["feedback_text"]
        if isinstance(feedback_texts, np.ndarray):
            feedback_texts = feedback_texts.tolist()
        print("prompt_texts:",prompt_texts)
        print("feedback_texts:",feedback_texts)
        # for i in range(len(prompt_texts)):
        #     print(f"prompt_texts[{i}]:",prompt_texts[i])
        #     print(f"feedback_texts[{i}]:",feedback_texts[i])
        if len(prompt_texts) != len(feedback_texts):
            raise ValueError(
                f"Prompt/feedback size mismatch in _build_regen_batch: "
                f"{len(prompt_texts)} vs {len(feedback_texts)}"
            )

        # 3. 非任务特定的 regen 模板（不再硬编码 math 逻辑）
        regen_template = (
            "{prompt}\n\n"
            "--------------------\n"
            "Feedback about your previous answer:\n"
            "{feedback}\n"
            "--------------------\n\n"
            "Please answer the original prompt again, improving your response according to the feedback above.\n"
        )

        merged_texts = [
            regen_template.format(prompt=p, feedback=f)
            for p, f in zip(prompt_texts, feedback_texts)
        ]

        # 4. tokenizer 编码 regen 文本
        # tokenized = self.tokenizer(
        #     merged_texts,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        tokenized = self.tokenizer(
            merged_texts,
            padding="max_length",      # ✅ 所有样本统一 pad 到 max_prompt_len
            truncation=True,
            max_length=self.config.data.get("max_prompt_length"), # ✅ 明确上限
            return_tensors="pt",
        )

        # 推断 device：优先用 failed_feedback_batch.device，没有就从它 batch 里拿
        device = getattr(failed_feedback_batch, "device", None)
        if device is None:
            for v in failed_feedback_batch.batch.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break

        failed_feedback_batch.batch["input_ids"] = tokenized["input_ids"].to(device)
        failed_feedback_batch.batch["attention_mask"] = tokenized["attention_mask"].to(device)
        if "position_ids" in failed_feedback_batch.batch:
            failed_feedback_batch.batch["position_ids"] = build_position_ids(
                tokenized["attention_mask"]
            ).to(device)

        # 5. 从 feedback_batch 现有的 messages 里推断 ChatMessage 类型
        msgs_array = failed_feedback_batch.non_tensor_batch.get("messages")
        if msgs_array is None or len(msgs_array) == 0:
            raise ValueError(
                "failed_feedback_batch.non_tensor_batch['messages'] is missing or empty; "
                "cannot infer Message class to build raw_prompt."
            )

        sample_msgs = msgs_array[0]["messages"]
        if not sample_msgs:
            raise ValueError("Empty messages in failed_feedback_batch.non_tensor_batch['messages'][0]")
        MessageCls = type(sample_msgs[0])

        # 6. 构造新的 raw_prompt：对 regen 来说，一个 user message 就够了
        regen_raw_prompts = [
            [MessageCls(role="user", content=text)]
            for text in merged_texts
        ]

        failed_feedback_batch.non_tensor_batch = dict(failed_feedback_batch.non_tensor_batch)
        failed_feedback_batch.non_tensor_batch["raw_prompt"] = np.array(
            regen_raw_prompts, dtype=object
        )

        # 7. tools_kwargs 简化：regen 这一步我们通常不再用工具，统一给空 dict
        num_samples = len(merged_texts)
        failed_feedback_batch.non_tensor_batch["tools_kwargs"] = np.array(
            [{} for _ in range(num_samples)],
            dtype=object,
        )

        # 8. 记录 regen 用到的文本，方便 debug
        failed_feedback_batch.non_tensor_batch["regen_prompt"] = np.array(
            merged_texts, dtype=object
        )

        return failed_feedback_batch


    # def _merge_failed_feedbacks_global(
    #     self,
    #     feedback_batch: DataProto,
    #     repeat_times: int,
    #     failed_prompt_count: int,
    # ) -> DataProto:
    #     """
    #     在 *全局* 的 feedback_batch 上，把 rollout / group 粒度的 feedback
    #     合并成 prompt 粒度的 feedback。

    #     新逻辑：
    #     - 优先使用 non_tensor_batch["feedback_group_keys"] 里的 prompt_idx 信息，
    #       把同一题的所有 feedback 收集成 list[str]。
    #     - 构造一个新的 DataProto（每行对应一道题，带一个 feedback_list），
    #       通过 actor_rollout_wg.generate_feedbacks 做「二次 summarize」。
    #     - 返回值仍然是 prompt 粒度的 feedback_batch，meta_info["failed_prompt_count"]
    #       表示题目数。
    #     """
    #     import torch
    #     import numpy as np
    #     from collections import defaultdict

    #     # ===== 0. 空 batch 直接返回 =====
    #     batch_size = None
    #     for v in feedback_batch.batch.values():
    #         if isinstance(v, torch.Tensor):
    #             batch_size = v.shape[0]
    #             break
    #     if batch_size is None or batch_size == 0:
    #         return feedback_batch

    #     # ===== 1. 看有没有我们之前塞进去的分组信息 =====
    #     non_tensor = getattr(feedback_batch, "non_tensor_batch", {}) or {}
    #     group_keys_arr = non_tensor.get("feedback_group_keys", None)

    #     # TODO: 这里假设「第一阶段」的 per-group 文本 feedback
    #     # 存在 non_tensor_batch["feedback_text"] 里，如果你实际字段叫别的，
    #     # 比如 "tool_feedback" / "feedback", 把下面这一行改一下即可。
    #     feedback_text_field = non_tensor.get("feedback_text", None)
    #     # print("group_keys_arr:",group_keys_arr)
    #     # print("feedback_text_field:",feedback_text_field)
    #     # 如果没有分组信息或没有 feedback 文本，就退回到旧的随机采样逻辑
    #     if group_keys_arr is None or feedback_text_field is None:
    #         # ===== 旧逻辑：每个 prompt 从 repeat_times 个样本里随机挑 1 个 =====
    #         assert batch_size % repeat_times == 0, (
    #             f"Global feedback_batch size {batch_size} not divisible by repeat_times={repeat_times}"
    #         )
    #         num_prompts = batch_size // repeat_times
    #         if num_prompts != failed_prompt_count:
    #             print(
    #                 f"[merge_failed_feedbacks_global] num_prompts={num_prompts} != "
    #                 f"failed_prompt_count={failed_prompt_count}"
    #             )

    #         import random
    #         selected_indices: list[int] = []
    #         for prompt_idx in range(num_prompts):
    #             start = prompt_idx * repeat_times
    #             end = start + repeat_times
    #             chosen = random.randint(start, end - 1)
    #             selected_indices.append(chosen)

    #         device = next(iter(feedback_batch.batch.values())).device
    #         idx_tensor = torch.tensor(selected_indices, device=device)
    #         merged = feedback_batch.select_idxs(idx_tensor)

    #         merged.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
    #         merged.meta_info["repeat_times"] = 1
    #         merged.meta_info["failed_prompt_count"] = num_prompts
    #         return merged

    #     # ===== 2. 有分组信息：按 prompt 收集所有 group-feedback =====
    #     # group_keys_arr: np.array(shape=[num_groups], dtype=object)
    #     group_keys: list[dict] = (
    #         group_keys_arr.tolist()
    #         if isinstance(group_keys_arr, np.ndarray)
    #         else list(group_keys_arr)
    #     )

    #     feedback_texts: list[str] = (
    #         feedback_text_field.tolist()
    #         if isinstance(feedback_text_field, np.ndarray)
    #         else list(feedback_text_field)
    #     )

    #     if len(group_keys) != batch_size or len(feedback_texts) != batch_size:
    #         raise ValueError(
    #             f"[merge_failed_feedbacks_global] feedback_group_keys / feedback_text "
    #             f"length mismatch with batch_size: "
    #             f"{len(group_keys)} / {len(feedback_texts)} / {batch_size}"
    #         )

    #     # print("goto big merge_failed_feedbacks_global")
    #     # prompt_key -> list(feedback_texts)
    #     prompt_to_texts: dict[int, list[str]] = defaultdict(list)
    #     # prompt_key -> 代表这一题的某个 row idx（方便后面复用 skeleton）
    #     prompt_to_rep_row: dict[int, int] = {}

    #     for row_idx, key in enumerate(group_keys):
    #         if key is None:
    #             # 理论上不会出现，稳妥一点直接跳过
    #             continue

    #         prompt_idx = key.get("prompt_idx", None)
    #         # 如果 prompt_idx 为空，就退回到用 row_idx 作为 key（意味着一题一 group）
    #         prompt_key = int(prompt_idx) if prompt_idx is not None else int(row_idx)

    #         prompt_to_texts[prompt_key].append(feedback_texts[row_idx])
    #         if prompt_key not in prompt_to_rep_row:
    #             prompt_to_rep_row[prompt_key] = row_idx

    #     num_prompts = len(prompt_to_texts)
    #     if num_prompts == 0:
    #         # 没有任何有效分组，当空处理
    #         return feedback_batch

    #     # ===== 3. 构造「二阶 summarize」的输入 batch =====
    #     # 用每个 prompt 的代表 row 当 skeleton，保证 tensor 形状兼容
    #     rep_row_indices = [prompt_to_rep_row[k] for k in sorted(prompt_to_rep_row.keys())]
    #     device = next(iter(feedback_batch.batch.values())).device
    #     rep_idx_tensor = torch.tensor(rep_row_indices, device=device)

    #     # 这一步之后，batch 维度 = 题目数；其它 tensor 字段沿用代表行的值
    #     merged_input = feedback_batch.select_idxs(rep_idx_tensor)

    #     # 替换 / 新增非 tensor 字段：把 per-group feedback 合成 list[str]
    #     merged_input.non_tensor_batch = dict(getattr(merged_input, "non_tensor_batch", {}) or {})

    #     merged_feedback_lists: list[list[str]] = [
    #         prompt_to_texts[k] for k in sorted(prompt_to_texts.keys())
    #     ]
    #     merged_input.non_tensor_batch["feedback_list"] = np.array(merged_feedback_lists, dtype=object)

    #     # 为了之后在 rollout 侧好区分「合并模式」，在 meta_info 打一个标记
    #     merged_input.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
    #     merged_input.meta_info["merge"] = True
    #     merged_input.meta_info["failed_prompt_count"] = num_prompts
    #     merged_input.meta_info["repeat_times"] = 1  # 这个 batch 里已经是一题一条了

    #     # 你还可以把 prompt_idx 映射也带过去，方便 rollout 侧做更强的控制
    #     merged_input.non_tensor_batch["merge_prompt_indices"] = np.array(
    #         sorted(prompt_to_texts.keys()), dtype=object
    #     )

    #     # ===== 4. 调远端 summarize：第二阶段 generate_feedbacks =====
    #     # 注意：Ray 这层通常不支持直接传 merge=True 这样的 kwargs，
    #     # 建议用 meta_info["merge"] 来区分模式（上面已经打了）。
    #     merged_output = self.actor_rollout_wg.generate_feedbacks(merged_input)

    #     # ===== 5. 收尾：保证返回形状 & meta_info 和原函数语义一致 =====
    #     merged_output.meta_info = dict(getattr(merged_output, "meta_info", {}) or {})
    #     merged_output.meta_info["repeat_times"] = 1
    #     merged_output.meta_info["failed_prompt_count"] = num_prompts
    #     # print("Merged feedback:",merged_output.non_tensor_batch.get("feedback_text", None))
    #     return merged_output

    # def _merge_failed_feedbacks_global(
    #     self,
    #     feedback_batch: DataProto,
    #     repeat_times: int,
    #     failed_prompt_count: int,
    # ) -> DataProto:
    #     """
    #     在 *全局* 的 feedback_batch 上，把 rollout / group 粒度的 feedback
    #     合并成 prompt 粒度的 feedback。

    #     逻辑：
    #     1）先把同一题的 group-feedback 聚成 list[str]；
    #     2）构造 merged_input（每行一道题，带 feedback_list）；
    #     3）根据 DP 个数做 padding，让 len(merged_input) 能被 dp_size 整除；
    #     4）调用 actor_rollout_wg.generate_feedbacks 做二次 summarize；
    #     5）去掉 padding 行，返回最终的 prompt 粒度 feedback_batch。
    #     """
    #     import torch
    #     import numpy as np
    #     from collections import defaultdict
    #     import math

    #     # ===== 0. 空 batch 直接返回 =====
    #     batch_size = None
    #     for v in feedback_batch.batch.values():
    #         if isinstance(v, torch.Tensor):
    #             batch_size = v.shape[0]
    #             break
    #     if batch_size is None or batch_size == 0:
    #         return feedback_batch

    #     # ===== 1. 取分组信息 + 第一阶段文本 feedback =====
    #     non_tensor = getattr(feedback_batch, "non_tensor_batch", {}) or {}

    #     group_keys_arr = non_tensor.get("feedback_group_keys", None)
    #     feedback_text_field = non_tensor.get("feedback_text", None)

    #     # 兜底：如果没有分组信息 / feedback_text，就退回到“随机挑一条”的旧逻辑
    #     if group_keys_arr is None or feedback_text_field is None:
    #         assert batch_size % repeat_times == 0, (
    #             f"Global feedback_batch size {batch_size} not divisible by repeat_times={repeat_times}"
    #         )
    #         num_prompts = batch_size // repeat_times
    #         if num_prompts != failed_prompt_count:
    #             print(
    #                 f"[merge_failed_feedbacks_global] num_prompts={num_prompts} != "
    #                 f"failed_prompt_count={failed_prompt_count}"
    #             )

    #         import random
    #         selected_indices: list[int] = []
    #         for prompt_idx in range(num_prompts):
    #             start = prompt_idx * repeat_times
    #             end = start + repeat_times
    #             chosen = random.randint(start, end - 1)
    #             selected_indices.append(chosen)

    #         device = next(iter(feedback_batch.batch.values())).device
    #         idx_tensor = torch.tensor(selected_indices, device=device)
    #         merged = feedback_batch.select_idxs(idx_tensor)

    #         merged.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
    #         merged.meta_info["repeat_times"] = 1
    #         merged.meta_info["failed_prompt_count"] = num_prompts
    #         return merged

    #     # numpy / list 统一化
    #     group_keys: list[dict] = (
    #         group_keys_arr.tolist()
    #         if isinstance(group_keys_arr, np.ndarray)
    #         else list(group_keys_arr)
    #     )
    #     feedback_texts: list[str] = (
    #         feedback_text_field.tolist()
    #         if isinstance(feedback_text_field, np.ndarray)
    #         else list(feedback_text_field)
    #     )

    #     if len(group_keys) != batch_size or len(feedback_texts) != batch_size:
    #         raise ValueError(
    #             f"[merge_failed_feedbacks_global] feedback_group_keys / feedback_text "
    #             f"length mismatch with batch_size: "
    #             f"{len(group_keys)} / {len(feedback_texts)} / {batch_size}"
    #         )

    #     # ===== 2. 按 “题目” 聚合所有 group-feedback =====
    #     # prompt_key -> list(feedback_texts)
    #     prompt_to_texts: dict[int, list[str]] = defaultdict(list)
    #     # prompt_key -> 代表这一题的某个 row idx（方便下面 select_idxs 复用 tensor）
    #     prompt_to_rep_row: dict[int, int] = {}

    #     for row_idx, key in enumerate(group_keys):
    #         if key is None:
    #             continue

    #         prompt_idx = key.get("prompt_idx", None)
    #         # prompt_idx 为空就用 row_idx 当 key（退化为一题一 group）
    #         prompt_key = int(prompt_idx) if prompt_idx is not None else int(row_idx)

    #         prompt_to_texts[prompt_key].append(feedback_texts[row_idx])
    #         if prompt_key not in prompt_to_rep_row:
    #             prompt_to_rep_row[prompt_key] = row_idx

    #     num_prompts = len(prompt_to_texts)
    #     if num_prompts == 0:
    #         return feedback_batch

    #     # ===== 3. 构造「二阶 summarize」的输入 merged_input =====
    #     device = next(iter(feedback_batch.batch.values())).device
    #     rep_row_indices = [prompt_to_rep_row[k] for k in sorted(prompt_to_rep_row.keys())]
    #     rep_idx_tensor = torch.tensor(rep_row_indices, device=device)

    #     # 这一步之后，batch 维度 = 题目数；其它 tensor 字段沿用代表行的值
    #     merged_input = feedback_batch.select_idxs(rep_idx_tensor)

    #     merged_input.non_tensor_batch = dict(
    #         getattr(merged_input, "non_tensor_batch", {}) or {}
    #     )

    #     merged_feedback_lists: list[list[str]] = [
    #         prompt_to_texts[k] for k in sorted(prompt_to_texts.keys())
    #     ]
    #     merged_input.non_tensor_batch["feedback_list"] = np.array(
    #         merged_feedback_lists, dtype=object
    #     )

    #     # meta 标记
    #     merged_input.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
    #     merged_input.meta_info["merge"] = True
    #     merged_input.meta_info["failed_prompt_count"] = num_prompts
    #     merged_input.meta_info["repeat_times"] = 1
    #     merged_input.non_tensor_batch["merge_prompt_indices"] = np.array(
    #         sorted(prompt_to_texts.keys()), dtype=object
    #     )

    #     # ===== 3.5 计算 DP 大小，并对 merged_input 做 padding =====
    #     # 尽量根据 config 推断 dp_size；推不到就当 1
    #     trainer_cfg = getattr(self.config, "trainer", None)
    #     ar_cfg = getattr(self.config, "actor_rollout_ref", None)

    #     world_gpus = 1
    #     if trainer_cfg is not None:
    #         world_gpus = int(getattr(trainer_cfg, "n_gpus_per_node", 1)) * int(
    #             getattr(trainer_cfg, "nnodes", 1)
    #         )

    #     tp_size = 1
    #     if ar_cfg is not None and hasattr(ar_cfg, "rollout"):
    #         tp_size = int(getattr(ar_cfg.rollout, "tensor_model_parallel_size", 1) or 1)

    #     dp_size = max(1, world_gpus // max(1, tp_size))

    #     # pad 到最近的 dp_size 的倍数
    #     if dp_size > 1:
    #         orig_num = num_prompts
    #         target_num = int(math.ceil(orig_num / dp_size) * dp_size)
    #         pad_num = target_num - orig_num

    #         if pad_num > 0:
    #             # 用最后一行重复填充
    #             pad_indices = [orig_num - 1] * pad_num
    #             all_indices = list(range(orig_num)) + pad_indices
    #             all_idx_tensor = torch.tensor(all_indices, device=device)

    #             merged_padded = merged_input.select_idxs(all_idx_tensor)

    #             nt = dict(merged_padded.non_tensor_batch)
    #             # True 表示 padding 行
    #             is_pad = np.array(
    #                 [False] * orig_num + [True] * pad_num,
    #                 dtype=bool,
    #             )
    #             nt["merge_is_pad"] = is_pad
    #             merged_padded.non_tensor_batch = nt
    #         else:
    #             merged_padded = merged_input
    #             nt = dict(merged_input.non_tensor_batch)
    #             nt["merge_is_pad"] = np.array([False] * num_prompts, dtype=bool)
    #             merged_padded.non_tensor_batch = nt
    #     else:
    #         merged_padded = merged_input
    #         nt = dict(merged_input.non_tensor_batch)
    #         nt["merge_is_pad"] = np.array([False] * num_prompts, dtype=bool)
    #         merged_padded.non_tensor_batch = nt

    #     # ===== 4. 调远端 summarize：第二阶段 generate_feedbacks =====
    #     merged_output = self.actor_rollout_wg.generate_feedbacks(merged_padded)

    #     # ===== 5. 去掉 padding 行，只保留真实题目 =====
    #     nt_out = getattr(merged_output, "non_tensor_batch", {}) or {}
    #     pad_mask = nt_out.get("merge_is_pad", None)

    #     if pad_mask is not None:
    #         if isinstance(pad_mask, np.ndarray):
    #             pad_mask_list = pad_mask.tolist()
    #         else:
    #             pad_mask_list = list(pad_mask)

    #         keep_indices = [i for i, is_pad in enumerate(pad_mask_list) if not is_pad]

    #         if keep_indices:
    #             keep_idx_tensor = torch.tensor(keep_indices, device=device)
    #             merged_output = merged_output.select_idxs(keep_idx_tensor)

    #         merged_output.non_tensor_batch = dict(merged_output.non_tensor_batch)
    #         merged_output.non_tensor_batch.pop("merge_is_pad", None)

    #     # ===== 6. 收尾 =====
    #     merged_output.meta_info = dict(getattr(merged_output, "meta_info", {}) or {})
    #     merged_output.meta_info["repeat_times"] = 1
    #     merged_output.meta_info["failed_prompt_count"] = num_prompts

    #     return merged_output
    
    def _merge_failed_feedbacks_global(
        self,
        feedback_batch: DataProto,
        repeat_times: int,
        failed_prompt_count: int,
    ) -> DataProto:
        """
        在 *全局* 的 feedback_batch 上，把 rollout / group 粒度的 feedback
        合并成 prompt 粒度的 feedback。

        新逻辑：
        - 第一阶段：feedback_batch 是「按 group」的输出，
        non_tensor_batch["feedback_group_keys"] 里有每个 group 对应的 prompt_idx。
        - 这里先把同一题的所有 group-level feedback 收成 list[str]，
        得到 merged_input（一题一条，带 feedback_list）。
        - 第二阶段：调用 actor_rollout_wg.generate_feedbacks(merged_input)，让工具做 summarize。
        为了适配多卡 DP：
            * 如果 num_prompts % dp_size != 0，则对 merged_input 做 padding；
            * 调完 generate_feedbacks 之后，再把 padding 部分裁掉，只保留前 num_prompts 条。
        """

        import torch
        import numpy as np
        from collections import defaultdict

        # ===== 0. 空 batch 直接返回 =====
        batch_size = None
        for v in feedback_batch.batch.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
        if batch_size is None or batch_size == 0:
            return feedback_batch

        # ===== 1. 读取我们在第一阶段塞进去的辅助字段 =====
        non_tensor = getattr(feedback_batch, "non_tensor_batch", {}) or {}
        group_keys_arr = non_tensor.get("feedback_group_keys", None)
        feedback_text_field = non_tensor.get("feedback_text", None)
        # print("feedback_text_field:",feedback_text_field)
        # 没有分组信息或没有 feedback 文本，就退回旧逻辑：每个 prompt 随机挑 1 个
        if group_keys_arr is None or feedback_text_field is None:
            assert batch_size % repeat_times == 0, (
                f"Global feedback_batch size {batch_size} not divisible by repeat_times={repeat_times}"
            )
            num_prompts = batch_size // repeat_times
            if num_prompts != failed_prompt_count:
                print(
                    f"[merge_failed_feedbacks_global] num_prompts={num_prompts} != "
                    f"failed_prompt_count={failed_prompt_count}"
                )

            import random
            selected_indices: list[int] = []
            for prompt_idx in range(num_prompts):
                start = prompt_idx * repeat_times
                end = start + repeat_times
                chosen = random.randint(start, end - 1)
                selected_indices.append(chosen)

            device = next(iter(feedback_batch.batch.values())).device
            idx_tensor = torch.tensor(selected_indices, device=device)
            merged = feedback_batch.select_idxs(idx_tensor)

            merged.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
            merged.meta_info["repeat_times"] = 1
            merged.meta_info["failed_prompt_count"] = num_prompts
            return merged

        # ===== 2. 有分组信息：按 prompt 收集所有 group-feedback =====
        # group_keys_arr: np.array(shape=[num_groups], dtype=object)
        group_keys: list[dict] = (
            group_keys_arr.tolist()
            if isinstance(group_keys_arr, np.ndarray)
            else list(group_keys_arr)
        )

        feedback_texts: list[str] = (
            feedback_text_field.tolist()
            if isinstance(feedback_text_field, np.ndarray)
            else list(feedback_text_field)
        )

        if len(group_keys) != batch_size or len(feedback_texts) != batch_size:
            raise ValueError(
                f"[merge_failed_feedbacks_global] feedback_group_keys / feedback_text "
                f"length mismatch with batch_size: "
                f"{len(group_keys)} / {len(feedback_texts)} / {batch_size}"
            )

        # prompt_key -> list(feedback_texts)
        prompt_to_texts: dict[int, list[str]] = defaultdict(list)
        # prompt_key -> 代表这一题的某个 row idx（方便后面复用 skeleton）
        prompt_to_rep_row: dict[int, int] = {}

        for row_idx, key in enumerate(group_keys):
            if key is None:
                continue

            prompt_idx = key.get("prompt_idx", None)
            # 如果 prompt_idx 为空，就退回到用 row_idx 作为 key（意味着一题一 group）
            prompt_key = int(prompt_idx) if prompt_idx is not None else int(row_idx)

            prompt_to_texts[prompt_key].append(feedback_texts[row_idx])
            if prompt_key not in prompt_to_rep_row:
                prompt_to_rep_row[prompt_key] = row_idx

        num_prompts = len(prompt_to_texts)
        if num_prompts == 0:
            # 没有任何有效分组，当空处理
            return feedback_batch

        # ===== 3. 构造「二阶 summarize」的输入 batch =====
        # 用每个 prompt 的代表 row 当 skeleton，保证 tensor 形状兼容
        rep_row_indices = [prompt_to_rep_row[k] for k in sorted(prompt_to_rep_row.keys())]
        device = next(iter(feedback_batch.batch.values())).device
        rep_idx_tensor = torch.tensor(rep_row_indices, device=device)

        # 这一步之后，batch 维度 = 题目数；其它 tensor 字段沿用代表行的值
        merged_input = feedback_batch.select_idxs(rep_idx_tensor)

        # 替换 / 新增非 tensor 字段：把 per-group feedback 合成 list[str]
        merged_input.non_tensor_batch = dict(
            getattr(merged_input, "non_tensor_batch", {}) or {}
        )

        merged_feedback_lists: list[list[str]] = [
            prompt_to_texts[k] for k in sorted(prompt_to_texts.keys())
        ]
        merged_input.non_tensor_batch["feedback_list"] = np.array(
            merged_feedback_lists, dtype=object
        )

        # meta_info 标记：告诉 rollout 这是「merge 第二阶段」
        merged_input.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
        merged_input.meta_info["merge"] = True
        merged_input.meta_info["failed_prompt_count"] = num_prompts
        merged_input.meta_info["repeat_times"] = 1  # 这个 batch 里已经是一题一条了

        # 你还可以把 prompt_idx 映射也带过去，方便 rollout 侧做更强的控制
        merged_input.non_tensor_batch["merge_prompt_indices"] = np.array(
            sorted(prompt_to_texts.keys()), dtype=object
        )

        # ===== 4. 多卡 DP 适配：padding + 调用 generate_feedbacks + unpadding =====
        # 推断 dp_size（按你的实际 trainer 实现改一下这行）
        try:
            dp_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        except Exception:
            # 不行就退回 1 卡逻辑
            dp_size = 1

        # if dp_size <= 1:
        #     # 单卡训练：直接调即可
        #     merged_output = self.actor_rollout_wg.generate_feedbacks(merged_input)
        # else:
        #     # 多卡：保证 DataProto 长度能被 dp_size 整除
        #     if num_prompts % dp_size == 0:
        #         merged_padded = merged_input
        #         num_pad = 0
        #     else:
        #         num_pad = dp_size - (num_prompts % dp_size)
        #         last_idx = num_prompts - 1
        #         pad_indices = torch.full(
        #             (num_pad,), last_idx, dtype=torch.long, device=device
        #         )
        #         padded_tail = merged_input.select_idxs(pad_indices)
        #         merged_padded = merged_input.concat(padded_tail)

        #     # 调用多卡 rollout
        #     merged_output_padded = self.actor_rollout_wg.generate_feedbacks(merged_padded)

        #     # —— 关键：把 padding 出来的那几条裁掉，只保留前 num_prompts 条 —— #
        #     out_device = next(iter(merged_output_padded.batch.values())).device
        #     keep_indices = torch.arange(num_prompts, device=out_device)
        #     merged_output = merged_output_padded.select_idxs(keep_indices)

        if dp_size <= 1:
            # 单卡训练：直接调即可
            merged_output = self.actor_rollout_wg.generate_feedbacks(merged_input)
        else:
            # 多卡：保证 DataProto 长度能被 dp_size 整除
            if num_prompts % dp_size == 0:
                merged_padded = merged_input
                num_pad = 0
            else:
                num_pad = dp_size - (num_prompts % dp_size)
                last_idx = num_prompts - 1

                # 用「索引拼接」替代 DataProto.concat，避免 tensordict.cat 的坑
                base_indices = torch.arange(num_prompts, device=device)
                pad_indices = torch.full(
                    (num_pad,), last_idx, dtype=torch.long, device=device
                )
                all_indices = torch.cat([base_indices, pad_indices], dim=0)

                merged_padded = merged_input.select_idxs(all_indices)

            # 调用多卡 rollout
            merged_output_padded = self.actor_rollout_wg.generate_feedbacks(merged_padded)

            # 把 padding 出来的那几条裁掉，只保留前 num_prompts 条
            out_device = next(iter(merged_output_padded.batch.values())).device
            keep_indices = torch.arange(num_prompts, device=out_device)
            merged_output = merged_output_padded.select_idxs(keep_indices)

        # ===== 5. 收尾：保证返回形状 & meta_info 和原函数语义一致 =====
        merged_output.meta_info = dict(getattr(merged_output, "meta_info", {}) or {})
        merged_output.meta_info["repeat_times"] = 1
        merged_output.meta_info["failed_prompt_count"] = num_prompts

        return merged_output


    def _pad_dataproto_to_multiple(self, dp: DataProto, multiple: int) -> DataProto:
        """把 DataProto 的长度 pad 到 multiple 的整数倍（简单重复一些样本即可）"""
        import torch

        size = len(dp)
        if size == 0:
            return dp

        remainder = size % multiple
        if remainder == 0:
            return dp

        pad = multiple - remainder

        # 设备上随便取几条样本重复
        idx_all = torch.arange(size, device=next(iter(dp.batch.values())).device)
        pad_idx = idx_all[:pad]  # 比如重复前 pad 条，也可以随机选

        new_idx = torch.cat([idx_all, pad_idx], dim=0)
        return dp.select_idxs(new_idx)

    def _replace_failed_with_regen(
        self,
        batch: DataProto,
        failed_batch: DataProto,
        regen_batch_output: DataProto,
    ) -> DataProto:
        """
        在大 batch（长度 B * n）中，把 all-negative 的 rollout 行，
        用第二轮 regen 的结果逐条覆盖掉。

        参数：
            batch:
                原始 rollout 后的 batch，长度 B * n
            failed_batch:
                _collect_all_negative_rollout_prompts 返回的 failed_batch，
                长度 F * n，meta_info 里有:
                - "failed_rollout_indices": 这些行在 batch 里的全局索引
            regen_batch_output:
                第二轮 generate_sequences 输出，长度 >= F * n
                （外面已经用 select_idxs 截断到 F * n）
        """
        import torch
        import numpy as np
        # 1. 从 meta_info 里拿到要替换的全局下标
        index_list = failed_batch.meta_info.get("failed_rollout_indices", None)
        if index_list is None:
            raise ValueError(
                "failed_batch.meta_info['failed_rollout_indices'] 缺失，"
                "请确认在 _collect_all_negative_rollout_prompts 中已写入该字段。"
            )

        # 找一个 batch 里的张量拿 device
        some_tensor = None
        for v in batch.batch.values():
            if isinstance(v, torch.Tensor):
                some_tensor = v
                break
        if some_tensor is None:
            # 极端情况：batch 里没有 tensor，就直接返回
            return batch

        device = some_tensor.device
        idx = torch.as_tensor(index_list, device=device, dtype=torch.long)

        if idx.numel() != len(regen_batch_output):
            raise ValueError(
                f"失败 rollout 个数 ({idx.numel()}) 和 regen_batch_output 长度 ({len(regen_batch_output)}) 不一致，"
                "请检查 repeat / padding / 截断逻辑是否一致。"
            )

        # 2. 替换 tensor 部分
        for key, tensor in batch.batch.items():
            if key not in regen_batch_output.batch:
                # 有些 key 只在大 batch 里存在（比如 rm_scores），就跳过
                continue

            new_tensor = regen_batch_output.batch[key]

            if not (isinstance(tensor, torch.Tensor) and isinstance(new_tensor, torch.Tensor)):
                continue

            # 只处理 batch 维度 >= 1
            if tensor.dim() == 0 or new_tensor.dim() == 0:
                continue

            # 检查除了 batch 维外的形状是否一致
            if tensor.shape[1:] != new_tensor.shape[1:]:
                raise ValueError(
                    f"Tensor shape mismatch on key '{key}': "
                    f"orig {tuple(tensor.shape)} vs regen {tuple(new_tensor.shape)}"
                )

            # 原地覆盖：把 batch 中 idx 这些行，用 regen 的对应行覆盖
            tensor[idx] = new_tensor[: idx.numel()]

        # 3. 替换 non_tensor_batch（例如 uid / messages / reward_scores 等）
        if batch.non_tensor_batch and regen_batch_output.non_tensor_batch:
            idx_np = idx.detach().cpu().numpy()

            for key, arr in regen_batch_output.non_tensor_batch.items():
                if key not in batch.non_tensor_batch:
                    continue

                dst_arr = batch.non_tensor_batch[key]
                # 简单防御：长度要够
                if len(dst_arr) <= idx_np.max():
                    raise ValueError(
                        f"Non-tensor field '{key}' length {len(dst_arr)} "
                        f"小于需要写入的最大 index {idx_np.max()}。"
                    )

                # 只在这些位置做覆盖
                dst_arr[idx_np] = arr[: idx_np.shape[0]]

        return batch


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        failed_batch, failed_prompts, failed_prompt_metrics = self._collect_all_negative_rollout_prompts_by_group(
                            batch,
                            reward_tensor,
                            self.config.actor_rollout_ref.rollout.n,
                        )
                        metrics.update(failed_prompt_metrics)

                    if failed_batch is not None and self.need_feedback:
                        with marked_timer("gen_feedback", timing_raw, color="red"):
                            # failed_batch_regen_output = failed_batch.repeat(
                            #     repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            # )
                            failed_batch_feedback = failed_batch
                            # print("prompt_text in failed_batch_feedback:", failed_batch_feedback.non_tensor_batch["prompt_text"]) 
                            if not self.async_rollout_mode:
                                failed_batch_feedback = self.actor_rollout_wg.generate_feedbacks(failed_batch_feedback)
                            else:
                                failed_batch_feedback = self.async_rollout_manager.generate_feedbacks(failed_batch_feedback)

                            timing_raw.update(failed_batch_feedback.meta_info["timing"])
                            failed_batch_feedback.meta_info.pop("timing", None) 

                            repeat_times = int(failed_batch.meta_info["repeat_times"])
                            failed_prompt_count = int(failed_batch.meta_info["failed_prompt_count"])
                            failed_batch_feedback = self._merge_failed_feedbacks_global(
                                failed_batch_feedback,
                                repeat_times=repeat_times,
                                failed_prompt_count=failed_prompt_count,
                            )

                        #TODO: CONTANTENT THE FEEDBACK AND ORIGINAL PROMPT (MAY CAN CHANGE THE CALCULATION OF THE REWARD)
                        with marked_timer("regen", timing_raw, color="red"):
                            regen_input = self._build_regen_batch(
                                failed_prompts=failed_prompts,
                                failed_feedback_batch=failed_batch_feedback,
                            )
                            # 1) 第二轮：对每个失败 prompt 再 repeat n 次
                            n_rollout = int(self.config.actor_rollout_ref.rollout.n)
                            regen_input = regen_input.repeat(repeat_times=n_rollout, interleave=True)  # len = F * n

                            # 2) 为了 DP 拆分，对 regen_input 做 padding
                            dp_size = (
                                self.actor_rollout_wg.dp_size
                                if hasattr(self.actor_rollout_wg, "dp_size")
                                else self.config.trainer.n_gpus_per_node
                            )
                            regen_input = self._pad_dataproto_to_multiple(regen_input, multiple=dp_size)
                            # dp_size = self.actor_rollout_wg.dp_size if hasattr(self.actor_rollout_wg, "dp_size") else self.config.trainer.n_gpus_per_node
                            # regen_input = self._pad_dataproto_to_multiple(regen_input, multiple=dp_size)
                            if not self.async_rollout_mode:
                                regen_batch_output = self.actor_rollout_wg.generate_sequences(regen_input)
                            else:
                                regen_batch_output = self.async_rollout_manager.generate_sequences(regen_input)

                            timing_raw.update(regen_batch_output.meta_info["timing"])
                            regen_batch_output.meta_info.pop("timing", None)
                            # raise ValueError("DONE")
                        # TODO: MERGE THE REGEN OUTPUT TO GEN OUTPUT AND CALCULATE THE REWARD AGAIN                    
                                            
                        orig_failed_rollout_num = len(failed_batch)  # failed_batch 长度就是 F * n

                        cur_len = len(regen_batch_output)
                        if cur_len > orig_failed_rollout_num:
                            # 找一个 tensor 拿 device
                            first_tensor = next(iter(regen_batch_output.batch.values()))
                            idx_device = first_tensor.device if hasattr(first_tensor, "device") else "cpu"
                            keep_idx = torch.arange(orig_failed_rollout_num, device=idx_device)
                            regen_batch_output = regen_batch_output.select_idxs(keep_idx)

                        # 5) 把第二轮 regen 的结果写回到 batch 里，替换原来的那些失败 rollout
                        batch = self._replace_failed_with_regen(
                            batch=batch,
                            failed_batch=failed_batch,              # ⚠️ 注意这里传的是 failed_batch
                            regen_batch_output=regen_batch_output,
                        )
                                    
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor


                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
