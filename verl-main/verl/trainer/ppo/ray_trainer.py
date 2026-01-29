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

        prompt_indices = torch.arange(batch_size, device=reward_tensor.device).view(
            -1, repeat_times
        )
        selected_indices = prompt_indices[all_negative_mask].reshape(-1).cpu()
        failed_batch = batch.select_idxs(selected_indices)

        prompt_anchor_indices = prompt_indices[all_negative_mask][:, 0].cpu()
        failed_prompts = batch.select_idxs(prompt_anchor_indices)

        failed_batch.non_tensor_batch = dict(getattr(failed_batch, "non_tensor_batch", {}) or {})
        failed_prompts.non_tensor_batch = dict(getattr(failed_prompts, "non_tensor_batch", {}) or {})

        prompt_texts_per_prompt: list[str] = []
        # if "prompts" in failed_prompts.batch:
        #     prompt_texts_per_prompt = _decode_token_sequences(failed_prompts.batch["prompts"])
        # elif "input_ids" in failed_prompts.batch:
        #     prompt_texts_per_prompt = _decode_token_sequences(
        #         failed_prompts.batch["input_ids"],
        #         failed_prompts.batch.get("attention_mask"),
        # 
        num_prompts = batch_size // repeat_times
        extra_info_mat = batch.non_tensor_batch["extra_info"].reshape(num_prompts, repeat_times)

        failed_anchor_extra = extra_info_mat[all_negative_mask.cpu().numpy(), 0]
        prompt_texts_per_prompt = [e["question_raw"] for e in failed_anchor_extra.tolist()]

        prompt_texts_repeated = np.repeat(np.array(prompt_texts_per_prompt, dtype=object), repeat_times)

        failed_batch.non_tensor_batch["prompt_text"] = prompt_texts_repeated
        failed_prompts.non_tensor_batch["prompt_text"] = np.array(prompt_texts_per_prompt, dtype=object)

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

        feedback_group_size = int(getattr(self, "feedback_group_size", 2) or 2)
        total_failed_rollouts = failed_prompt_count * repeat_times

        if (
            feedback_group_size > 0
            and failed_rollout_size > 0
            and total_failed_rollouts == failed_rollout_size
        ):
            local_index_matrix = torch.arange(total_failed_rollouts).view(
                failed_prompt_count, repeat_times
            )

            group_ids_per_row: list[int] = [-1] * total_failed_rollouts
            group_prompt_indices: list[int] = []
            group_prompt_uids: list[Any] = []
            num_groups_per_prompt: list[int] = []

            for p_idx in range(failed_prompt_count):
                row_indices = local_index_matrix[p_idx]

                perm = torch.randperm(
                    row_indices.numel(), device=row_indices.device
                )
                shuffled = row_indices[perm]

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

            failed_batch.non_tensor_batch["feedback_group_ids"] = np.array(
                group_ids_per_row, dtype=object
            )

            failed_batch.meta_info["feedback_group_size"] = feedback_group_size
            failed_batch.meta_info["feedback_num_groups"] = len(group_prompt_indices)
            failed_batch.meta_info["feedback_group_prompt_indices"] = group_prompt_indices
            if group_prompt_uids:
                failed_batch.meta_info["feedback_group_prompt_uids"] = group_prompt_uids

            failed_prompts.meta_info["feedback_num_groups_per_prompt"] = num_groups_per_prompt

        self.latest_all_negative_batch = failed_batch
        self.latest_all_negative_prompts = failed_prompts
        metrics = {
            "rollout/all_negative_prompt_cnt": failed_prompt_count,
            "rollout/all_negative_sample_cnt": failed_prompt_count * repeat_times,
        }
        return failed_batch, failed_prompts, metrics

    def _build_regen_batch(self, failed_prompts: DataProto, failed_feedback_batch: DataProto) -> DataProto:
        import numpy as np
        import torch

        question_texts = None
        base_nt = dict(getattr(failed_prompts, "non_tensor_batch", {}) or {})
        fb_nt = dict(getattr(failed_feedback_batch, "non_tensor_batch", {}) or {})

        base_nt.update(fb_nt)
        failed_feedback_batch.non_tensor_batch = base_nt

        extra_infos = failed_prompts.non_tensor_batch.get("extra_info")
        if extra_infos is not None:
            if isinstance(extra_infos, np.ndarray):
                extra_infos = extra_infos.tolist()
            question_texts = [(ei or {}).get("question", "") for ei in extra_infos]

        if not question_texts or all(q == "" for q in question_texts):
            prompt_texts = failed_prompts.non_tensor_batch.get("prompt_text")
            if prompt_texts is None:
                raise ValueError(
                    "Neither extra_info['question'] nor prompt_text is available "
                    "for failed_prompts; cannot build regen batch."
                )
            if isinstance(prompt_texts, np.ndarray):
                prompt_texts = prompt_texts.tolist()
            question_texts = prompt_texts
        else:
            prompt_texts = failed_prompts.non_tensor_batch.get("prompt_text")
            if isinstance(prompt_texts, np.ndarray):
                prompt_texts = prompt_texts.tolist()
            if prompt_texts is None:
                prompt_texts = ["" for _ in question_texts]
            question_texts = [
                q if q not in (None, "") else p
                for q, p in zip(question_texts, prompt_texts)
            ]

        feedback_texts = failed_feedback_batch.non_tensor_batch["feedback_text"]
        if isinstance(feedback_texts, np.ndarray):
            feedback_texts = feedback_texts.tolist()

        if len(question_texts) != len(feedback_texts):
            raise ValueError(
                f"Question/feedback size mismatch in _build_regen_batch: "
                f"{len(question_texts)} vs {len(feedback_texts)}"
            )

        seq_template = (
        "You are a reasoning assistant.\n"
        "Re-solve the problem from scratch. Use the provided feedback only as hints.\n"
        "Do NOT create another <feedback> block.\n"
        "Do NOT modify, repeat, or paraphrase the provided feedback text.\n\n"
        "Output format (must follow exactly):\n"
        "1) Start with EXACTLY: <thinking><feedback>{feedback}</feedback>\n"
        "2) Continue your reasoning, then close: </thinking>\n"
        "3) The final answer MUST be written as: \\boxed{{answer}}\n"
        "Do NOT output anything else.\n\n"
        "Problem:\n"
        "{question}\n"
        "Start your new solution by continuing from the following prefix exactly:"
        "<thinking><feedback>{feedback}</feedback>"
        )
        
        regen_prompts = [
            seq_template.format(question=q, feedback=f)
            for q, f in zip(question_texts, feedback_texts)
        ]

        tokenized = self.tokenizer(
            regen_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.data.get("max_prompt_length"),
            return_tensors="pt",
        )




        device = getattr(failed_feedback_batch, "device", None)
        if device is None:
            for v in failed_feedback_batch.batch.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        failed_feedback_batch.batch["input_ids"] = tokenized["input_ids"].to(device)
        failed_feedback_batch.batch["attention_mask"] = tokenized["attention_mask"].to(device)

        if "position_ids" in failed_feedback_batch.batch:
            failed_feedback_batch.batch["position_ids"] = build_position_ids(
                tokenized["attention_mask"]
            ).to(device)

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

        regen_raw_prompts = [
            [MessageCls(role="user", content=text)]
            for text in regen_prompts
        ]

        failed_feedback_batch.non_tensor_batch = dict(failed_feedback_batch.non_tensor_batch)
        failed_feedback_batch.non_tensor_batch["raw_prompt"] = np.array(
            regen_raw_prompts, dtype=object
        )

        num_samples = len(regen_prompts)
        failed_feedback_batch.non_tensor_batch["tools_kwargs"] = np.array(
            [{} for _ in range(num_samples)],
            dtype=object,
        )

        failed_feedback_batch.non_tensor_batch["regen_prompt"] = np.array(
            regen_prompts, dtype=object
        )
        failed_feedback_batch.non_tensor_batch["regen_question"] = np.array(
            question_texts, dtype=object
        )
        failed_feedback_batch.non_tensor_batch["regen_feedback"] = np.array(
            feedback_texts, dtype=object
        )

        regen_prefix_texts = [
            f"<thinking><feedback>{f}</feedback>"
            for f in feedback_texts
        ]
        failed_feedback_batch.non_tensor_batch["regen_prefix"] = np.array(
            regen_prefix_texts, dtype=object
        )

        prefix_tok = self.tokenizer(
            regen_prefix_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )
        prefix_lens = [len(x) for x in prefix_tok["input_ids"]]
        failed_feedback_batch.batch["regen_prefix_len"] = torch.tensor(
            prefix_lens, dtype=torch.long, device=device
        )
        failed_uids = failed_prompts.meta_info.get("failed_prompt_uids", None)
        if failed_uids is not None:
            if len(failed_uids) != len(question_texts):
                raise ValueError(
                    f"failed_uids len {len(failed_uids)} != question_texts len {len(question_texts)}"
                )
            failed_feedback_batch.non_tensor_batch["uid"] = np.array(
                failed_uids, dtype=object
            )


        return failed_feedback_batch
    

    def _merge_failed_feedbacks_global(
        self,
        feedback_batch: DataProto,
        repeat_times: int,
        failed_prompt_count: int,
    ) -> DataProto:

        import torch
        import numpy as np
        from collections import defaultdict

        batch_size = None
        for v in feedback_batch.batch.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
        if batch_size is None or batch_size == 0:
            return feedback_batch

        non_tensor = getattr(feedback_batch, "non_tensor_batch", {}) or {}
        group_keys_arr = non_tensor.get("feedback_group_keys", None)
        feedback_text_field = non_tensor.get("feedback_text", None)

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

        prompt_to_texts: dict[int, list[str]] = defaultdict(list)
        prompt_to_rep_row: dict[int, int] = {}

        for row_idx, key in enumerate(group_keys):
            if key is None:
                continue

            prompt_idx = key.get("prompt_idx", None)
            prompt_key = int(prompt_idx) if prompt_idx is not None else int(row_idx)

            prompt_to_texts[prompt_key].append(feedback_texts[row_idx])
            if prompt_key not in prompt_to_rep_row:
                prompt_to_rep_row[prompt_key] = row_idx




        num_prompts = len(prompt_to_texts)
        if num_prompts == 0:
            return feedback_batch


        rep_row_indices = [prompt_to_rep_row[k] for k in sorted(prompt_to_rep_row.keys())]
        device = next(iter(feedback_batch.batch.values())).device
        rep_idx_tensor = torch.tensor(rep_row_indices, device=device)

        merged_input = feedback_batch.select_idxs(rep_idx_tensor)

        merged_input.non_tensor_batch = dict(
            getattr(merged_input, "non_tensor_batch", {}) or {}
        )

        merged_feedback_lists: list[list[str]] = [
            prompt_to_texts[k] for k in sorted(prompt_to_texts.keys())
        ]
        merged_input.non_tensor_batch["feedback_list"] = np.array(
            merged_feedback_lists, dtype=object
        )
        

        merged_input.meta_info = dict(getattr(feedback_batch, "meta_info", {}) or {})
        merged_input.meta_info["merge"] = True
        merged_input.meta_info["failed_prompt_count"] = num_prompts
        merged_input.meta_info["repeat_times"] = 1 


        merged_input.non_tensor_batch["merge_prompt_indices"] = np.array(
            sorted(prompt_to_texts.keys()), dtype=object
        )


        try:
            dp_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        except Exception:
            dp_size = 1
        if dp_size <= 1:
            
            merged_output = self.actor_rollout_wg.generate_feedbacks(merged_input)
        else:
            if num_prompts % dp_size == 0:
                merged_padded = merged_input
                num_pad = 0
            else:
                num_pad = dp_size - (num_prompts % dp_size)
                last_idx = num_prompts - 1

                base_indices = torch.arange(num_prompts, device=device)
                pad_indices = torch.full(
                    (num_pad,), last_idx, dtype=torch.long, device=device
                )
                all_indices = torch.cat([base_indices, pad_indices], dim=0)

                merged_padded = merged_input.select_idxs(all_indices)

            merged_output_padded = self.actor_rollout_wg.generate_feedbacks(merged_padded)

            out_device = next(iter(merged_output_padded.batch.values())).device
            keep_indices = torch.arange(num_prompts, device=out_device)
            merged_output = merged_output_padded.select_idxs(keep_indices)

        merged_output.meta_info = dict(getattr(merged_output, "meta_info", {}) or {})
        merged_output.meta_info["repeat_times"] = 1
        merged_output.meta_info["failed_prompt_count"] = num_prompts

        return merged_output


    def _pad_dataproto_to_multiple(self, dp: DataProto, multiple: int) -> DataProto:
        import torch

        size = len(dp)
        if size == 0:
            return dp

        remainder = size % multiple
        if remainder == 0:
            return dp

        pad = multiple - remainder

        idx_all = torch.arange(size, device=next(iter(dp.batch.values())).device)
        pad_idx = idx_all[:pad]  

        new_idx = torch.cat([idx_all, pad_idx], dim=0)
        return dp.select_idxs(new_idx)


    def _merge_failed_with_regen(
        self,
        batch: DataProto,
        failed_batch: DataProto,        
        regen_batch_output: DataProto,
    ) -> DataProto:

        import torch
        import numpy as np
        from tensordict import TensorDict

        size_before = len(batch)             
        regen_size = len(regen_batch_output)  

        if regen_size == 0:
            return batch

        total_size = size_before + regen_size

        device = None
        for v in batch.batch.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        if device is None:
            for v in regen_batch_output.batch.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        base_keys = set(batch.batch.keys())
        regen_keys = set(regen_batch_output.batch.keys())
        all_keys = base_keys | regen_keys

        new_td_data: dict[str, torch.Tensor] = {}

        for key in all_keys:
            base_tensor = batch.batch.get(key, None)
            regen_tensor = regen_batch_output.batch.get(key, None)

            if base_tensor is None and regen_tensor is None:
                continue

            if isinstance(base_tensor, torch.Tensor):
                base_tensor = base_tensor.to(device)
            if isinstance(regen_tensor, torch.Tensor):
                regen_tensor = regen_tensor.to(device)

            if base_tensor is not None and regen_tensor is None:
                tail_shape = base_tensor.shape[1:]
                regen_tensor = torch.zeros(
                    (regen_size, *tail_shape),
                    dtype=base_tensor.dtype,
                    device=device,
                )

            if base_tensor is None and regen_tensor is not None:
                tail_shape = regen_tensor.shape[1:]
                base_tensor = torch.zeros(
                    (size_before, *tail_shape),
                    dtype=regen_tensor.dtype,
                    device=device,
                )

            assert base_tensor is not None and regen_tensor is not None

            if base_tensor.shape[1:] != regen_tensor.shape[1:]:
                raise ValueError(
                    f"[merge_failed_with_regen] Tensor shape mismatch on key '{key}': "
                    f"base {tuple(base_tensor.shape)} vs regen {tuple(regen_tensor.shape)}"
                )

            new_td_data[key] = torch.cat([base_tensor, regen_tensor], dim=0)

        regen_mask = torch.zeros(total_size, dtype=torch.bool, device=device)
        regen_mask[size_before:] = True
        new_td_data["regen_mask"] = regen_mask

        new_td = TensorDict(new_td_data, batch_size=[total_size], device=device)
        batch.batch = new_td

        old_nt = getattr(batch, "non_tensor_batch", None) or {}
        regen_nt = regen_batch_output.non_tensor_batch or {}

        old_keys = set(old_nt.keys())
        regen_keys = set(regen_nt.keys())
        all_nt_keys = old_keys | regen_keys

        indices = failed_batch.meta_info.get("failed_rollout_indices", None)
        if indices is not None:
            indices_np = np.asarray(indices, dtype=int)
            if len(indices_np) != regen_size:
                raise ValueError(
                    f"[merge_failed_with_regen] failed_rollout_indices len {len(indices_np)} "
                    f"!= regen_size {regen_size}"
                )
        else:
            indices_np = None

        new_nt: dict[str, np.ndarray] = {}


        for key in all_nt_keys:
            base_arr = old_nt.get(key, None)
            regen_arr = regen_nt.get(key, None)

            if base_arr is None:
                base_arr_np = np.array([None] * size_before, dtype=object)
            else:
                base_arr_np = np.array(base_arr, dtype=object)
                if len(base_arr_np) != size_before:
                    raise ValueError(
                        f"[merge_failed_with_regen] non_tensor '{key}' len {len(base_arr_np)} "
                        f"!= size_before {size_before}"
                    )

            if regen_arr is None:

                if indices_np is not None and base_arr is not None:
                    regen_arr_np = base_arr_np[indices_np]
                else:
                    regen_arr_np = np.array([None] * regen_size, dtype=object)
            else:
                regen_arr_np = np.array(regen_arr, dtype=object)
                if len(regen_arr_np) != regen_size:
                    raise ValueError(
                        f"[merge_failed_with_regen] regen non_tensor '{key}' len {len(regen_arr_np)} "
                        f"!= regen_size {regen_size}"
                    )

            new_nt[key] = np.concatenate([base_arr_np, regen_arr_np], axis=0)

        batch.non_tensor_batch = new_nt


        return batch

    def _refresh_global_token_num(self, dp: DataProto) -> None:
        """Recompute meta_info['global_token_num'] for a (possibly filtered) DataProto."""
        import torch
        if dp is None or len(dp) == 0:
            return
        attn = dp.batch.get("attention_mask", None)
        if isinstance(attn, torch.Tensor) and attn.dim() >= 2:
            dp.meta_info = dict(getattr(dp, "meta_info", {}) or {})
            dp.meta_info["global_token_num"] = torch.sum(attn, dim=-1).tolist()

    def _concat_two_dataprotos(self, a: DataProto, b: DataProto) -> DataProto:
        """
        Safe concat of two DataProto along batch dim.
        Assumes both are on compatible devices; will move tensors to a's device.
        """
        import numpy as np
        import torch
        from tensordict import TensorDict

        if a is None or len(a) == 0:
            return b
        if b is None or len(b) == 0:
            return a

        # pick device
        device = None
        for v in a.batch.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        if device is None:
            for v in b.batch.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        size_a, size_b = len(a), len(b)
        all_keys = set(a.batch.keys()) | set(b.batch.keys())

        new_td_data: dict[str, torch.Tensor] = {}
        for key in all_keys:
            ta = a.batch.get(key, None)
            tb = b.batch.get(key, None)

            if isinstance(ta, torch.Tensor):
                ta = ta.to(device)
            if isinstance(tb, torch.Tensor):
                tb = tb.to(device)

            if ta is None and tb is None:
                continue

            if ta is None and isinstance(tb, torch.Tensor):
                ta = torch.zeros((size_a, *tb.shape[1:]), dtype=tb.dtype, device=device)
            if tb is None and isinstance(ta, torch.Tensor):
                tb = torch.zeros((size_b, *ta.shape[1:]), dtype=ta.dtype, device=device)

            if not (isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor)):
                continue

            if ta.shape[1:] != tb.shape[1:]:
                raise ValueError(f"[concat] shape mismatch on '{key}': {tuple(ta.shape)} vs {tuple(tb.shape)}")

            new_td_data[key] = torch.cat([ta, tb], dim=0)

        new_td = TensorDict(new_td_data, batch_size=[size_a + size_b], device=device)
        out = DataProto(batch=new_td, non_tensor_batch={}, meta_info={})
        # non-tensor concat
        nt_a = getattr(a, "non_tensor_batch", None) or {}
        nt_b = getattr(b, "non_tensor_batch", None) or {}
        all_nt_keys = set(nt_a.keys()) | set(nt_b.keys())
        new_nt = {}

        import numpy as np

        def _to_obj_1d(v, size: int) -> np.ndarray:
            """Force v into shape [size] object array, one element per sample."""
            if v is None:
                return np.array([None] * size, dtype=object)

            # numpy array
            if isinstance(v, np.ndarray):
                # align first dim to batch
                if v.shape[0] != size:
                    v = np.resize(v, (size,) + v.shape[1:])
                if v.ndim == 1:
                    return v.astype(object, copy=False)
                out = np.empty(size, dtype=object)
                for i in range(size):
                    out[i] = v[i]   # each row/subarray becomes one object
                return out

            # scalar -> broadcast
            if not isinstance(v, (list, tuple)):
                return np.array([v] * size, dtype=object)

            # list/tuple -> enforce length, then pack per-sample
            v = list(v)
            if len(v) < size:
                v = v + [None] * (size - len(v))
            elif len(v) > size:
                v = v[:size]

            out = np.empty(size, dtype=object)
            for i in range(size):
                out[i] = v[i]
            return out

        for key in all_nt_keys:
            va = _to_obj_1d(nt_a.get(key, None), size_a)
            vb = _to_obj_1d(nt_b.get(key, None), size_b)
            new_nt[key] = np.concatenate([va, vb], axis=0)

        out.non_tensor_batch = new_nt
        # # non-tensor concat
        # nt_a = getattr(a, "non_tensor_batch", None) or {}
        # nt_b = getattr(b, "non_tensor_batch", None) or {}
        # all_nt_keys = set(nt_a.keys()) | set(nt_b.keys())
        # new_nt: dict[str, np.ndarray] = {}
        # for key in all_nt_keys:
        #     va = nt_a.get(key, None)
        #     vb = nt_b.get(key, None)

        #     if va is None:
        #         va = np.array([None] * size_a, dtype=object)
        #     else:
        #         va = np.array(va, dtype=object)
        #         if len(va) != size_a:
        #             va = np.resize(va, size_a)

        #     if vb is None:
        #         vb = np.array([None] * size_b, dtype=object)
        #     else:
        #         vb = np.array(vb, dtype=object)
        #         if len(vb) != size_b:
        #             vb = np.resize(vb, size_b)

        #     new_nt[key] = np.concatenate([va, vb], axis=0)

        # out.non_tensor_batch = new_nt

        # meta_info: keep minimal keys, callers should refresh if needed
        out.meta_info = dict(getattr(a, "meta_info", {}) or {})
        return out


    def _build_dpo_pair_batch(
        self,
        batch: DataProto,
        chosen_idx: "torch.Tensor",
        rejected_idx: "torch.Tensor",
        metrics: dict,
    ) -> Optional[DataProto]:
        """
        Build a DPO training batch from aligned chosen/rejected indices.

        Output layout (2*K rows), interleaved for stable micro-batching:
          - rows: [c0, r0, c1, r1, ..., c{K-1}, r{K-1}]
          - batch.batch["dpo_is_chosen"]: [2K] bool (True for chosen c_i, False for rejected r_i)
          - batch.batch["dpo_pair_id"]:   [2K] long in [0..K-1], repeated twice per pair
          - batch.meta_info["loss_type"] = "dpo"
        """
        import torch

        if chosen_idx.numel() == 0:
            return None
        if chosen_idx.numel() != rejected_idx.numel():
            raise ValueError(f"[DPO] chosen/rejected size mismatch: {chosen_idx.numel()} vs {rejected_idx.numel()}")

        # Interleave chosen/rejected rows to keep each pair inside the same micro-batch split.
        # Layout: [c0, r0, c1, r1, ..., c{K-1}, r{K-1}]
        interleave_idx = torch.stack([chosen_idx, rejected_idx], dim=1).reshape(-1)  # [2K]
        dpo_dp = batch.select_idxs(interleave_idx)

        K = int(chosen_idx.numel())
        device = next(iter(dpo_dp.batch.values())).device

        # is_chosen: [True, False, True, False, ...]
        dpo_dp.batch["dpo_is_chosen"] = (torch.arange(2 * K, device=device) % 2 == 0)

        # pair_id: [0, 0, 1, 1, 2, 2, ...]
        dpo_dp.batch["dpo_pair_id"] = torch.arange(K, device=device, dtype=torch.long).repeat_interleave(2)

        
        # Placeholder for mixed-loss actor: DPO rows don't use seq_level_rewards, but mixed routing
        # expects the tensor to exist for the whole batch.
        if "seq_level_rewards" not in dpo_dp.batch:
            dpo_dp.batch["seq_level_rewards"] = torch.zeros((2 * K,), device=device, dtype=torch.float32)
        dpo_dp.meta_info = dict(getattr(dpo_dp, "meta_info", {}) or {})
        dpo_dp.meta_info["loss_type"] = "dpo"
        dpo_dp.meta_info["dpo_pairs"] = K

        # refresh token count for flops/metrics
        self._refresh_global_token_num(dpo_dp)

        metrics["feedback/dpo_pair_cnt"] = K
        metrics["feedback/dpo_sample_cnt"] = int(2 * K)
        return dpo_dp


    def _prepare_grpo_and_dpo_batches(
        self,
        batch: DataProto,
        n_rollout: int,
        norm_adv_by_std_in_grpo: bool,
        metrics: dict,
    ) -> tuple[DataProto, Optional[DataProto]]:
        """
        Scheme-1 (per-uid loss routing, NO advantage injection):

        - base-only uid: if not all-negative among its n rollouts -> GRPO on its n responses
        - regen uid: look at regen's n rewards
            * all 1 -> DPO (regen as chosen vs base as rejected, 1-1 pairing)
            * all 0 -> drop
            * mixed -> GRPO on regen only

        Returns:
            grpo_batch: DataProto for GRPO update (has advantages/returns)
            dpo_batch : DataProto for DPO update (has dpo_pair_id/dpo_is_chosen; NO advantages needed)
        """
        import numpy as np
        import torch

        if metrics is None:
            metrics = {}

        # --------- 1) scalar reward for routing ---------
        rm = batch.batch.get("response_mask", None)
        if rm is None:
            raise ValueError("response_mask is required before mixed-loss routing.")
        rm_bool = rm.bool()

        tl_score = batch.batch.get("token_level_scores", None)
        if tl_score is None:
            tl_score = batch.batch.get("token_level_rewards", None)
        if tl_score is None:
            raise ValueError("token_level_scores/token_level_rewards missing; cannot route mixed losses.")

        total_reward = (tl_score * rm_bool.to(dtype=tl_score.dtype)).sum(dim=-1)
        is_pos = total_reward > 0

        regen_mask = batch.batch.get("regen_mask", None)
        if regen_mask is None:
            regen_mask = torch.zeros((len(batch),), device=rm.device, dtype=torch.bool)
        else:
            regen_mask = regen_mask.to(device=rm.device, dtype=torch.bool)

        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            raise ValueError("non_tensor_batch['uid'] is required for per-uid loss routing.")
        if isinstance(uids, np.ndarray):
            uids = uids.tolist()

        # --------- 2) group by uid ---------
        uid_to_base: dict[str, list[int]] = {}
        uid_to_regen: dict[str, list[int]] = {}
        for i, uid in enumerate(uids):
            if regen_mask[i].item():
                uid_to_regen.setdefault(uid, []).append(i)
            else:
                uid_to_base.setdefault(uid, []).append(i)

        grpo_idx_list = []
        chosen_idx_list = []
        rejected_idx_list = []
        dropped_uid_cnt = 0
        dpo_uid_cnt = 0
        regen_all1_uid_cnt = 0
        regen_all0_uid_cnt = 0
        regen_mixed_uid_cnt = 0

        for uid, base_list in uid_to_base.items():
            regen_list = uid_to_regen.get(uid, [])
            base_t = torch.tensor(base_list, device=rm.device, dtype=torch.long)

            if len(regen_list) == 0:
                # base-only
                if is_pos[base_t].any().item():
                    grpo_idx_list.append(base_t)
                else:
                    dropped_uid_cnt += 1
                continue

            # has regen
            regen_t = torch.tensor(regen_list, device=rm.device, dtype=torch.long)
            regen_pos = is_pos[regen_t]

            if regen_pos.all().item():
                # DPO
                regen_all1_uid_cnt += 1
                dpo_uid_cnt += 1
                k = min(len(base_list), len(regen_list))
                if k <= 0:
                    dropped_uid_cnt += 1
                    continue
                # 1-1 pairing (stable): first k
                chosen_idx_list.append(regen_t[:k])
                rejected_idx_list.append(base_t[:k])
                continue

            if (~regen_pos).all().item():
                regen_all0_uid_cnt += 1
                dropped_uid_cnt += 1
                continue

            # mixed -> GRPO on regen only
            regen_mixed_uid_cnt += 1
            grpo_idx_list.append(regen_t)

        def _cat_or_empty(xs):
            if not xs:
                return torch.empty((0,), device=rm.device, dtype=torch.long)
            return torch.cat(xs, dim=0)

        grpo_idx = _cat_or_empty(grpo_idx_list)
        chosen_idx = _cat_or_empty(chosen_idx_list)
        rejected_idx = _cat_or_empty(rejected_idx_list)

        metrics["feedback/grpo_sample_cnt"] = int(grpo_idx.numel())
        metrics["feedback/dropped_uid_cnt"] = int(dropped_uid_cnt)
        metrics["feedback/dpo_uid_cnt"] = int(dpo_uid_cnt)
        metrics["feedback/regen_all1_uid_cnt"] = int(regen_all1_uid_cnt)
        metrics["feedback/regen_all0_uid_cnt"] = int(regen_all0_uid_cnt)
        metrics["feedback/regen_mixed_uid_cnt"] = int(regen_mixed_uid_cnt)

        # --------- 3) build GRPO batch ---------
        if grpo_idx.numel() == 0:
            grpo_batch = batch.select_idxs(grpo_idx)  # empty
        else:
            grpo_batch = batch.select_idxs(grpo_idx)
            grpo_batch = compute_advantage(
                grpo_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=n_rollout,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
            self._refresh_global_token_num(grpo_batch)

        # --------- 4) build DPO batch ---------
        dpo_batch = self._build_dpo_pair_batch(
            batch=batch,
            chosen_idx=chosen_idx,
            rejected_idx=rejected_idx,
            metrics=metrics,
        )

        return grpo_batch, dpo_batch

    def _concat_dataproto_list(self, dps: list[DataProto]) -> Optional[DataProto]:
        out = None
        for dp in dps:
            if dp is None or len(dp) == 0:
                continue
            out = dp if out is None else self._concat_two_dataprotos(out, dp)
        return out

    def _sum_token_reward(self, reward_tensor: torch.Tensor, response_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # reward_tensor: [B, T]
        if response_mask is None:
            return reward_tensor.sum(dim=-1)
        return (reward_tensor * response_mask.to(dtype=reward_tensor.dtype)).sum(dim=-1)

    def _classify_prompts(self, total_reward_row: torch.Tensor, n_rollout: int):
        """
        total_reward_row: [P*n]  (P prompts, each has n_rollout rollouts, contiguous)
        return (all1_mask, all0_mask, mixed_mask) each: [P]
        """
        assert total_reward_row.numel() % n_rollout == 0, (
            f"batch_size {total_reward_row.numel()} must be divisible by n_rollout={n_rollout}"
        )
        m = total_reward_row.view(-1, n_rollout)  # [P, n]
        all1 = (m > 0).all(dim=1)
        all0 = (m <= 0).all(dim=1)
        mixed = ~(all1 | all0)
        return all1, all0, mixed

    def _prompt_mask_to_flat_idx(self, prompt_mask: torch.Tensor, n_rollout: int, device: torch.device) -> torch.Tensor:
        """prompt_mask: [P] -> flat idx [K*n] over a (P*n) batch in prompt-major contiguous layout."""
        pidx = torch.nonzero(prompt_mask, as_tuple=False).squeeze(-1)
        if pidx.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        ar = torch.arange(n_rollout, device=device).unsqueeze(0)  # [1, n]
        return (pidx.unsqueeze(1) * n_rollout + ar).reshape(-1)

    def _apply_regen_prefix_mask_inplace(self, dp: DataProto) -> None:
        if "regen_prefix_len" not in dp.batch:
            return
        if "response_mask" not in dp.batch:
            dp.batch["response_mask"] = compute_response_mask(dp)

        rm0 = dp.batch["response_mask"]
        rm = rm0.bool()

        T = rm.size(1)
        k = dp.batch["regen_prefix_len"].clamp(min=0, max=T)  # [B]
        
        pos = torch.arange(T, device=rm.device).unsqueeze(0)  # [1,T]
        rm = rm & (pos >= k.unsqueeze(1))

        dp.batch["response_mask"] = rm.to(dtype=rm0.dtype)

    def _collect_multiturn_regen_batches(
        self,
        base_batch: DataProto,
        base_reward_tensor: torch.Tensor,
        n_rollout: int,
        timing_raw: dict,
    ) -> tuple[Optional[DataProto], Optional[DataProto], dict]:

        metrics: dict = {}

        if "response_mask" not in base_batch.batch:
            base_batch.batch["response_mask"] = compute_response_mask(base_batch)
        base_batch.batch["token_level_scores"] = base_reward_tensor

        device = base_batch.batch["response_mask"].device
        total_reward_row = self._sum_token_reward(base_reward_tensor, base_batch.batch["response_mask"])
        base_all1, base_all0, base_mixed = self._classify_prompts(total_reward_row, n_rollout)

        metrics["regen/base_prompt_cnt"] = int(base_all0.numel())
        metrics["regen/base_all0_prompt_cnt"] = int(base_all0.sum().item())
        metrics["regen/base_non_all0_prompt_cnt"] = int((~base_all0).sum().item())

        grpo_parts: list[DataProto] = []
        chosen_parts: list[DataProto] = []
        rejected_parts: list[DataProto] = []

        keep_base_prompt_mask = ~base_all0
        keep_base_flat = self._prompt_mask_to_flat_idx(keep_base_prompt_mask, n_rollout, device=device)
        if keep_base_flat.numel() > 0:
            grpo_parts.append(base_batch.select_idxs(keep_base_flat))

        max_turn = self.config.algorithm.get("max_regen_turns", 2)
        if (not self.need_feedback) or max_turn <= 0 or base_all0.sum().item() == 0:
            return self._concat_dataproto_list(grpo_parts), None, metrics

        active_base_prompt_idx = torch.nonzero(base_all0, as_tuple=False).squeeze(-1)  # [F]
        active_flat = (active_base_prompt_idx.unsqueeze(1) * n_rollout + torch.arange(n_rollout, device=device)).reshape(-1)
        current_dp = base_batch.select_idxs(active_flat)
        current_reward = base_reward_tensor.index_select(0, active_flat.to(base_reward_tensor.device))

        for turn in range(1, max_turn + 1):
            if len(current_dp) == 0:
                break
            prompt_cnt = len(current_dp) // n_rollout
            metrics[f"regen/turn{turn}_active_prompt_cnt"] = int(prompt_cnt)

            failed_batch, failed_prompts, m = self._collect_all_negative_rollout_prompts_by_group(
                current_dp, current_reward, n_rollout
            )
            for k, v in (m or {}).items():
                metrics[f"regen/turn{turn}/{k}"] = v

            if failed_batch is None or failed_prompts is None or len(failed_batch) == 0:
                break

            with marked_timer(f"regen_turn{turn}_feedback", timing_raw, color="red"):
                fb_in = failed_batch
                if not self.async_rollout_mode:
                    fb_out = self.actor_rollout_wg.generate_feedbacks(fb_in)
                else:
                    fb_out = self.async_rollout_manager.generate_feedbacks(fb_in)

                timing_raw.update(fb_out.meta_info.get("timing", {}))
                fb_out.meta_info.pop("timing", None)

                fb_merged = self._merge_failed_feedbacks_global(
                    fb_out,
                    repeat_times=int(failed_batch.meta_info.get("repeat_times", n_rollout)),
                    failed_prompt_count=int(failed_batch.meta_info.get("failed_prompt_count", prompt_cnt)),
                )

            with marked_timer(f"regen_turn{turn}_gen", timing_raw, color="red"):
                regen_input = self._build_regen_batch(failed_prompts=failed_prompts, failed_feedback_batch=fb_merged)
                regen_input = regen_input.repeat(repeat_times=n_rollout, interleave=True)  # [F*n]

                dp_size = getattr(self.actor_rollout_wg, "dp_size", None)
                if dp_size is None:
                    dp_size = int(self.config.trainer.get("n_gpus_per_node", 1) or 1)
                regen_input = self._pad_dataproto_to_multiple(regen_input, multiple=int(dp_size))
                regen_input_unpadded = regen_input 

                if not self.async_rollout_mode:
                    regen_out_padded = self.actor_rollout_wg.generate_sequences(regen_input)
                else:
                    regen_out_padded = self.async_rollout_manager.generate_sequences(regen_input)

                timing_raw.update(regen_out_padded.meta_info.get("timing", {}))
                regen_out_padded.meta_info.pop("timing", None)

                orig_len = len(failed_batch)  # F*n
                if len(regen_out_padded) > orig_len:
                    idx = torch.arange(orig_len, device=next(iter(regen_out_padded.batch.values())).device)
                    regen_out = regen_out_padded.select_idxs(idx)
                else:
                    regen_out = regen_out_padded

                idx2 = torch.arange(orig_len, device=next(iter(regen_out.batch.values())).device)
                regen_in_nt = regen_input_unpadded.select_idxs(idx2)
                out_nt = dict(getattr(regen_in_nt, "non_tensor_batch", {}) or {})
                out_nt.update(dict(getattr(regen_out, "non_tensor_batch", {}) or {}))  # 若输出侧也写了非tensor字段，保留
                regen_out.non_tensor_batch = out_nt

                if "regen_prefix_len" in regen_input.batch:
                    regen_out.batch["regen_prefix_len"] = regen_input.batch["regen_prefix_len"][:orig_len].to(
                        regen_out.batch["responses"].device
                    )
                rm0 = regen_out.batch["response_mask"].clone()
                responses = regen_out.batch.get("responses", None)
                # response_mask & mask prefix
                regen_out.batch["response_mask"] = compute_response_mask(regen_out)

                self._apply_regen_prefix_mask_inplace(regen_out)
                rm1 = regen_out.batch["response_mask"]

                try:
                    if responses is None:
                        print("[regen_mask_debug] no 'responses' in regen_out.batch, skip printing.")
                    else:
                        N = responses.size(0)
                        k_print = min(5, N)

                        if "regen_mask" in regen_out.batch:
                            regen_mask = regen_out.batch["regen_mask"].bool()
                            cand = torch.nonzero(regen_mask, as_tuple=False).squeeze(-1)
                            if cand.numel() == 0:
                                cand = torch.arange(N, device=responses.device)
                        else:
                            cand = torch.arange(N, device=responses.device)

                        pick = cand[:k_print]

                        for t, i in enumerate(pick.tolist()):
                            ids_before = responses[i][rm0[i].bool()].detach().cpu().tolist()
                            ids_after  = responses[i][rm1[i].bool()].detach().cpu().tolist()

                            txt_before = self.tokenizer.decode(ids_before, skip_special_tokens=False)
                            txt_after  = self.tokenizer.decode(ids_after,  skip_special_tokens=False)

                            uid = None
                            try:
                                uid_list = regen_out.non_tensor_batch.get("uid", None)
                                uid = uid_list[i] if uid_list is not None and i < len(uid_list) else None
                            except Exception:
                                uid = None

                            prefix_len = None
                            try:
                                if "regen_prefix_len" in regen_out.batch:
                                    prefix_len = int(regen_out.batch["regen_prefix_len"][i].item())
                            except Exception:
                                prefix_len = None

                            is_regen = None
                            try:
                                if "regen_mask" in regen_out.batch:
                                    is_regen = bool(regen_out.batch["regen_mask"][i].item())
                            except Exception:
                                is_regen = None

                            print("\n" + "=" * 120)
                            print(f"[regen_mask_debug] sample#{t} i={i} uid={uid} is_regen={is_regen} "
                                f"prefix_len={prefix_len} before_tokens={len(ids_before)} after_tokens={len(ids_after)}")
                            print("-" * 120)
                            print("[BEFORE(prefix-mask)]")
                            print(txt_before)
                            print("-" * 120)
                            print("[AFTER(prefix-mask)]")
                            print(txt_after)
                            print("=" * 120 + "\n")

                except Exception as e:
                    print("[regen_mask_debug] failed:", repr(e))
                
            # 3) reward
            with marked_timer(f"regen_turn{turn}_reward", timing_raw, color="yellow"):
                reward_regen, _ = compute_reward(regen_out, self.reward_fn)
                regen_out.batch["token_level_scores"] = reward_regen

            total_reward_row = self._sum_token_reward(reward_regen, regen_out.batch.get("response_mask", None))
            r_all1, r_all0, r_mixed = self._classify_prompts(total_reward_row, n_rollout)

            metrics[f"regen/turn{turn}_all1_prompt_cnt"] = int(r_all1.sum().item())
            metrics[f"regen/turn{turn}_mixed_prompt_cnt"] = int(r_mixed.sum().item())
            metrics[f"regen/turn{turn}_all0_prompt_cnt"] = int(r_all0.sum().item())

            if r_all1.any().item():
                flat_chosen = self._prompt_mask_to_flat_idx(r_all1, n_rollout, device=device)
                chosen_parts.append(regen_out.select_idxs(flat_chosen))

                sel_base_prompt_idx = active_base_prompt_idx[r_all1]  # prompt idx in base
                flat_rejected = (sel_base_prompt_idx.unsqueeze(1) * n_rollout + torch.arange(n_rollout, device=device)).reshape(-1)
                rejected_parts.append(base_batch.select_idxs(flat_rejected))

            if r_mixed.any().item():
                flat_grpo = self._prompt_mask_to_flat_idx(r_mixed, n_rollout, device=device)
                grpo_parts.append(regen_out.select_idxs(flat_grpo))

            if r_all0.any().item():
                flat_continue = self._prompt_mask_to_flat_idx(r_all0, n_rollout, device=device)
                current_dp = regen_out.select_idxs(flat_continue)
                current_reward = reward_regen.index_select(0, flat_continue.to(reward_regen.device))
                active_base_prompt_idx = active_base_prompt_idx[r_all0]
            else:
                current_dp = regen_out.select_idxs(torch.empty((0,), dtype=torch.long, device=device))
                break

        metrics["regen/after_max_turn_all0_prompt_cnt"] = int(active_base_prompt_idx.numel())

        grpo_batch_pre = self._concat_dataproto_list(grpo_parts)

        chosen_all = self._concat_dataproto_list(chosen_parts)
        rejected_all = self._concat_dataproto_list(rejected_parts)

        dpo_batch_pre = None
        if chosen_all is not None and rejected_all is not None and len(chosen_all) > 0:
            pool = self._concat_two_dataprotos(chosen_all, rejected_all)
            K = len(chosen_all)
            dev = next(iter(pool.batch.values())).device
            chosen_idx = torch.arange(K, device=dev, dtype=torch.long)
            rejected_idx = torch.arange(K, device=dev, dtype=torch.long) + K
            dpo_batch_pre = self._build_dpo_pair_batch(pool, chosen_idx, rejected_idx, metrics)

        return grpo_batch_pre, dpo_batch_pre, metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        import numpy as np
        import torch
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
                orig_reward_tensor = None

                skip_updates = False

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
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(batch, metrics=metrics)

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
                        
                    n_rollout = int(self.config.actor_rollout_ref.rollout.n)
                    grpo_pre, dpo_pre, regen_metrics = self._collect_multiturn_regen_batches(
                        base_batch=batch,
                        base_reward_tensor=reward_tensor,
                        n_rollout=n_rollout,
                        timing_raw=timing_raw,
                    )
                    metrics.update(regen_metrics)

                    has_grpo = (grpo_pre is not None and len(grpo_pre) > 0)
                    has_dpo = (dpo_pre is not None and len(dpo_pre) > 0)

                    batch_for_critic = grpo_pre  

                    actor_batch = None
                    dpo_size = 0
                    grpo_size = 0

                    if has_dpo:
                        actor_batch = dpo_pre
                        dpo_size = len(dpo_pre)

                    if has_grpo:
                        device = next(iter(grpo_pre.batch.values())).device
                        grpo_pre.batch["dpo_pair_id"] = torch.full((len(grpo_pre),), -1, dtype=torch.long, device=device)
                        grpo_pre.batch["dpo_is_chosen"] = torch.zeros((len(grpo_pre),), dtype=torch.bool, device=device)
                        actor_batch = grpo_pre if actor_batch is None else self._concat_two_dataprotos(actor_batch, grpo_pre)
                        grpo_size = len(grpo_pre)

                    if actor_batch is None or len(actor_batch) == 0:
                        metrics["feedback/empty_update_batch"] = 1
                        skip_updates = True
                    else:
                        actor_batch.meta_info = dict(getattr(actor_batch, "meta_info", {}) or {})
                        actor_batch.meta_info["_dpo_size"] = int(dpo_size)
                        actor_batch.meta_info["_grpo_size"] = int(grpo_size)

                        if has_dpo and has_grpo:
                            actor_batch.meta_info["loss_type"] = "mixed"
                        elif has_dpo:
                            actor_batch.meta_info["loss_type"] = "dpo"
                        else:
                            actor_batch.meta_info["loss_type"] = "grpo"

                        if has_dpo:
                            actor_batch.meta_info["dpo_beta"] = float(self.config.algorithm.get("dpo_beta", 0.1))
                            actor_batch.meta_info["dpo_lambda"] = float(self.config.algorithm.get("dpo_lambda", 1.0))

                        self._refresh_global_token_num(actor_batch)
                        batch = actor_batch

                    
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
                    if (not skip_updates) and self.use_critic and (batch_for_critic is not None) and (len(batch_for_critic) > 0):
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        metrics.update(is_metrics)

                        dpo_size = int(batch.meta_info.get("_dpo_size", 0) or 0)
                        grpo_size = int(batch.meta_info.get("_grpo_size", 0) or 0)

                        device = next(iter(batch.batch.values())).device

                        dpo_part = None
                        grpo_part = None

                        if dpo_size > 0:
                            dpo_idx = torch.arange(dpo_size, device=device, dtype=torch.long)
                            dpo_part = batch.select_idxs(dpo_idx)

                        if grpo_size > 0:
                            grpo_idx = torch.arange(dpo_size, dpo_size + grpo_size, device=device, dtype=torch.long)
                            grpo_part = batch.select_idxs(grpo_idx)

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            grpo_part = compute_advantage(
                                grpo_part,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=n_rollout,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                            self._refresh_global_token_num(grpo_part)

                            grpo_part.batch["dpo_pair_id"] = torch.full((len(grpo_part),), -1, dtype=torch.long, device=device)
                            grpo_part.batch["dpo_is_chosen"] = torch.zeros((len(grpo_part),), dtype=torch.bool, device=device)

                        batch_for_critic = grpo_part

                        # rebuild actor batch
                        actor_batch = None
                        if dpo_part is not None and len(dpo_part) > 0:
                            actor_batch = dpo_part
                        if grpo_part is not None and len(grpo_part) > 0:
                            actor_batch = grpo_part if actor_batch is None else self._concat_two_dataprotos(actor_batch, grpo_part)

                        if actor_batch is None or len(actor_batch) == 0:
                            metrics["feedback/empty_update_batch"] = 1
                            skip_updates = True
                        else:
                            actor_batch.meta_info = dict(getattr(batch.meta_info, "copy", lambda: {})() or batch.meta_info or {})
                            self._refresh_global_token_num(actor_batch)
                            batch = actor_batch
                    print("IF SKIP UPDATES:", skip_updates)
                    print("LOSS TYPE:", batch.meta_info.get("loss_type", "N/A"))

                    # update critic
                    if (not skip_updates) and self.use_critic and (batch_for_critic is not None) and (len(batch_for_critic) > 0):
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch_for_critic)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if (not skip_updates) and (self.config.trainer.critic_warmup <= self.global_steps):
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
                    
                    # raise ValueError(
                    #     f"debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    # )
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
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # metrics.update(compute_data_metrics(batch=batch_for_critic , use_critic=self.use_critic))

                use_critic_metrics = bool(self.use_critic and (batch_for_critic is not None and len(batch_for_critic) > 0))
                metrics.update(compute_data_metrics(batch=batch_for_critic, use_critic=use_critic_metrics))


                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                
                if skip_updates:
                    metrics["feedback/skip_updates"] = 1

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
