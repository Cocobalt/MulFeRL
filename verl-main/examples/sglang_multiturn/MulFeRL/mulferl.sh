#!/bin/bash
# Run on 8xH100 (anonymized for submission)
# Make sure your current working directory is the root of the project

set -x

ulimit -n 65535

# Set PROJECT_DIR to the root of your project (e.g., absolute path or $PWD)
# This script assumes PROJECT_DIR is set correctly before execution.
PROJECT_DIR=""  # ← Please set this to your project root before running

CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

export TORCH_CUDA_ARCH_LIST="8.0"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mulferl' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="<MODEL_PATH>" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.over_sample_rate=0.0 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.feedback=True \
    custom_reward_function.path="<CUSTOM_REWARD_SCRIPT>" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    +algorithm.regen_adv_scale=0.5 \
    +algorithm.max_regen_turns=2 \
    +algorithm.dpo_lambda=0.01 \
    +algorithm.dpo_beta=0.005 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='anonymous_project' \
    trainer.experiment_name='anonymous_exp' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="<VALID_DATA_DIR>" \
    data.train_files="<TRAIN_DATA_PATH>" \
    data.val_files="<VAL_DATA_PATH>" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/mulferl.yaml" \
    trainer.total_epochs=5 $@