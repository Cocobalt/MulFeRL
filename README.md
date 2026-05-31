# MulFeRL

This repository contains the anonymous implementation of **MulFeRL**,
**Mul**ti-turn **Fe**edback-guided **R**einforcement **L**earning.

MulFeRL extends RLVR with three components:

- **Event-triggered progress induction:** uniformly failed rollout groups trigger
  verbal feedback and feedback-guided regeneration.
- **Progress credit assignment:** contrastive regenerated groups are optimized
  with GRPO, while feedback-induced all-solved transitions are optimized with
  Feedback-Contrastive Optimization (FCO).
- **Structured feedback injection:** issue-and-fix feedback is inserted into a
  fixed reasoning slot so the policy can condition on feedback consistently.

The implementation is built on top of the open-source `verl` training framework.
Reward functions, benchmark evaluators, and evaluation scripts are intentionally
omitted from this anonymous release. To run training, provide your own verifier
through `custom_reward_function.path`.

## Environment

```bash
conda env create -f environment.yaml
conda activate <ENV_NAME>
```

If the environment already exists:

```bash
conda env update -f environment.yaml --prune
```

## Data

Training data should be stored in parquet format compatible with `verl` RL
datasets. The preprocessing helper rewrites prompts into the structured MulFeRL
format and preserves the raw question in `extra_info["question_raw"]`:

```bash
python verl-main/examples/data_preprocess/data.py \
  --input_parquet <RAW_TRAIN_PARQUET> \
  --output_parquet <TRAIN_PARQUET> \
  --output_jsonl <TRAIN_JSONL>
```

## Feedback Provider

MulFeRL uses an external feedback provider during training. Configure it through
environment variables:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export OPENAI_API_BASE=<OPTIONAL_API_BASE>
export MULFERL_FEEDBACK_MODEL=gpt-4o
```

The feedback tool only returns verbal feedback. It does not compute rewards.

## Training

Edit placeholders in the launch script, then run:

```bash
bash verl-main/examples/sglang_multiturn/MulFeRL/mulferl.sh
```

Important placeholders:

- `<MODEL_PATH>`: base model checkpoint.
- `<TRAIN_DATA_PATH>`: preprocessed training parquet.
- `<VAL_DATA_PATH>`: validation parquet used by your local verifier.
- `<CUSTOM_REWARD_SCRIPT>`: your private verifier/reward function. This file is
  not included in the anonymous release.

The default launch script uses:

- rollout group size `K=8`,
- up to two feedback-guided regeneration turns,
- GRPO for within-state contrast,
- FCO with `lambda_FCO=0.01`, `tau=0.005`, and margin `0.0`.

## Key Paths

| Path | Purpose |
|---|---|
| `verl-main/examples/sglang_multiturn/MulFeRL/mulferl.sh` | Training launch script |
| `verl-main/examples/sglang_multiturn/config/mulferl.yaml` | MulFeRL Hydra config |
| `verl-main/examples/sglang_multiturn/config/tool_config/mulferl.yaml` | Feedback tool config |
| `verl-main/verl/trainer/ppo/ray_trainer.py` | Progress induction and credit assignment |
| `verl-main/verl/workers/actor/dp_actor.py` | GRPO/FCO actor update |
| `verl-main/verl/tools/mulferl_tool_feedback_group.py` | Verbal feedback provider interface |
| `verl-main/examples/data_preprocess/data.py` | Structured-format data preprocessing |

## Citation

The paper citation will be added after the review process.

If you use the underlying training framework, please cite `verl` / HybridFlow:

```bibtex
@article{sheng2024hybridflow,
  title={HybridFlow: A Flexible and Efficient RLHF Framework},
  author={Sheng, Guangming and Zhang, Chi and Ye, Zilingfeng and Wu, Xibin and Zhang, Wang and Zhang, Ru and Peng, Yanghua and Lin, Haibin and Wu, Chuan},
  journal={arXiv preprint arXiv:2409.19256},
  year={2024}
}
```
