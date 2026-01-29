# Multi-Turn RLHF with Structured Feedback (MulFeRL)

This repository contains the implementation of **MulFeRL**, a method for **multi-turn RLHF/RLVR training with structured natural-language feedback**.

**Implementation note:** The code is developed **on top of the open-source `verl` RL training framework** (Volcano Engine Reinforcement Learning for LLMs; the open-source release of *HybridFlow*). See **Citation** below.

---

## What’s included

- MulFeRL training pipeline (trainer/workers/tools).
- Experiment configs aligned with the paper’s hyperparameters.

---

## Quickstart

### 1) Environment setup

Create the conda environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate <ENV_NAME>
```

If you need to update the environment after changes:

```bash
conda env update -f environment.yaml --prune
```

---

### 2) Prepare training data

Preprocess / convert your **training** dataset into the expected parquet format via:

```bash
python verl-main/examples/data_preprocess/data.py
```

You can adapt this script to map your raw dataset into the required fields used by the training loop.

---

### 3) Run training

Launch MulFeRL training with:

```bash
bash verl-main/examples/sglang_multiturn/MulFeRL/mulferl.sh
```
---

## Repository structure (key paths)

| Path | Notes |
|---|---|
| `verl/trainer/` | Trainer logic for MulFeRL |
| `verl/workers/` | Worker logic for MulFeRL updates |
| `verl/tools/` | Interfaces/utilities for interacting with an external feedback provider |
| `verl-main/examples/sglang_multiturn/config/` | Experiment configs |
| `verl-main/examples/data_preprocess/data.py` | Training data preprocessing entry |

---

## Citation

### Cite `verl` / HybridFlow

If you use this codebase (built on `verl`), please cite **HybridFlow**:

```bibtex
@article{sheng2024hybridflow,
  title={HybridFlow: A Flexible and Efficient RLHF Framework},
  author={Sheng, Guangming and Zhang, Chi and Ye, Zilingfeng and Wu, Xibin and Zhang, Wang and Zhang, Ru and Peng, Yanghua and Lin, Haibin and Wu, Chuan},
  journal={arXiv preprint arXiv:2409.19256},
  year={2024}
}
```

Optionally, you may also cite the `verl` repository:

```bibtex
@misc{verl_github,
  title        = {verl: Volcano Engine Reinforcement Learning for LLMs},
  howpublished = {\url{https://github.com/volcengine/verl}},
  note         = {Open-source implementation of HybridFlow},
  year         = {2025}
}
```

### Cite MulFeRL

A bibtex entry for MulFeRL will be provided with the final paper/release.
