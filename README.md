---
title: PyTorch Debug Env
emoji: 🔥
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
short_description: RL environment for diagnosing broken PyTorch training jobs
tags:
  - openenv
  - pytorch
  - reinforcement-learning
  - debugging
  - ml-training
  - agent
pinned: true
---

# PyTorch Debug Env 🔥

A complete [OpenEnv](https://meta-pytorch.org/OpenEnv/) environment for the **Meta PyTorch Hackathon** where an AI agent investigates and diagnoses broken PyTorch training jobs.

## Quick Start

```python
from openenv import AutoEnv, AutoAction

env = AutoEnv.from_env("ArchCoder/pytorch-debug-env")
Action = AutoAction.from_env("ArchCoder/pytorch-debug-env")

with env.sync() as client:
    result = client.reset(task_id="easy")
    action = Action(
        current_hypothesis={
            "bug_type": "missing_zero_grad",
            "affected_file": "train.py",
            "confidence": 0.7
        },
        commit_diagnosis=False
    )
    step_result = client.step(action)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/reset?task_id=easy` | POST | Start new episode |
| `/step` | POST | Submit hypothesis + action |
| `/state` | GET | Current episode state |

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy` | ⭐ | Single-file bug — missing `zero_grad`, wrong loss |
| `medium` | ⭐⭐ | Multi-file root cause — data leakage, learning-rate misconfig |
| `hard` | ⭐⭐⭐ | Silent failure — memory leak, AMP overflow, red herrings |

Each difficulty draws from multiple bug templates, so repeated runs do not recycle the same exact failure.

## Reward Structure

- **Hypothesis delta** (60%) — reward for improving your bug hypothesis each step
- **Investigation** (20%) — reward for inspecting the right files
- **Final diagnosis** (20%) — accuracy of committed diagnosis vs ground truth

Scores range from `0.0` to `1.0`. Partial credit for correct bug category on hard tasks.

## Investigation Actions

- `reveal_file`: reveal a file from the synthetic repo
- `extend_loss_curve`: reveal more loss-curve points
- `extend_gpu_profile`: reveal more GPU profile points
- `reveal_log_chunk`: append additional training log lines
- `run_diagnostic`: expose a diagnostic summary report

## Reproducibility

Use `SEED` to make scenario selection and artifacts deterministic across runs:

```bash
set SEED=42
python inference.py
```

## Baseline Scores

Run `inference.py` with a fixed `SEED` to record your baseline scores. The script prints per-task `[END]` lines with the final rewards.

Example template (fill after running):

| Model | Seed | Easy | Medium | Hard |
|------|------|------|--------|------|
| gpt-3.5-turbo | 42 | 0.xx | 0.xx | 0.xx |

## Environment State

Each episode provides a synthetic PyTorch repo with:
- Source files (`train.py`, `model/`, `data/`, `config/`)
- Loss curves and GPU memory profiles
- Training logs with realistic noise and red herrings

The agent reveals files progressively across up to 5–6 steps, refining its hypothesis before committing a final diagnosis.

## Author

**Priyansh Saxena** — IIIT Gwalior