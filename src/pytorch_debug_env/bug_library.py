# src/pytorch_debug_env/bug_library.py
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np


@dataclass
class BugTemplate:
    bug_type: str
    category: str
    difficulty: str
    primary_bug_file: str
    related_files: List[str]
    red_herring_file: Optional[str]
    fix_strategy: str
    line_range: List[int]
    description: str
    artifact_generator: Callable
    repo_mutator: Callable
    metadata: Dict = field(default_factory=dict)


BUG_CATEGORIES = {
    "shape_mismatch": "model",
    "missing_zero_grad": "optimization",
    "wrong_loss_function": "optimization",
    "learning_rate_too_high": "optimization",
    "gradient_explosion": "optimization",
    "memory_leak": "resource",
    "data_leakage": "data",
    "incorrect_normalization": "data",
    "distributed_sync_error": "distributed",
    "amp_overflow": "numerics",
}

# Realistic artifact generator
def dummy_artifact_generator(artifact_type: str, rng):
    if artifact_type == "loss_curve":
        t = np.arange(100)
        base = 2.3 * np.exp(-0.01 * t) + 0.15
        oscillation = 0.22 * np.sin(0.25 * t) * np.exp(-0.002 * t)
        return [
            {"step": int(i), "train_loss": float(base[i] + oscillation[i])}
            for i in range(100)
        ]
    if artifact_type == "gpu_profile":
        t = np.arange(100)
        allocated = 2048 + 2.4 * t
        return [
            {"step": int(i), "allocated_mb": float(allocated[i])}
            for i in range(100)
        ]
    if artifact_type == "training_log":
        return "Epoch 1, Step 0: loss 2.45\nEpoch 1, Step 1: loss 2.43\n"
    if artifact_type == "diagnostic_report":
        return "No critical diagnostics found. Review optimizer and data pipeline."
    return []


def artifact_generator_wrong_loss(artifact_type: str, rng):
    if artifact_type == "loss_curve":
        t = np.arange(100)
        base = 1.8 + 0.05 * np.sin(0.15 * t)
        return [{"step": int(i), "train_loss": float(base[i])} for i in range(100)]
    if artifact_type == "gpu_profile":
        t = np.arange(100)
        allocated = 1900 + 1.8 * t
        return [{"step": int(i), "allocated_mb": float(allocated[i])} for i in range(100)]
    if artifact_type == "training_log":
        return "Epoch 1: loss 1.82, acc 0.11\nEpoch 2: loss 1.80, acc 0.12\n"
    if artifact_type == "diagnostic_report":
        return "Loss plateaus early while accuracy stays near chance. Check loss function."
    return []


def artifact_generator_lr_high(artifact_type: str, rng):
    if artifact_type == "loss_curve":
        t = np.arange(100)
        base = 0.9 + 0.02 * (t ** 1.1)
        return [{"step": int(i), "train_loss": float(base[i])} for i in range(100)]
    if artifact_type == "gpu_profile":
        t = np.arange(100)
        allocated = 2100 + 2.0 * t
        return [{"step": int(i), "allocated_mb": float(allocated[i])} for i in range(100)]
    if artifact_type == "training_log":
        return "Step 10: loss 3.20 (spike)\nStep 20: loss 5.10 (diverged)\n"
    if artifact_type == "diagnostic_report":
        return "Loss spikes suggest unstable updates. Consider lowering learning rate."
    return []


def artifact_generator_amp_overflow(artifact_type: str, rng):
    if artifact_type == "loss_curve":
        t = np.arange(100)
        base = 2.1 * np.exp(-0.008 * t) + 0.2
        base[30:] = base[30:] + 0.6
        return [{"step": int(i), "train_loss": float(base[i])} for i in range(100)]
    if artifact_type == "gpu_profile":
        t = np.arange(100)
        allocated = 2300 + 3.2 * t
        return [{"step": int(i), "allocated_mb": float(allocated[i])} for i in range(100)]
    if artifact_type == "training_log":
        return "AMP: overflow detected, skipping step\nAMP: scale reduced to 32768\n"
    if artifact_type == "diagnostic_report":
        return "AMP overflow warnings observed. Ensure GradScaler is used correctly."
    return []

def mutate_missing_zero_grad(repo_files, rng):
    repo_files["train.py"] = """import torch
from model.architecture import Net

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in dataloader:
        # optimizer.zero_grad()  # BUG: commented out
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
"""
    return repo_files

def mutate_data_leakage(repo_files, rng):
    repo_files["data/dataset.py"] = """from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, data, split="train"):
        # BUG: We use the entire data instead of just the split
        self.data = data
        self.split = split
"""
    return repo_files

def mutate_memory_leak(repo_files, rng):
    repo_files["data/dataset.py"] = """from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self):
        # BUG: Storing huge tensors in a class-level variable leading to memory accumulation
        self.cache = []

    def load(self, x):
        self.cache.append(x)
        return x
"""
    return repo_files


def mutate_wrong_loss_function(repo_files, rng):
    repo_files["train.py"] = """import torch
from model.architecture import Net

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # BUG: wrong loss for classification

for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
"""
    return repo_files


def mutate_learning_rate_too_high(repo_files, rng):
    repo_files["config/training_config.yaml"] = """lr: 1.0
batch_size: 32
"""
    return repo_files


def mutate_amp_overflow(repo_files, rng):
    repo_files["train.py"] = """import torch
from model.architecture import Net

model = Net().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(x.cuda())
            loss = torch.nn.functional.cross_entropy(output, y.cuda())
        # BUG: missing GradScaler handling can cause overflows
        loss.backward()
        optimizer.step()
"""
    return repo_files

BUG_TEMPLATES = [
    BugTemplate(
        bug_type="missing_zero_grad",
        category="optimization",
        difficulty="easy",
        primary_bug_file="train.py",
        related_files=[],
        red_herring_file="model/architecture.py",
        fix_strategy="Call optimizer.zero_grad() before loss.backward()",
        line_range=[9, 14],
        description="Missing zero grad",
        artifact_generator=dummy_artifact_generator,
        repo_mutator=mutate_missing_zero_grad,
    ),
    BugTemplate(
        bug_type="wrong_loss_function",
        category="optimization",
        difficulty="easy",
        primary_bug_file="train.py",
        related_files=["config/training_config.yaml"],
        red_herring_file="data/dataset.py",
        fix_strategy="Use CrossEntropyLoss for classification logits",
        line_range=[6, 12],
        description="Wrong loss function",
        artifact_generator=artifact_generator_wrong_loss,
        repo_mutator=mutate_wrong_loss_function,
    ),
    BugTemplate(
        bug_type="data_leakage",
        category="data",
        difficulty="medium",
        primary_bug_file="data/dataset.py",
        related_files=["data/preprocessing.py"],
        red_herring_file="train.py",
        fix_strategy="Ensure validation split is strictly separate from training",
        line_range=[4, 6],
        description="Data leakage",
        artifact_generator=dummy_artifact_generator,
        repo_mutator=mutate_data_leakage,
    ),
    BugTemplate(
        bug_type="learning_rate_too_high",
        category="optimization",
        difficulty="medium",
        primary_bug_file="config/training_config.yaml",
        related_files=["train.py"],
        red_herring_file="model/attention.py",
        fix_strategy="Reduce learning rate or use a scheduler",
        line_range=[1, 1],
        description="Learning rate too high",
        artifact_generator=artifact_generator_lr_high,
        repo_mutator=mutate_learning_rate_too_high,
    ),
    BugTemplate(
        bug_type="memory_leak",
        category="resource",
        difficulty="hard",
        primary_bug_file="data/dataset.py",
        related_files=["train.py"],
        red_herring_file="model/attention.py",
        fix_strategy="Avoid holding reference to tensors in class cache",
        line_range=[5, 9],
        description="Memory leak",
        artifact_generator=dummy_artifact_generator,
        repo_mutator=mutate_memory_leak,
    ),
    BugTemplate(
        bug_type="amp_overflow",
        category="numerics",
        difficulty="hard",
        primary_bug_file="train.py",
        related_files=["config/training_config.yaml"],
        red_herring_file="model/architecture.py",
        fix_strategy="Use GradScaler and scale updates for AMP",
        line_range=[7, 13],
        description="AMP overflow",
        artifact_generator=artifact_generator_amp_overflow,
        repo_mutator=mutate_amp_overflow,
    ),
]
