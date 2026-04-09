# src/pytorch_debug_env/scenario_generator.py
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .bug_library import BugTemplate


@dataclass
class Scenario:
    scenario_id: str
    task_id: str
    repo_files: Dict[str, str]
    loss_curve: List[Dict]
    gpu_profile: List[Dict]
    training_log: str
    diagnostic_report: str
    ground_truth: Dict


class ScenarioGenerator:
    def __init__(self, bug_templates: List[BugTemplate]):
        """Create a generator that samples from a set of bug templates."""
        self.bug_templates = bug_templates

    def generate(self, difficulty: str, seed: int | None = None) -> Scenario:
        """Build a scenario with deterministic artifacts when a seed is provided."""
        rng = random.Random(seed)
        candidates = [b for b in self.bug_templates if b.difficulty == difficulty]
        if not candidates:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        template = rng.choice(candidates)

        repo_files = self._base_repo(rng)
        repo_files = template.repo_mutator(repo_files, rng)

        loss_curve = template.artifact_generator("loss_curve", rng)
        gpu_profile = template.artifact_generator("gpu_profile", rng)
        training_log = template.artifact_generator("training_log", rng)
        diagnostic_report = template.artifact_generator("diagnostic_report", rng)

        ground_truth = {
            "bug_type": template.bug_type,
            "category": template.category,
            "primary_bug_file": template.primary_bug_file,
            "related_files": template.related_files,
            "red_herring_file": template.red_herring_file,
            "fix_strategy": template.fix_strategy,
            "line_range": template.line_range,
        }

        return Scenario(
            scenario_id=str(uuid.uuid4())[:8],
            task_id=difficulty,
            repo_files=repo_files,
            loss_curve=loss_curve,
            gpu_profile=gpu_profile,
            training_log=training_log,
            diagnostic_report=diagnostic_report,
            ground_truth=ground_truth,
        )

    def _base_repo(self, rng: random.Random) -> Dict[str, str]:
        return {
            "train.py": self._train_py(),
            "model/architecture.py": self._model_py(),
            "model/attention.py": self._attention_py(),
            "data/dataset.py": self._dataset_py(),
            "data/preprocessing.py": self._preprocess_py(),
            "config/training_config.yaml": self._config_yaml(),
        }

    def _train_py(self) -> str:
        return """import torch\nfrom model.architecture import Net\n\n# training loop placeholder\n"""

    def _model_py(self) -> str:
        return """import torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"""

    def _attention_py(self) -> str:
        return """# custom attention layer\n"""

    def _dataset_py(self) -> str:
        return """from torch.utils.data import Dataset\n\nclass ImageDataset(Dataset):\n    pass\n"""

    def _preprocess_py(self) -> str:
        return """def normalize(x):\n    return x\n"""

    def _config_yaml(self) -> str:
        return "lr: 0.001\nbatch_size: 32\n"
