# src/pytorch_debug_env/environment.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .models import (
    HypothesisRecord,
    PyTorchDebugAction,
    PyTorchDebugObservation,
    PyTorchDebugState,
)
from .reward import compute_step_reward
from .scenario_generator import ScenarioGenerator
from .graders import grade_easy, grade_medium, grade_hard

GRADER_MAP = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}


@dataclass
class RuntimeState:
    scenario: object | None = None
    max_steps: int = 5
    current_step: int = 0
    revealed_files: List[str] = field(default_factory=list)
    hypothesis_history: List[HypothesisRecord] = field(default_factory=list)
    done: bool = False
    final_score: float = 0.0


class PyTorchDebugEnv:
    def __init__(self, generator: ScenarioGenerator, max_steps: int = 5):
        self.generator = generator
        self.runtime = RuntimeState(max_steps=max_steps)

    async def reset(self, task_id: str = "easy"):
        scenario = self.generator.generate(task_id)
        self.runtime = RuntimeState(
            scenario=scenario,
            max_steps=5 if task_id == "easy" else 6,
            current_step=0,
            revealed_files=["train.py", "config/training_config.yaml"],
            hypothesis_history=[],
            done=False,
            final_score=0.0,
        )
        return self._build_observation(last_feedback="Episode reset.")

    async def step(self, action: PyTorchDebugAction):
        if self.runtime.scenario is None:
            raise RuntimeError("Call /reset before /step")

        if self.runtime.done:
            raise RuntimeError("Episode already completed")

        self.runtime.current_step += 1
        scenario = self.runtime.scenario
        previous_quality = self.runtime.hypothesis_history[-1].quality if self.runtime.hypothesis_history else 0.0

        investigation_target = None
        if action.investigation_action and action.investigation_action.action == "reveal_file":
            investigation_target = action.investigation_action.target
            if investigation_target in scenario.repo_files and investigation_target not in self.runtime.revealed_files:
                self.runtime.revealed_files.append(investigation_target)

        committed = action.final_diagnosis.model_dump() if action.commit_diagnosis and action.final_diagnosis else None
        reward, components = compute_step_reward(
            previous_quality=previous_quality,
            current_hypothesis=action.current_hypothesis.model_dump(),
            ground_truth=scenario.ground_truth,
            investigation_target=investigation_target,
            committed_diagnosis=None, # Temporarily don't compute diagnosis reward here to use grader
            step_num=self.runtime.current_step,
            max_steps=self.runtime.max_steps,
        )

        if committed:
            grader = GRADER_MAP.get(scenario.task_id, grade_easy)
            diagnosis_reward = grader(committed, scenario.ground_truth)

            # Combine the diagnosis reward logic from `compute_step_reward` that applies on top
            if diagnosis_reward > 0.7:
                diagnosis_reward += max(0.0, 0.08 * (self.runtime.max_steps - self.runtime.current_step))

            # Update the total reward incorporating diagnosis
            components["diagnosis_reward"] = round(diagnosis_reward, 4)
            delta = components["hypothesis_delta"]
            inv_reward = components["investigation_reward"]
            conf_bonus = components["confirmation_bonus"]

            total = 0.60 * delta + 0.20 * inv_reward + 0.20 * diagnosis_reward + conf_bonus
            reward = round(min(max(total, 0.0), 1.0), 4)

        self.runtime.hypothesis_history.append(
            HypothesisRecord(
                step=self.runtime.current_step,
                hypothesis=action.current_hypothesis,
                quality=components["hypothesis_quality"],
            )
        )

        if action.commit_diagnosis or self.runtime.current_step >= self.runtime.max_steps:
            self.runtime.done = True
            self.runtime.final_score = reward

        observation = self._build_observation(
            last_feedback=self._feedback(action, scenario.ground_truth)
        )
        return {
            "observation": observation,
            "reward": reward,
            "done": self.runtime.done,
            "info": components,
        }

    async def state(self):
        scenario = self.runtime.scenario
        if not scenario:
            return None
        return PyTorchDebugState(
            scenario_id=scenario.scenario_id,
            task_id=scenario.task_id,
            max_steps=self.runtime.max_steps,
            current_step=self.runtime.current_step,
            revealed_files=self.runtime.revealed_files,
            remaining_files=[
                f for f in scenario.repo_files.keys() if f not in self.runtime.revealed_files
            ],
            done=self.runtime.done,
            final_score=self.runtime.final_score,
        )

    def _build_observation(self, last_feedback: str) -> PyTorchDebugObservation:
        scenario = self.runtime.scenario
        revealed = {k: v for k, v in scenario.repo_files.items() if k in self.runtime.revealed_files}
        available = [k for k in scenario.repo_files.keys() if k not in self.runtime.revealed_files]

        loss_window_size = min(len(scenario.loss_curve), 100 * (self.runtime.current_step + 1))
        gpu_window_size = min(len(scenario.gpu_profile), 100 * (self.runtime.current_step + 1))
        log_lines = scenario.training_log.splitlines()
        visible_log = "\n".join(log_lines[-min(len(log_lines), 10 * (self.runtime.current_step + 1)):])

        return PyTorchDebugObservation(
            scenario_id=scenario.scenario_id,
            task_id=scenario.task_id,
            revealed_files=revealed,
            available_files=available,
            loss_curve_window=scenario.loss_curve[:loss_window_size],
            gpu_profile_window=scenario.gpu_profile[:gpu_window_size],
            training_log_tail=visible_log,
            step_num=self.runtime.current_step,
            steps_remaining=max(0, self.runtime.max_steps - self.runtime.current_step),
            investigation_budget=max(0, self.runtime.max_steps - self.runtime.current_step),
            hypothesis_history=self.runtime.hypothesis_history,
            last_feedback=last_feedback,
        )

    def _feedback(self, action: PyTorchDebugAction, gt: Dict) -> str:
        suspected_file = action.current_hypothesis.affected_file
        suspected_bug = action.current_hypothesis.bug_type

        if suspected_file == gt.get("red_herring_file"):
            return "That file contains a plausible symptom, but not the root cause. Investigate upstream causes."
        if suspected_file == gt["primary_bug_file"] and suspected_bug != gt["bug_type"]:
            return "Correct region, wrong failure mode. Re-check the training artifacts more carefully."
        if suspected_bug == gt["bug_type"] and suspected_file != gt["primary_bug_file"]:
            return "The bug class looks right, but the faulty implementation is in another file."
        return "Continue refining the hypothesis using newly revealed evidence."
