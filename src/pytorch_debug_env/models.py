# src/pytorch_debug_env/models.py
from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    bug_type: str = Field(..., description="Current suspected bug type")
    affected_file: str = Field(..., description="Current suspected file")
    confidence: float = Field(..., ge=0.0, le=1.0)


class InvestigationAction(BaseModel):
    action: Literal[
        "reveal_file",
        "extend_loss_curve",
        "extend_gpu_profile",
        "reveal_log_chunk",
        "run_diagnostic",
    ]
    target: Optional[str] = None


class FinalDiagnosis(BaseModel):
    bug_type: str
    affected_file: str
    line_range: List[int]
    fix_strategy: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PyTorchDebugAction(BaseModel):
    current_hypothesis: Hypothesis
    investigation_action: Optional[InvestigationAction] = None
    commit_diagnosis: bool = False
    final_diagnosis: Optional[FinalDiagnosis] = None


class HypothesisRecord(BaseModel):
    step: int
    hypothesis: Hypothesis
    quality: float


class PyTorchDebugObservation(BaseModel):
    scenario_id: str
    task_id: str
    revealed_files: Dict[str, str]
    available_files: List[str]
    loss_curve_window: List[Dict]
    gpu_profile_window: List[Dict]
    training_log_tail: str
    diagnostic_report: Optional[str] = None
    step_num: int
    steps_remaining: int
    investigation_budget: int
    hypothesis_history: List[HypothesisRecord]
    last_feedback: str


class PyTorchDebugState(BaseModel):
    scenario_id: str
    task_id: str
    max_steps: int
    current_step: int
    revealed_files: List[str]
    remaining_files: List[str]
    diagnostic_revealed: bool = False
    done: bool
    final_score: float = 0.0


class PyTorchDebugReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float]
