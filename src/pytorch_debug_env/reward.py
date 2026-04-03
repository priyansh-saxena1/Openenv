# src/pytorch_debug_env/reward.py
from __future__ import annotations

from .bug_library import BUG_CATEGORIES


def hypothesis_quality(hypothesis: dict, ground_truth: dict) -> float:
    q = 0.0

    if hypothesis.get("affected_file") == ground_truth["primary_bug_file"]:
        q += 0.45
    elif hypothesis.get("affected_file") in ground_truth.get("related_files", []):
        q += 0.15

    if hypothesis.get("bug_type") == ground_truth["bug_type"]:
        q += 0.40
    elif BUG_CATEGORIES.get(hypothesis.get("bug_type")) == BUG_CATEGORIES.get(ground_truth["bug_type"]):
        q += 0.13

    calibration = 1.0 - abs(hypothesis.get("confidence", 0.5) - min(q, 1.0))
    q += 0.15 * calibration
    return round(min(q, 1.0), 4)


def final_diagnosis_score(diagnosis: dict, ground_truth: dict) -> float:
    score = 0.0

    if diagnosis.get("bug_type") == ground_truth["bug_type"]:
        score += 0.40
    if diagnosis.get("affected_file") == ground_truth["primary_bug_file"]:
        score += 0.25

    predicted = diagnosis.get("line_range", [0, 0])
    actual = ground_truth.get("line_range", [0, 0])
    overlap = line_overlap(predicted, actual)
    score += 0.20 * overlap

    if diagnosis.get("fix_strategy") == ground_truth["fix_strategy"]:
        score += 0.15

    return round(min(score, 1.0), 4)


def line_overlap(pred: list[int], actual: list[int]) -> float:
    p1, p2 = pred
    a1, a2 = actual
    inter = max(0, min(p2, a2) - max(p1, a1) + 1)
    union = max(p2, a2) - min(p1, a1) + 1
    return inter / union if union else 0.0


def compute_step_reward(
    previous_quality: float,
    current_hypothesis: dict,
    ground_truth: dict,
    investigation_target: str | None = None,
    committed_diagnosis: dict | None = None,
    step_num: int = 1,
    max_steps: int = 5,
) -> tuple[float, dict]:
    current_quality = hypothesis_quality(current_hypothesis, ground_truth)
    delta = current_quality - previous_quality

    confirmation_bonus = 0.03 * current_quality if abs(delta) < 0.01 else 0.0

    investigation_reward = 0.0
    if investigation_target:
        if investigation_target == ground_truth["primary_bug_file"]:
            investigation_reward = 0.07
        elif investigation_target in ground_truth.get("related_files", []):
            investigation_reward = 0.025
        elif investigation_target == ground_truth.get("red_herring_file"):
            investigation_reward = -0.04
        else:
            investigation_reward = -0.01

    diagnosis_reward = 0.0
    if committed_diagnosis:
        diagnosis_reward = final_diagnosis_score(committed_diagnosis, ground_truth)
        if diagnosis_reward > 0.7:
            diagnosis_reward += max(0.0, 0.08 * (max_steps - step_num))

    total = 0.60 * delta + 0.20 * investigation_reward + 0.20 * diagnosis_reward + confirmation_bonus
    total = round(min(max(total, -0.2), 1.0), 4)

    return total, {
        "hypothesis_quality": current_quality,
        "hypothesis_delta": round(delta, 4),
        "investigation_reward": round(investigation_reward, 4),
        "diagnosis_reward": round(diagnosis_reward, 4),
        "confirmation_bonus": round(confirmation_bonus, 4),
    }
