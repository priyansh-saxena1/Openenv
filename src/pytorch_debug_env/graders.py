# src/pytorch_debug_env/graders.py
from __future__ import annotations

from .reward import final_diagnosis_score


def grade_easy(action: dict, gt: dict) -> float:
    return final_diagnosis_score(action, gt)


def grade_medium(action: dict, gt: dict) -> float:
    score = final_diagnosis_score(action, gt)
    if action.get("affected_file") in gt.get("related_files", []):
        score = min(1.0, score + 0.05)
    return round(score, 4)


def grade_hard(action: dict, gt: dict) -> float:
    score = final_diagnosis_score(action, gt)

    # partial credit if model gets the right category on subtle bugs
    if score < 0.2 and action.get("bug_type"):
        if gt.get("category"):
            from .bug_library import BUG_CATEGORIES
            if BUG_CATEGORIES.get(action["bug_type"]) == gt["category"]:
                score = max(score, 0.18)

    if action.get("affected_file") == gt.get("red_herring_file"):
        score = max(0.0, score - 0.1)

    return round(min(score, 1.0), 4)
