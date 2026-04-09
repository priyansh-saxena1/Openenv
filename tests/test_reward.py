# tests/test_reward.py
from src.pytorch_debug_env.reward import (
    clamp_score,
    compute_step_reward,
    final_diagnosis_score,
    hypothesis_quality,
    line_overlap,
)


def test_hypothesis_quality_exact_match():
    gt = {
        "bug_type": "missing_zero_grad",
        "primary_bug_file": "train.py",
        "related_files": [],
    }
    hyp = {
        "bug_type": "missing_zero_grad",
        "affected_file": "train.py",
        "confidence": 0.8,
    }
    assert hypothesis_quality(hyp, gt) > 0.8


def test_line_overlap_handles_no_overlap():
    assert line_overlap([1, 2], [5, 6]) == 0.0


def test_final_diagnosis_score_bounds():
    gt = {
        "bug_type": "missing_zero_grad",
        "primary_bug_file": "train.py",
        "related_files": [],
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    action = {
        "bug_type": "missing_zero_grad",
        "affected_file": "train.py",
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    score = final_diagnosis_score(action, gt)
    assert 0.0 < score < 1.0


def test_final_diagnosis_score_perfect_clamped():
    gt = {
        "bug_type": "missing_zero_grad",
        "primary_bug_file": "train.py",
        "related_files": [],
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    action = {
        "bug_type": "missing_zero_grad",
        "affected_file": "train.py",
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    score = final_diagnosis_score(action, gt)
    assert 0.0 < score < 1.0


def test_compute_step_reward_clamps_non_negative():
    gt = {
        "bug_type": "missing_zero_grad",
        "primary_bug_file": "train.py",
        "related_files": [],
        "red_herring_file": "model/architecture.py",
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    hypothesis = {
        "bug_type": "data_leakage",
        "affected_file": "unknown.py",
        "confidence": 0.1,
    }
    reward, components = compute_step_reward(
        previous_quality=0.6,
        current_hypothesis=hypothesis,
        ground_truth=gt,
        investigation_target="model/architecture.py",
        committed_diagnosis=None,
        step_num=1,
        max_steps=5,
    )
    assert 0.0 < reward < 1.0
    assert components["investigation_reward"] <= 0.0


def test_clamp_score_open_interval():
    assert 0.0 < clamp_score(0.0) < 1.0
    assert 0.0 < clamp_score(1.0) < 1.0
