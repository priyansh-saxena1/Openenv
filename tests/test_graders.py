# tests/test_graders.py
from src.pytorch_debug_env.graders import grade_easy, grade_hard, grade_medium

def test_grade_easy():
    gt = {
        "bug_type": "missing_zero_grad",
        "primary_bug_file": "train.py",
        "related_files": [],
        "line_range": [10, 15],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    action = {
        "bug_type": "missing_zero_grad",
        "affected_file": "train.py",
        "line_range": [10, 15],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
        "confidence": 0.8
    }
    score = grade_easy(action, gt)
    assert score > 0.8
    assert score < 1.0


def test_grade_medium_related_file_bonus():
    gt = {
        "bug_type": "data_leakage",
        "primary_bug_file": "data/dataset.py",
        "related_files": ["data/preprocessing.py"],
        "line_range": [4, 6],
        "fix_strategy": "Ensure validation split is strictly separate from training",
    }
    action = {
        "bug_type": "data_leakage",
        "affected_file": "data/preprocessing.py",
        "line_range": [1, 2],
        "fix_strategy": "Ensure validation split is strictly separate from training",
        "confidence": 0.6,
    }
    score = grade_medium(action, gt)
    assert score >= grade_easy(action, gt)
    assert 0.0 < score < 1.0


def test_grade_hard_category_partial_credit():
    gt = {
        "bug_type": "missing_zero_grad",
        "category": "optimization",
        "primary_bug_file": "train.py",
        "related_files": [],
        "red_herring_file": "model/attention.py",
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    action = {
        "bug_type": "wrong_loss_function",
        "affected_file": "data/dataset.py",
        "line_range": [1, 2],
        "fix_strategy": "Use CrossEntropyLoss instead of MSE",
        "confidence": 0.5,
    }
    score = grade_hard(action, gt)
    assert score >= 0.18
    assert 0.0 < score < 1.0


def test_grade_hard_penalizes_red_herring():
    gt = {
        "bug_type": "memory_leak",
        "category": "resource",
        "primary_bug_file": "data/dataset.py",
        "related_files": ["train.py"],
        "red_herring_file": "model/attention.py",
        "line_range": [5, 9],
        "fix_strategy": "Avoid holding reference to tensors in class cache",
    }
    action = {
        "bug_type": "memory_leak",
        "affected_file": "model/attention.py",
        "line_range": [5, 9],
        "fix_strategy": "Avoid holding reference to tensors in class cache",
        "confidence": 0.7,
    }
    penalized = grade_hard(action, gt)
    assert penalized <= 0.9
    assert 0.0 < penalized < 1.0


def test_grade_easy_perfect_is_not_one():
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
        "confidence": 1.0,
    }
    score = grade_easy(action, gt)
    assert 0.0 < score < 1.0


def test_grader_empty_action_clamped():
    gt = {
        "bug_type": "data_leakage",
        "primary_bug_file": "data/dataset.py",
        "related_files": [],
        "line_range": [4, 6],
        "fix_strategy": "Ensure validation split is strictly separate from training",
    }
    action = {}
    assert 0.0 < grade_easy(action, gt) < 1.0
    assert 0.0 < grade_medium(action, gt) < 1.0
    assert 0.0 < grade_hard(action, gt) < 1.0
