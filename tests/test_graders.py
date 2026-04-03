# tests/test_graders.py
from src.pytorch_debug_env.graders import grade_easy

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
    assert grade_easy(action, gt) > 0.8
