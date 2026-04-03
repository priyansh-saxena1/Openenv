# tests/test_reward.py
from src.pytorch_debug_env.reward import hypothesis_quality


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
