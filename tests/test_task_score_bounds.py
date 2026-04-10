import pytest

from src.pytorch_debug_env.bug_library import BUG_TEMPLATES
from src.pytorch_debug_env.environment import PyTorchDebugEnv
from src.pytorch_debug_env.graders import grade_easy, grade_medium, grade_hard
from src.pytorch_debug_env.models import FinalDiagnosis, Hypothesis, PyTorchDebugAction
from src.pytorch_debug_env.scenario_generator import ScenarioGenerator


def _build_action_from_gt(gt: dict) -> PyTorchDebugAction:
    hypothesis = Hypothesis(
        bug_type=gt["bug_type"],
        affected_file=gt["primary_bug_file"],
        confidence=0.9,
    )
    final = FinalDiagnosis(
        bug_type=gt["bug_type"],
        affected_file=gt["primary_bug_file"],
        line_range=gt["line_range"],
        fix_strategy=gt["fix_strategy"],
        confidence=0.9,
    )
    return PyTorchDebugAction(
        current_hypothesis=hypothesis,
        commit_diagnosis=True,
        final_diagnosis=final,
    )


@pytest.mark.parametrize(
    "task_id,grader",
    [
        ("easy", grade_easy),
        ("medium", grade_medium),
        ("hard", grade_hard),
    ],
)
@pytest.mark.asyncio
async def test_task_scores_strict_bounds(task_id, grader):
    env = PyTorchDebugEnv(generator=ScenarioGenerator(BUG_TEMPLATES))
    await env.reset(task_id, seed=7)
    scenario = env.runtime.scenario
    action = _build_action_from_gt(scenario.ground_truth)

    score = grader(action.final_diagnosis.model_dump(), scenario.ground_truth)
    assert 0.0 < score < 1.0

    result = await env.step(action)
    assert 0.0 < result["reward"] < 1.0
    state = await env.state()
    assert 0.0 < state.final_score < 1.0


@pytest.mark.parametrize(
    "grader",
    [grade_easy, grade_medium, grade_hard],
)
def test_empty_action_is_clamped(grader):
    gt = {
        "bug_type": "missing_zero_grad",
        "primary_bug_file": "train.py",
        "related_files": [],
        "line_range": [10, 12],
        "fix_strategy": "Call optimizer.zero_grad() before loss.backward()",
    }
    score = grader({}, gt)
    assert 0.0 < score < 1.0
