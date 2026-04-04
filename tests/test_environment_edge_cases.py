import pytest

from src.pytorch_debug_env.bug_library import BUG_TEMPLATES
from src.pytorch_debug_env.environment import PyTorchDebugEnv
from src.pytorch_debug_env.models import (
    FinalDiagnosis,
    Hypothesis,
    InvestigationAction,
    PyTorchDebugAction,
)
from src.pytorch_debug_env.scenario_generator import ScenarioGenerator


def make_env():
    generator = ScenarioGenerator(BUG_TEMPLATES)
    return PyTorchDebugEnv(generator=generator)


def base_hypothesis():
    return Hypothesis(
        bug_type="missing_zero_grad",
        affected_file="train.py",
        confidence=0.6,
    )


def final_diagnosis():
    return FinalDiagnosis(
        bug_type="missing_zero_grad",
        affected_file="train.py",
        line_range=[9, 14],
        fix_strategy="Call optimizer.zero_grad() before loss.backward()",
        confidence=0.7,
    )


@pytest.mark.asyncio
async def test_state_before_reset_returns_none():
    env = make_env()
    assert await env.state() is None


@pytest.mark.asyncio
async def test_step_without_reset_raises():
    env = make_env()
    action = PyTorchDebugAction(current_hypothesis=base_hypothesis())
    with pytest.raises(RuntimeError):
        await env.step(action)


@pytest.mark.asyncio
async def test_reveal_file_adds_to_observation():
    env = make_env()
    await env.reset("easy")
    target = "data/dataset.py"
    action = PyTorchDebugAction(
        current_hypothesis=base_hypothesis(),
        investigation_action=InvestigationAction(action="reveal_file", target=target),
    )
    result = await env.step(action)
    assert target in result["observation"].revealed_files


@pytest.mark.asyncio
async def test_step_after_done_raises():
    env = make_env()
    await env.reset("easy")
    action = PyTorchDebugAction(
        current_hypothesis=base_hypothesis(),
        commit_diagnosis=True,
        final_diagnosis=final_diagnosis(),
    )
    await env.step(action)
    with pytest.raises(RuntimeError):
        await env.step(action)


@pytest.mark.asyncio
async def test_reward_range_and_info_keys():
    env = make_env()
    await env.reset("easy")
    action = PyTorchDebugAction(
        current_hypothesis=base_hypothesis(),
        investigation_action=InvestigationAction(
            action="reveal_file",
            target="model/attention.py",
        ),
    )
    result = await env.step(action)
    assert 0.0 <= result["reward"] <= 1.0
    for key in (
        "hypothesis_quality",
        "hypothesis_delta",
        "investigation_reward",
        "diagnosis_reward",
        "confirmation_bonus",
    ):
        assert key in result["info"]
