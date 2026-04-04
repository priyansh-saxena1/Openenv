# tests/test_environment.py
import pytest
from src.pytorch_debug_env.environment import PyTorchDebugEnv
from src.pytorch_debug_env.scenario_generator import ScenarioGenerator
from src.pytorch_debug_env.bug_library import BUG_TEMPLATES

@pytest.mark.asyncio
async def test_env_reset():
    generator = ScenarioGenerator(BUG_TEMPLATES)
    env = PyTorchDebugEnv(generator=generator)
    obs = await env.reset("easy")
    assert obs.task_id == "easy"
    assert "train.py" in obs.revealed_files
    assert "config/training_config.yaml" in obs.revealed_files
    assert obs.step_num == 0
    assert obs.steps_remaining >= 0
