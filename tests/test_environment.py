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
