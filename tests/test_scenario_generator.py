import pytest

from src.pytorch_debug_env.bug_library import BUG_TEMPLATES
from src.pytorch_debug_env.scenario_generator import ScenarioGenerator


def test_generate_invalid_difficulty_raises():
    generator = ScenarioGenerator(BUG_TEMPLATES)
    with pytest.raises(ValueError):
        generator.generate("unknown")
