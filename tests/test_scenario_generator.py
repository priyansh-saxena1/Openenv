import pytest

from src.pytorch_debug_env.bug_library import BUG_TEMPLATES
from src.pytorch_debug_env.scenario_generator import ScenarioGenerator


def test_generate_invalid_difficulty_raises():
    generator = ScenarioGenerator(BUG_TEMPLATES)
    with pytest.raises(ValueError):
        generator.generate("unknown")


def test_generate_seed_reproducibility():
    generator = ScenarioGenerator(BUG_TEMPLATES)
    first = generator.generate("easy", seed=123)
    second = generator.generate("easy", seed=123)
    assert first.ground_truth == second.ground_truth
    assert first.repo_files == second.repo_files
    assert first.training_log == second.training_log
