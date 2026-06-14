"""Parametrized tests for python.profiles.get_profile() model-family detection."""

from pathlib import Path

import pytest

try:
    from tests._path_setup import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _path_setup import ensure_repo_root_on_path

ensure_repo_root_on_path()

from python.profiles import get_profile, QuantProfile  # noqa: E402

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "fixture,expected_family,expected_method",
    [
        ("gte", "gte", "int8"),
        ("e5", "e5", "int8"),
        ("bert", "bert", "nf4"),
        ("bge", "bge", "nf4"),
        ("qwen2", "qwen2", "int8"),
        ("deberta", "deberta", "nf4"),
        ("unknown", "generic", "auto"),
    ],
)
def test_get_profile(fixture, expected_family, expected_method):
    profile = get_profile(FIXTURE_DIR / fixture)
    assert profile.family == expected_family
    assert profile.method == expected_method
    assert isinstance(profile.architectures, list)


def test_quant_profile_invalid_method():
    with pytest.raises(ValueError, match="method must be one of"):
        QuantProfile(family="test", method="bad_method")


def test_quant_profile_frozen():
    p = QuantProfile(family="gte", method="int8")
    with pytest.raises((AttributeError, TypeError)):
        p.family = "modified"  # frozen dataclass must raise


def test_get_profile_missing_config(tmp_path):
    """Missing config.json must return the generic fallback, not raise."""
    profile = get_profile(tmp_path)
    assert profile.family == "generic"
    assert profile.method == "auto"


def test_get_profile_malformed_config(tmp_path):
    """config.json with no 'architectures' key falls back to generic."""
    (tmp_path / "config.json").write_text('{"model_type": "custom"}')
    profile = get_profile(tmp_path)
    assert profile.family == "generic"
    assert profile.method == "auto"
