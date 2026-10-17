import pytest
from pydantic import ValidationError

from stillwatch.config import Settings


@pytest.mark.parametrize(
    "changes",
    [
        {"pose_fps": 0},
        {"buffer_max_bytes": 0},
        {"evidence_seconds": 20},
        {"retries": 2},
        {"privacy_masks": [(0.5, 0, 0.2, 1)]},
        {"provider": "openai"},
        {"source": ""},
        {"source": -1},
        {"recheck_seconds": [10, 3]},
        {"requests_per_day": 1},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValidationError):
        Settings(**changes)


def test_default_is_offline_and_hash_is_reproducible():
    settings = Settings()
    assert settings.provider == "mock"
    assert not settings.cloud_spending_enabled
    assert settings.fingerprint == Settings().fingerprint


def test_run_directory_does_not_change_algorithm_hash(tmp_path):
    assert Settings(data_dir=tmp_path / "one").fingerprint == Settings(data_dir=tmp_path / "two").fingerprint
