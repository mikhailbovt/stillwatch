from pathlib import Path

import numpy as np
import pytest

from stillwatch.buffer import FrameBuffer
from stillwatch.capture import make_sample
from stillwatch.config import Settings
from stillwatch.decisions import MockDecisionProvider
from stillwatch.evidence import EvidenceBuilder
from stillwatch.incidents import IncidentManager
from stillwatch.scheduler import Scheduler
from stillwatch.storage import Storage


@pytest.fixture
def settings(tmp_path: Path):
    return Settings(
        data_dir=tmp_path / "data",
        pose_weights=tmp_path / "missing-pose.safetensors",
        fire_weights=tmp_path / "missing-fire.safetensors",
    )


@pytest.fixture
def system(settings):
    # Scheduling scenarios opt into the confirmation gate where it is under test.
    settings.medical_confirmation_frames = 0
    settings.medical_observation_seconds = 0
    buffer = FrameBuffer(settings.buffer_seconds, settings.buffer_max_bytes, settings.buffer_fps)
    image = np.full((120, 160, 3), 120, np.uint8)
    for i in range(61):
        buffer.add(make_sample(image, i, i / 10, 1, settings).frame)
    storage = Storage(settings)
    evidence = EvidenceBuilder(settings, buffer)
    manager = IncidentManager(settings, storage, evidence)
    provider = MockDecisionProvider(delay=0)
    scheduler = Scheduler(settings, manager, provider)
    scheduler.epoch = 1
    scheduler.source_ts = 6
    yield settings, buffer, storage, evidence, manager, provider, scheduler
    storage.close()
