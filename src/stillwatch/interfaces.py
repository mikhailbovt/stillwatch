"""Public local-inference contract. Adapter code runs inside the trusted server."""

from typing import Protocol

import numpy as np

from .types import Observation


class PerceptionAdapter(Protocol):
    states: dict[str, str]
    errors: dict[str, str]

    def load(self) -> None: ...

    def reset(self) -> None: ...

    def people(self, image: np.ndarray, capture_ts: float) -> list[Observation]: ...

    def hazards(self, image: np.ndarray, capture_ts: float) -> list[Observation]: ...
