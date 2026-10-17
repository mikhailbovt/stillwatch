from .client import AsyncStillwatch, Stillwatch, StillwatchError
from .models import Event, EventDetail, EventPage, Motion, Source, SourceSelection, Status, Video

__version__ = "0.5.0"
__all__ = [
    "AsyncStillwatch",
    "Stillwatch",
    "StillwatchError",
    "Event",
    "EventDetail",
    "EventPage",
    "Motion",
    "Source",
    "SourceSelection",
    "Status",
    "Video",
]
