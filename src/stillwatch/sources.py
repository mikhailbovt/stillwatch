"""Source selection and bounded local uploads. No credentials in public descriptors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


class SourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["none", "file", "camera", "rtsp"]
    file_id: str | None = None
    path: str | None = None
    camera_index: int = Field(default=0, ge=0, le=31)
    url: str | None = Field(default=None, repr=False)
    loop: bool = False


class SourceDescriptor(BaseModel):
    kind: Literal["none", "file", "camera", "rtsp"]
    label: str
    state: str
    epoch: int
    loop: bool = False
    file_id: str | None = None


class UploadedVideo(BaseModel):
    file_id: str
    name: str
    bytes: int


class SourceStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = (settings.data_dir / "sources").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[UploadedVideo]:
        entries = []
        for path in sorted(self.root.glob("*.json")):
            try:
                entry = UploadedVideo.model_validate_json(path.read_text(encoding="utf-8"))
                if self.file(entry.file_id).is_file():
                    entries.append(entry)
            except (ValueError, OSError):
                continue
        return entries

    def file(self, file_id: str) -> Path:
        if len(file_id) != 32 or any(c not in "0123456789abcdef" for c in file_id):
            raise ValueError("Invalid uploaded file ID")
        candidates = [p for p in self.root.glob(file_id + ".*") if p.suffix in VIDEO_SUFFIXES]
        if len(candidates) != 1 or candidates[0].is_symlink():
            raise ValueError("Uploaded video not found")
        return candidates[0]

    def prepare_upload(self, name: str) -> tuple[str, Path, str]:
        name = Path(name.replace("\\", "/")).name[:160]
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("Choose an MP4, AVI, MKV, MOV, M4V or WebM video")
        identifier = uuid4().hex
        return identifier, self.root / (identifier + ".part"), name

    def bytes_used(self) -> int:
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())

    def finish_upload(self, identifier: str, temporary: Path, name: str) -> UploadedVideo:
        final = self.root / (identifier + Path(name).suffix.lower())
        temporary.replace(final)
        entry = UploadedVideo(file_id=identifier, name=name, bytes=final.stat().st_size)
        (self.root / (identifier + ".json")).write_text(entry.model_dump_json() + "\n", encoding="utf-8")
        return entry

    def resolve(self, request: SourceRequest) -> tuple[str | int | None, str]:
        if request.kind == "none":
            return None, "No source"
        if request.kind == "camera":
            return request.camera_index, f"Camera {request.camera_index}"
        if request.kind == "rtsp":
            try:
                parsed = urlsplit(request.url or "")
                if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname or parsed.fragment:
                    raise ValueError
                _ = parsed.port
            except ValueError:
                raise ValueError("A valid RTSP or RTSPS URL is required") from None
            return request.url, f"RTSP · {parsed.hostname}"
        if request.file_id:
            path = self.file(request.file_id)
            metadata = json.loads((self.root / (request.file_id + ".json")).read_text(encoding="utf-8"))
            return str(path), metadata["name"]
        if not request.path:
            raise ValueError("Upload a video or provide an allowed server path")
        path = Path(request.path).resolve()
        if not any(path.is_relative_to(root.resolve()) for root in self.settings.source_roots):
            raise ValueError("Server path is outside configured source_roots")
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("Video file not found or unsupported extension")
        return str(path), path.name


def describe_source(source: str | int | None) -> tuple[str, str]:
    if source is None:
        return "none", "No source"
    if isinstance(source, int):
        return "camera", f"Camera {source}"
    if source.startswith(("rtsp://", "rtsps://")):
        return "rtsp", f"RTSP · {urlsplit(source).hostname or 'stream'}"
    return "file", Path(source).name
