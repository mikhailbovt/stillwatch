from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx

from .models import Event, EventDetail, EventPage, Source, SourceSelection, Video


class StillwatchError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Stillwatch HTTP {status_code}: {detail}")


def checked(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        try:
            detail = response.json().get("detail", "Request failed")
        except (ValueError, AttributeError):
            detail = "Request failed"
        # Do not echo arbitrary proxy pages, request bodies or credentials in errors.
        if not isinstance(detail, str) or len(detail) > 300:
            detail = "Request rejected; inspect the server configuration"
        raise StillwatchError(response.status_code, detail)
    return response


def options(base_url: str, token: str | None, timeout: float) -> dict:
    url = httpx.URL(base_url)
    if url.scheme not in ("http", "https") or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment")
    headers = {"X-Stillwatch-Action": "operator", "User-Agent": "stillwatch-sdk/0.5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return {
        "base_url": str(url).rstrip("/") + "/api/v1/",
        "headers": headers,
        "timeout": timeout,
        "follow_redirects": False,
    }


class SnapshotDecoder:
    """Bound a single SSE message; never accumulate a stream history."""

    def __init__(self):
        self.parts: list[str] = []
        self.size = 0

    def feed(self, line: str) -> dict[str, Any] | None:
        if line.startswith("data:"):
            self.parts.append(line[5:].lstrip())
            self.size += len(line)
            if self.size > 8 * 1024**2:
                raise ValueError("SSE snapshot exceeds the client limit")
        elif not line and self.parts:
            text = "\n".join(self.parts)
            self.parts.clear()
            self.size = 0
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("Expected a snapshot object")
            return value
        return None


class Stillwatch:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        token: str | None = None,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ):
        self._http = httpx.Client(**options(base_url, token, timeout), transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._http.close()

    def _json(self, method: str, path: str, **kwargs):
        return checked(self._http.request(method, path, **kwargs)).json()

    def health(self) -> dict[str, Any]:
        return self._json("GET", "health")

    def telemetry(self) -> dict[str, Any]:
        return self._json("GET", "telemetry")

    def source(self) -> Source:
        return Source.model_validate(self._json("GET", "source"))

    def select_source(self, selection: SourceSelection) -> Source:
        return Source.model_validate(
            self._json("PUT", "source", json=selection.model_dump(exclude_none=True))
        )

    def stop_source(self) -> Source:
        return self.select_source(SourceSelection(kind="none"))

    def videos(self) -> list[Video]:
        return [Video.model_validate(v) for v in self._json("GET", "videos")]

    def upload_video(self, path: str | Path) -> Video:
        path = Path(path)
        with path.open("rb") as stream:
            return Video.model_validate(
                self._json(
                    "POST",
                    "videos",
                    params={"name": path.name},
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(path.stat().st_size),
                    },
                    content=stream,
                )
            )

    def events(self, *, limit: int = 100, offset: int = 0) -> EventPage:
        return EventPage.model_validate(
            self._json("GET", "events", params={"limit": limit, "offset": offset})
        )

    def event(self, incident_id: str) -> EventDetail:
        return EventDetail.model_validate(self._json("GET", f"events/{quote(incident_id, safe='')}"))

    def feedback(self, incident_id: str, action: Literal["confirmed", "false_alarm"]) -> Event:
        return Event.model_validate(
            self._json("POST", f"events/{quote(incident_id, safe='')}/feedback", json={"action": action})
        )

    def close_event(self, incident_id: str) -> Event:
        return Event.model_validate(self._json("POST", f"events/{quote(incident_id, safe='')}/close"))

    def evidence(self, incident_id: str, packet_id: str, index: int) -> bytes:
        path = f"evidence/{quote(incident_id, safe='')}/{quote(packet_id, safe='')}/{index}"
        return checked(self._http.get(path)).content

    def updates(self) -> Iterator[dict[str, Any]]:
        decoder = SnapshotDecoder()
        with self._http.stream("GET", "updates", timeout=httpx.Timeout(30, read=None)) as response:
            if response.is_error:
                response.read()
            checked(response)
            for line in response.iter_lines():
                snapshot = decoder.feed(line)
                if snapshot is not None:
                    yield snapshot


class AsyncStillwatch:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        token: str | None = None,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._http = httpx.AsyncClient(**options(base_url, token, timeout), transport=transport)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self._http.aclose()

    async def _json(self, method: str, path: str, **kwargs):
        return checked(await self._http.request(method, path, **kwargs)).json()

    async def health(self) -> dict[str, Any]:
        return await self._json("GET", "health")

    async def telemetry(self) -> dict[str, Any]:
        return await self._json("GET", "telemetry")

    async def source(self) -> Source:
        return Source.model_validate(await self._json("GET", "source"))

    async def select_source(self, selection: SourceSelection) -> Source:
        return Source.model_validate(
            await self._json("PUT", "source", json=selection.model_dump(exclude_none=True))
        )

    async def stop_source(self) -> Source:
        return await self.select_source(SourceSelection(kind="none"))

    async def videos(self) -> list[Video]:
        return [Video.model_validate(v) for v in await self._json("GET", "videos")]

    async def upload_video(self, path: str | Path) -> Video:
        path = Path(path)

        async def chunks() -> AsyncIterator[bytes]:
            with path.open("rb") as stream:
                while chunk := await asyncio.to_thread(stream.read, 1024 * 1024):
                    yield chunk

        return Video.model_validate(
            await self._json(
                "POST",
                "videos",
                params={"name": path.name},
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(path.stat().st_size),
                },
                content=chunks(),
            )
        )

    async def events(self, *, limit: int = 100, offset: int = 0) -> EventPage:
        return EventPage.model_validate(
            await self._json("GET", "events", params={"limit": limit, "offset": offset})
        )

    async def event(self, incident_id: str) -> EventDetail:
        return EventDetail.model_validate(await self._json("GET", f"events/{quote(incident_id, safe='')}"))

    async def feedback(self, incident_id: str, action: Literal["confirmed", "false_alarm"]) -> Event:
        return Event.model_validate(
            await self._json(
                "POST", f"events/{quote(incident_id, safe='')}/feedback", json={"action": action}
            )
        )

    async def close_event(self, incident_id: str) -> Event:
        return Event.model_validate(await self._json("POST", f"events/{quote(incident_id, safe='')}/close"))

    async def evidence(self, incident_id: str, packet_id: str, index: int) -> bytes:
        return checked(
            await self._http.get(
                f"evidence/{quote(incident_id, safe='')}/{quote(packet_id, safe='')}/{index}"
            )
        ).content

    async def updates(self) -> AsyncIterator[dict[str, Any]]:
        decoder = SnapshotDecoder()
        async with self._http.stream("GET", "updates", timeout=httpx.Timeout(30, read=None)) as response:
            if response.is_error:
                await response.aread()
            checked(response)
            async for line in response.aiter_lines():
                snapshot = decoder.feed(line)
                if snapshot is not None:
                    yield snapshot
