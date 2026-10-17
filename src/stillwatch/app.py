from __future__ import annotations

import asyncio
import hmac
import json
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .auth import COOKIE, authorized, session_value
from .config import Settings
from .runtime import Runtime
from .sources import SourceDescriptor, SourceRequest, UploadedVideo
from .types import Incident


class Feedback(BaseModel):
    action: Literal["confirmed", "false_alarm"]


class EventPage(BaseModel):
    events: list[Incident]
    total: int


def create_app(
    settings: Settings | None = None, *, runtime_factory: Callable[[Settings], Runtime] = Runtime
) -> FastAPI:
    settings = settings or Settings()
    token = os.environ.get(settings.api_token_env, "")
    if settings.host not in ("127.0.0.1", "localhost", "::1") and len(token) < 24:
        raise ValueError("Network binding requires an API token of at least 24 characters")
    uploads_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory(settings)
        app.state.runtime = runtime
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="Stillwatch", version=__version__, lifespan=lifespan)
    api = APIRouter()
    static = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.middleware("http")
    async def local_headers(request: Request, call_next):
        public = request.url.path in (
            "/",
            "/health",
            "/api/v1/health",
            "/api/v1/session",
        ) or request.url.path.startswith("/static/")
        if not public and not authorized(request, token):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Validation responses must not reflect RTSP passwords or other request input.
        return JSONResponse(
            {"detail": [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]},
            status_code=422,
        )

    @app.post("/api/v1/session", include_in_schema=False)
    async def login(request: Request):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token and not hmac.compare_digest(supplied, token):
            raise HTTPException(401, "Invalid access token")
        response = JSONResponse({"authenticated": True})
        if token:
            response.set_cookie(
                COOKIE,
                session_value(token),
                max_age=3600,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="strict",
            )
        return response

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @api.get("/health")
    async def health(request: Request):
        snapshot = request.app.state.runtime.snapshot()
        return {
            "service": "running",
            "mode": snapshot["mode"],
            "capture": snapshot["capture_state"],
            "routes": snapshot["routes"],
            "limitations": snapshot["limitations"],
        }

    @api.get("/telemetry")
    async def telemetry(request: Request):
        snapshot = request.app.state.runtime.snapshot()
        snapshot.pop("events")
        snapshot.pop("overlays")
        return snapshot

    @api.get("/events", response_model=EventPage)
    async def events(request: Request, limit: int = 100, offset: int = 0):
        if not 1 <= limit <= 500 or offset < 0:
            raise HTTPException(400, "Invalid pagination")
        items = sorted(
            request.app.state.runtime.manager.items.values(), key=lambda i: i.created_at, reverse=True
        )
        return {
            "events": [i.model_dump(mode="json") for i in items[offset : offset + limit]],
            "total": len(items),
        }

    @api.get("/events/{incident_id}")
    async def detail(incident_id: str, request: Request):
        runtime = request.app.state.runtime
        if incident_id not in runtime.manager.items:
            raise HTTPException(404, "Event not found")
        return {
            "event": runtime.manager.items[incident_id].model_dump(mode="json"),
            **runtime.manager.detail(incident_id),
        }

    def operator_action(runtime: Runtime, incident_id: str, action: str, marker: str | None):
        if marker != "operator":
            raise HTTPException(403, "Operator action header required")
        if incident_id not in runtime.manager.items:
            raise HTTPException(404, "Event not found")
        runtime.manager.annotate(incident_id, action)
        return runtime.manager.items[incident_id]

    @api.post("/events/{incident_id}/feedback", response_model=Incident)
    async def feedback(
        incident_id: str,
        payload: Feedback,
        request: Request,
        x_stillwatch_action: str | None = Header(default=None),
    ):
        return operator_action(request.app.state.runtime, incident_id, payload.action, x_stillwatch_action)

    @api.post("/events/{incident_id}/close", response_model=Incident)
    async def close(
        incident_id: str, request: Request, x_stillwatch_action: str | None = Header(default=None)
    ):
        return operator_action(request.app.state.runtime, incident_id, "close", x_stillwatch_action)

    @api.get("/evidence/{incident_id}/{packet_id}/{index}")
    def evidence(incident_id: str, packet_id: str, index: int, request: Request):
        path = request.app.state.runtime.storage.evidence_path(incident_id, packet_id, index)
        if path is None:
            raise HTTPException(404, "Evidence unavailable or expired")
        return FileResponse(path, media_type="image/jpeg")

    @api.get("/stream")
    async def stream(request: Request):
        async def frames():
            previous = None
            while not await request.is_disconnected():
                sample = request.app.state.runtime.capture.latest
                if sample and (sample.frame.epoch, sample.frame.frame_id) != previous:
                    previous = (sample.frame.epoch, sample.frame.frame_id)
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + sample.frame.jpeg + b"\r\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.get("/updates")
    async def updates(request: Request):
        async def snapshots():
            while not await request.is_disconnected():
                payload = request.app.state.runtime.snapshot()
                yield "event: snapshot\ndata: " + json.dumps(payload) + "\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            snapshots(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @api.get("/source", response_model=SourceDescriptor)
    async def source(request: Request):
        return request.app.state.runtime.source_descriptor()

    @api.put("/source", response_model=SourceDescriptor)
    async def select_source(
        payload: SourceRequest, request: Request, x_stillwatch_action: str | None = Header(default=None)
    ):
        if x_stillwatch_action != "operator":
            raise HTTPException(403, "Operator action header required")
        try:
            return await request.app.state.runtime.switch_source(payload)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None

    @api.get("/videos", response_model=list[UploadedVideo])
    async def videos(request: Request):
        return request.app.state.runtime.sources.list()

    @api.post("/videos", response_model=UploadedVideo, status_code=201)
    async def upload_video(
        request: Request, name: str, x_stillwatch_action: str | None = Header(default=None)
    ):
        if x_stillwatch_action != "operator":
            raise HTTPException(403, "Operator action header required")
        store = request.app.state.runtime.sources
        try:
            identifier, path, label = store.prepare_upload(name)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        async with uploads_lock:
            used = store.bytes_used()
            total = 0
            try:
                with path.open("xb") as output:
                    async for chunk in request.stream():
                        total += len(chunk)
                        if (
                            total > settings.source_upload_max_bytes
                            or used + total > settings.source_storage_quota_bytes
                        ):
                            raise HTTPException(413, "Video upload or source storage limit exceeded")
                        output.write(chunk)
                if total == 0:
                    raise HTTPException(400, "Video upload is empty")
                return store.finish_upload(identifier, path, label)
            finally:
                path.unlink(missing_ok=True)

    app.include_router(api, prefix="/api/v1")
    app.include_router(api, include_in_schema=False)
    # Document the middleware's authentication contract for generated clients.
    original_openapi = app.openapi

    def contract():
        schema = original_openapi()
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "Required when server API authentication is configured.",
        }
        for path, methods in schema["paths"].items():
            for operation in methods.values():
                if path != "/api/v1/health" and isinstance(operation, dict):
                    operation["security"] = [{"BearerAuth": []}]
        schema["paths"]["/api/v1/videos"]["post"]["requestBody"] = {
            "required": True,
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        }
        return schema

    app.openapi = contract
    return app
