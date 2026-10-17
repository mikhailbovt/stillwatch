"""Optional bearer authentication and short-lived same-origin operator sessions."""

from __future__ import annotations

import hashlib
import hmac
import time

from starlette.requests import Request

COOKIE = "stillwatch_session"


def session_value(token: str, expires: int | None = None) -> str:
    expires = int(time.time()) + 3600 if expires is None else expires
    payload = str(expires)
    signature = hmac.new(token.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def authorized(request: Request, token: str) -> bool:
    if not token:
        return True
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer ") and hmac.compare_digest(header[7:], token):
        return True
    value = request.cookies.get(COOKIE, "")
    try:
        expires = int(value.split(".", 1)[0])
        return time.time() < expires <= time.time() + 3600 and hmac.compare_digest(
            value, session_value(token, expires)
        )
    except (ValueError, IndexError):
        return False
