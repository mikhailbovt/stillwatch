from __future__ import annotations

import os
from pathlib import Path


def api_key() -> str | None:
    """An explicitly provisioned local key takes precedence over a stale parent environment."""
    local = Path.cwd() / ".env.local"
    if local.is_symlink():
        raise ValueError("Credential file must not be a symbolic link")
    if local.is_file():
        from dotenv import dotenv_values

        value = dotenv_values(local, interpolate=False).get("OPENAI_API_KEY")
        if value:
            return value
    return os.environ.get("OPENAI_API_KEY") or None
