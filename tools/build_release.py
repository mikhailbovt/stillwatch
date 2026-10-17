"""Build explicit server/SDK assets and a checksummed integration bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path

from stillwatch.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def contract() -> bytes:
    return (json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n").encode()


def build(output: Path):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    sdk_version = tomllib.loads((ROOT / "sdk/python/pyproject.toml").read_text())["project"]["version"]
    if version != sdk_version:
        raise ValueError("Server and SDK release versions must match")
    if output.exists() and any(p.name != ".gitignore" for p in output.iterdir()):
        raise ValueError("Choose an empty output directory; existing artifacts are never overwritten")
    if (ROOT / "api/openapi.json").read_bytes() != contract():
        raise ValueError("Export the current API contract before building")
    output.mkdir(parents=True, exist_ok=True)
    for source in (ROOT, ROOT / "sdk/python"):
        subprocess.run(["uv", "build", str(source), "--out-dir", str(output)], check=True, cwd=ROOT)
    (output / "openapi.json").write_bytes(contract())
    with zipfile.ZipFile(
        output / f"stillwatch-integration-{version}.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        paths = [
            ROOT / p
            for p in (
                "api/openapi.json",
                "docs/integration.md",
                "docs/deployment.md",
                "docs/limitations.md",
                "docs/models.md",
                "sdk/python/README.md",
                "sdk/python/LICENSE",
                "configs/default.yaml",
                "models.lock.json",
                "LICENSE",
                "NOTICE",
                "THIRD_PARTY_NOTICES.md",
            )
        ]
        paths += sorted((ROOT / "examples").glob("*.py"))
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    artifacts = sorted(p for p in output.iterdir() if p.name != ".gitignore")
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in artifacts),
        encoding="utf-8",
    )
    print(
        json.dumps({"version": version, "directory": str(output), "artifacts": [p.name for p in artifacts]})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--export-contract", action="store_true")
    args = parser.parse_args()
    if args.export_contract:
        path = ROOT / "api/openapi.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(contract())
    else:
        build(args.output.resolve())


if __name__ == "__main__":
    main()
