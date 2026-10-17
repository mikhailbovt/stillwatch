"""Download immutable, hashed model candidates. Never unpickle in the downloader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert", action="store_true", help="Convert in a network-isolated Docker worker")
    parser.add_argument(
        "--local-conversion",
        action="store_true",
        help="Use the restricted local worker instead of OS isolation",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "models.lock.json").read_text())
    staging = root / "models" / "sources"
    staging.mkdir(parents=True, exist_ok=True)
    for name, entry in lock["models"].items():
        target = staging / f"{name}.pt"
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            temporary = target.with_suffix(".part")
            with urllib.request.urlopen(entry["url"], timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if (
                temporary.stat().st_size != entry["size"]
                or hashlib.sha256(temporary.read_bytes()).hexdigest() != entry["sha256"]
            ):
                temporary.unlink(missing_ok=True)
                raise SystemExit(f"Checksum mismatch: {name}")
            temporary.replace(target)
        print(f"Verified {name}: {entry['sha256']}")
        if args.convert:
            if not args.local_conversion:
                output_dir = root / "models" / "isolated"
                output_dir.mkdir(exist_ok=True)
                # Keep Linux bind-mount output writable without running the worker as root.
                container_user = []
                if os.name == "posix" and os.getuid() != 0:
                    container_user = ["--user", f"{os.getuid()}:{os.getgid()}"]
                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        *container_user,
                        "--network",
                        "none",
                        "--read-only",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--memory",
                        "1g",
                        "--cpus",
                        "2",
                        "--pids-limit",
                        "64",
                        "--tmpfs",
                        "/tmp:rw,noexec,nosuid,size=128m",
                        "--mount",
                        f"type=bind,source={staging},target=/inputs,readonly",
                        "--mount",
                        f"type=bind,source={root / 'tools' / 'convert_weights.py'},target=/convert.py,readonly",
                        "--mount",
                        f"type=bind,source={output_dir},target=/outputs",
                        "-e",
                        "YOLO_CONFIG_DIR=/tmp/runtime",
                        "-e",
                        "OMP_NUM_THREADS=2",
                        "stillwatch:0.5.0",
                        "/app/.venv/bin/python",
                        "-I",
                        "/convert.py",
                        f"/inputs/{name}.pt",
                        f"/outputs/{name}.safetensors",
                        entry["sha256"],
                    ],
                    check=True,
                )
                converted = output_dir / f"{name}.safetensors"
                if hashlib.sha256(converted.read_bytes()).hexdigest() != entry["tensor_sha256"]:
                    raise SystemExit(f"Converted artifact checksum mismatch: {name}")
                shutil.copyfile(converted, root / entry["local_path"])
                continue
            # No inherited API keys, user profile, proxy credentials, or repository secrets.
            worker_dir = root / "models" / "conversion"
            worker_dir.mkdir(exist_ok=True)
            env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "PATH") if key in os.environ}
            env.update(
                {
                    "TEMP": str(worker_dir),
                    "TMP": str(worker_dir),
                    "USERPROFILE": str(worker_dir),
                    "HOME": str(worker_dir),
                    "YOLO_CONFIG_DIR": str(worker_dir),
                    "OMP_NUM_THREADS": "2",
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(root / "tools" / "convert_weights.py"),
                    str(target),
                    str(root / entry["local_path"]),
                    entry["sha256"],
                ],
                env=env,
                cwd=worker_dir,
                check=True,
            )


if __name__ == "__main__":
    main()
