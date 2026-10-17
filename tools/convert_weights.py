"""Reject arbitrary pickle globals; export tensor-only weights in a secret-free worker.

This worker uses PyTorch's restricted unpickler, never weights_only=False. It is
defense in depth rather than a claim of an OS security sandbox. See docs/models.md.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != sys.argv[3]:
        raise SystemExit("Source hash mismatch")
    # Only known classes from the pinned runtime are eligible, never artifact-provided code.
    approved_prefixes = ("ultralytics.nn.modules.", "ultralytics.nn.tasks.", "torch.nn.modules.")
    approved = []
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(source):
        if not name.startswith(approved_prefixes) or name.rsplit(".", 1)[-1].startswith("_"):
            raise SystemExit(f"Unapproved serialized global: {name}")
        module_name, symbol = name.rsplit(".", 1)
        value = getattr(importlib.import_module(module_name), symbol)
        if not isinstance(value, type) or not issubclass(value, torch.nn.Module):
            raise SystemExit(f"Serialized global is not a trusted module class: {name}")
        approved.append((value, name))
    with torch.serialization.safe_globals(approved):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model = checkpoint.get("ema") or checkpoint["model"]
    modules = {
        name: type(layer).__module__ + "." + type(layer).__qualname__
        for name, layer in model.named_modules(remove_duplicate=False)
    }
    module_types_sha256 = hashlib.sha256(
        json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    task = "pose" if "pose" in type(model).__name__.lower() else "detect"
    tensors = {name: tensor.float().contiguous().clone() for name, tensor in model.state_dict().items()}
    metadata = {
        "task": task,
        "yaml": json.dumps(model.yaml),
        "names": json.dumps(model.names),
        "source_sha256": digest,
        "format": "stillwatch-tensors-v1",
    }
    save_file(tensors, str(destination), metadata=metadata)
    # safetensors metadata uses an unordered map. Canonicalize the JSON header so
    # the locked artifact has identical bytes across fresh conversion processes.
    payload = destination.read_bytes()
    header_size = struct.unpack("<Q", payload[:8])[0]
    header = json.dumps(
        json.loads(payload[8 : 8 + header_size]), sort_keys=True, separators=(",", ":")
    ).encode()
    header += b" " * ((-len(header)) % 8)
    destination.write_bytes(struct.pack("<Q", len(header)) + header + payload[8 + header_size :])
    print(
        json.dumps(
            {
                "file": destination.name,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "source_sha256": digest,
                "module_types_sha256": module_types_sha256,
                "task": task,
                "classes": model.names,
            }
        )
    )


if __name__ == "__main__":
    main()
