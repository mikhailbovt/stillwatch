# Model artifacts

`models.lock.json` records immutable sources, revisions, source/tensor hashes, module-type graph hashes, class maps and license references. Exact runtime versions are in `uv.lock`.

```powershell
uv sync --frozen --extra inference --extra cloud --group dev
uv run python tools/fetch_models.py --convert
```

Fetching is explicit. Normal startup never downloads a model. Missing or invalid artifacts disable that route and produce a visible limitation.

Build the conversion image first with `docker build -t stillwatch:0.5.0 .`. The default converter runs as a non-root user in a container with no network, a read-only root filesystem, dropped capabilities, no new privileges, and bounded memory/processes. Only the source weights, converter, and output directory are mounted; repository secrets and the user profile are not mounted.

On Linux, conversion uses the calling non-root UID/GID to write the bind-mounted output. Root callers must provision `models/isolated` for the image's UID 10001.

It also uses `torch.load(weights_only=True)` and allows only module classes from the pinned PyTorch/Ultralytics runtime. Unapproved serialized globals stop conversion. The safetensors header is canonicalized so the same source produces identical artifacts on Windows and Linux. The converter reports the source module-type graph hash, including shared-module paths. The application loads only hashed safetensors, restores audited non-tensor modules from the lock and rejects a different module-type graph before loading state.

The fire checkpoint stores an Identity activation at `model.9.cv1.act`. The loader restores this activation from the lock. Module-type and tensor hashes identify the audited structure and weights; they do not establish detection quality.

All locked models were converted in the Docker sandbox and their resulting hashes matched `models.lock.json`. The optional `--local-conversion` path uses a restricted Python process without an OS sandbox; it is a fallback, not equivalent isolation. Third-party files can exploit parser/runtime defects; hashes establish artifact identity, not model safety or detection quality.

The default pose artifact is **Ultralytics YOLO26s Pose** (`models/pose-balanced.safetensors`). The smaller YOLO26n artifact remains available as an explicit configuration option. The balanced model was selected for better continuity on the development recordings; it increases inference cost. Detection runs at 5 Hz by default, with bounded causal association owned by Stillwatch. A CPU prediction callback restores four Torch intra-op threads after backend setup and before preprocessing, warmup and inference.

The fire artifact is rabahdev's YOLOv8n D-Fire fine-tune: class 0 smoke, class 1 fire. Author-reported image metrics are not Stillwatch video-event results. The local motion classifier uses geometric rules; it is not a separately trained action model. See [known limitations](limitations.md).

Sources: [Ultralytics pose](https://docs.ultralytics.com/tasks/pose), [tracking](https://docs.ultralytics.com/modes/track), [fire model card](https://huggingface.co/rabahdev/fire-smoke-yolov8n), [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset), [Ultralytics licensing](https://www.ultralytics.com/license).

The upstream code and weights use AGPL-3.0 terms. Application privacy, model licenses, dataset licenses, and permission to redistribute material are separate. Stillwatch does not bundle third-party weights or training data.
