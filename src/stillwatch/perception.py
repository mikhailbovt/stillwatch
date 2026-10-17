from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .types import Box, Observation


def visible_keypoints(points, width: int, height: int) -> list[tuple[float, float, float]]:
    """Clipped boundary coordinates and nonfinite predictions are not visible joints."""
    visible = []
    for x, y, confidence in points:
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            visible.append((0, 0, 0))
        else:
            visible.append((x, y, confidence if 0 < x < width and 0 < y < height else 0))
    return visible


def torso_histogram(image: np.ndarray, points: list[tuple[float, float, float]]) -> tuple[float, ...]:
    """Coarse clothing colors from visible torso joints, with no face descriptor."""
    if len(points) < 13 or min(points[k][2] for k in (5, 6, 11, 12)) < 0.4:
        return ()
    polygon = np.array([points[k][:2] for k in (5, 6, 12, 11)], np.float32)
    height, width = image.shape[:2]
    if (
        not np.isfinite(polygon).all()
        or not (
            (polygon[:, 0] > 0) & (polygon[:, 0] < width) & (polygon[:, 1] > 0) & (polygon[:, 1] < height)
        ).all()
    ):
        return ()
    center = polygon.mean(axis=0)
    polygon = center + (polygon - center) * 0.8
    hull = cv2.convexHull(polygon)
    if cv2.contourArea(hull) < 64:
        return ()
    x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
    x2, y2 = np.ceil(polygon.max(axis=0)).astype(int) + 1
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return ()
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull - [x1, y1]).astype(np.int32), 255)
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[mask > 0]
    if len(hsv) < 48:
        return ()
    chromatic = (hsv[:, 1] >= 64) & (hsv[:, 2] >= 32)
    hue = np.bincount(hsv[chromatic, 0].astype(np.int32) * 12 // 180, minlength=12).astype(float)
    grey = np.bincount(hsv[~chromatic, 2].astype(np.int32) * 4 // 256, minlength=4).astype(float)
    hue = 0.5 * hue + 0.25 * np.roll(hue, 1) + 0.25 * np.roll(hue, -1)
    grey = 0.5 * grey + 0.25 * np.r_[grey[0], grey[:-1]] + 0.25 * np.r_[grey[1:], grey[-1]]
    histogram = np.r_[hue, grey]
    return tuple((histogram / histogram.sum()).tolist())


def _restore_cpu_threads(predictor):
    """Backend setup can reset the limit; restore it before warmup and inference."""
    if predictor.device.type == "cpu":
        import torch

        if torch.get_num_threads() != 4:
            torch.set_num_threads(4)


def _restore_locked_architecture(model, entry: dict):
    """Restore audited non-tensor modules and reject constructor graph drift."""
    import torch

    for name in entry.get("identity_activations", []):
        parent_path, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        if attribute != "act" or type(getattr(parent, attribute, None)) not in (
            torch.nn.SiLU,
            torch.nn.Identity,
        ):
            raise ValueError("Unsupported locked activation override")
        setattr(parent, attribute, torch.nn.Identity())
    # Shared activation objects still have distinct paths in the source graph.
    modules = {
        name: type(layer).__module__ + "." + type(layer).__qualname__
        for name, layer in model.named_modules(remove_duplicate=False)
    }
    digest = hashlib.sha256(json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != entry.get("module_types_sha256"):
        raise ValueError("Reconstructed model architecture does not match the lock")


def load_safe_model(path: Path, expected_task: str):
    """Reconstruct a pinned runtime model from tensor data. Never open a .pt at runtime."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, PoseModel

    if path.suffix != ".safetensors":
        raise ValueError("Runtime requires verified safetensors artifacts")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata.get("format") != "stillwatch-tensors-v1" or metadata.get("task") != expected_task:
        raise ValueError("Unexpected artifact format or task")
    cls = PoseModel if expected_task == "pose" else DetectionModel
    cfg = json.loads(metadata["yaml"])
    # Architecture definitions are accepted only from locked, known artifacts.
    root_lock = Path(__file__).with_name("models.lock.json")
    if not root_lock.exists():
        root_lock = Path(__file__).resolve().parents[2] / "models.lock.json"
    sources = json.loads(root_lock.read_text())["models"].values()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = next(
        (
            m
            for m in sources
            if m["sha256"] == metadata.get("source_sha256")
            and m["task"] == expected_task
            and m.get("tensor_sha256") == digest
        ),
        None,
    )
    if entry is None:
        raise ValueError("Unrecognized or modified tensor artifact")
    model = cls(cfg=cfg, verbose=False)
    _restore_locked_architecture(model, entry)
    model.load_state_dict(load_file(path), strict=True)
    model.names = {int(k): v for k, v in json.loads(metadata["names"]).items()}
    model.eval()
    # Construct a wrapper without loading weights or performing a download.
    wrapper = YOLO.__new__(YOLO)
    import torch
    from ultralytics.engine.model import Model

    torch.nn.Module.__init__(wrapper)
    wrapper.callbacks = __import__(
        "ultralytics.utils.callbacks", fromlist=["get_default_callbacks"]
    ).get_default_callbacks()
    wrapper.predictor = None
    wrapper.model = model
    wrapper.trainer = None
    wrapper.ckpt = {}
    wrapper.cfg = None
    wrapper.ckpt_path = None
    wrapper.overrides = {"task": expected_task, "model": str(path), "verbose": False}
    wrapper.metrics = None
    wrapper.session = None
    wrapper.task = expected_task
    wrapper.model_name = str(path)
    assert isinstance(wrapper, Model)
    return wrapper


class Perception:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pose = None
        self.fire = None
        self.states = {"pose": "disabled", "fire": "disabled"}
        self.errors: dict[str, str] = {}

    def load(self):
        (self.settings.data_dir / "runtime").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self.settings.data_dir / "runtime"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        if self.settings.device == "cpu":
            # Idle inference workers should yield the CPU between source frames.
            os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            os.environ.setdefault("KMP_BLOCKTIME", "0")
            cv2.setNumThreads(1)
        try:
            import torch

            torch.set_num_threads(4)
        except ImportError:
            pass
        for route, enabled, path, task in (
            ("pose", self.settings.pose_enabled, self.settings.pose_weights, "pose"),
            ("fire", self.settings.fire_enabled, self.settings.fire_weights, "detect"),
        ):
            if not enabled:
                continue
            if not path.is_file():
                self.states[route] = "missing_weights"
                continue
            try:
                model = load_safe_model(path, task)
                expected = {0: "person"} if route == "pose" else {0: "smoke", 1: "fire"}
                if model.names != expected:
                    raise ValueError("Class map mismatch")
                model.add_callback("on_predict_start", _restore_cpu_threads)
                setattr(self, route, model)
                self.states[route] = "ready"
            except Exception as error:
                self.states[route] = "unavailable"
                self.errors[route] = type(error).__name__

    def reset(self):
        # Person association is owned by MedicalRules and reset with its epoch.
        pass

    def people(self, image: np.ndarray, ts: float) -> list[Observation]:
        if self.pose is None:
            return []
        results = self.pose.predict(
            image,
            conf=0.15,
            iou=0.5,
            imgsz=self.settings.inference_size,
            device=self.settings.device,
            verbose=False,
        )[0]
        if results.boxes is None:
            return []
        height, width = image.shape[:2]
        observations = []
        for i, box in enumerate(results.boxes):
            xyxy = box.xyxy[0].cpu().tolist()
            keypoints = results.keypoints.data[i].cpu().tolist() if results.keypoints is not None else []
            keypoints = visible_keypoints(keypoints, width, height)
            observations.append(
                Observation(
                    capture_ts=ts,
                    box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                    score=float(box.conf[0]),
                    label="person",
                    track_id=int(box.id[0]) if box.id is not None else None,
                    keypoints=keypoints,
                    torso_histogram=torso_histogram(image, keypoints),
                    limited=(xyxy[3] - xyxy[1]) < self.settings.minimum_person_height,
                    truncated=xyxy[0] <= 3 or xyxy[1] <= 3 or xyxy[2] >= width - 3 or xyxy[3] >= height - 3,
                )
            )
        return observations

    def hazards(self, image: np.ndarray, ts: float) -> list[Observation]:
        if self.fire is None:
            return []
        results = self.fire.predict(
            image,
            conf=self.settings.fire_confidence,
            imgsz=self.settings.inference_size,
            device=self.settings.device,
            verbose=False,
        )[0]
        if results.boxes is None:
            return []
        observations = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            observations.append(
                Observation(
                    capture_ts=ts,
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=float(box.conf[0]),
                    label=self.fire.names[int(box.cls[0])],
                )
            )
        return observations
