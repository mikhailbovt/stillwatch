import builtins
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import stillwatch.perception as perception_module
from stillwatch.buffer import FrameBuffer
from stillwatch.capture import VideoCapture
from stillwatch.perception import Perception


@pytest.mark.parametrize("source", ["rtsp://localhost/video", "rtsps://localhost/video", "local.mp4", 0])
def test_decoder_workers_are_bounded_without_changing_usb_backend(settings, monkeypatch, source):
    settings.source = source
    calls = []
    decoder = object()

    def open_capture(*args):
        calls.append(args)
        return decoder

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    capture = VideoCapture(settings, FrameBuffer(30, 1024**2, 10))
    assert capture._open() is decoder
    assert len(calls) == 1
    args = calls[0]
    if isinstance(source, int):
        assert args == (source,)
    else:
        assert args[:2] == (source, cv2.CAP_FFMPEG)
        options = dict(zip(args[2][::2], args[2][1::2], strict=True))
        assert options[cv2.CAP_PROP_N_THREADS] == 2
        if source.startswith(("rtsp://", "rtsps://")):
            assert options[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC] == 3000
            assert options[cv2.CAP_PROP_READ_TIMEOUT_MSEC] == 3000


@pytest.mark.parametrize("existing", [False, True])
def test_cpu_idle_defaults_precede_torch_import_and_preserve_operator_values(settings, monkeypatch, existing):
    import os

    expected = (
        {"OMP_WAIT_POLICY": "ACTIVE", "KMP_BLOCKTIME": "25"}
        if existing
        else {
            "OMP_WAIT_POLICY": "PASSIVE",
            "KMP_BLOCKTIME": "0",
        }
    )
    for key, value in expected.items():
        if existing:
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    imports, workers, model_workers = [], [], []
    real_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "torch":
            imports.append({key: os.environ.get(key) for key in expected})
            return SimpleNamespace(set_num_threads=model_workers.append)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    monkeypatch.setattr(cv2, "setNumThreads", workers.append)
    Perception(settings).load()
    assert imports == [expected]
    assert workers == [1] and model_workers == [4]


@pytest.mark.parametrize("routes", [("pose",), ("fire",), ("pose", "fire")])
@pytest.mark.parametrize("backend_workers", [4, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inference_worker_limit_survives_predictor_setup(
    settings, monkeypatch, routes, backend_workers, device
):
    workers = SimpleNamespace(count=4)
    torch = SimpleNamespace(
        get_num_threads=lambda: workers.count,
        set_num_threads=lambda value: setattr(workers, "count", value),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    models = {}

    class Model:
        def __init__(self, task):
            self.names = {0: "person"} if task == "pose" else {0: "smoke", 1: "fire"}
            self.existing_callback_calls = 0
            self.callbacks = [self.existing_callback]
            self.warmup_workers = []
            self.inference_workers = []

        def existing_callback(self, predictor):
            self.existing_callback_calls += 1

        def add_callback(self, event, callback):
            assert event == "on_predict_start"
            self.callbacks.append(callback)

        def predict(self, *args, **kwargs):
            # First setup and later backend recreation can reset the global cap.
            torch.set_num_threads(backend_workers)
            predictor = SimpleNamespace(device=SimpleNamespace(type=device))
            for callback in self.callbacks:
                callback(predictor)
            self.warmup_workers.append(torch.get_num_threads())
            self.inference_workers.append(torch.get_num_threads())
            return [SimpleNamespace(boxes=[])]

    def load(path, task):
        model = Model(task)
        models["pose" if task == "pose" else "fire"] = model
        return model

    monkeypatch.setattr(perception_module, "load_safe_model", load)
    settings.device = device
    settings.pose_enabled, settings.fire_enabled = "pose" in routes, "fire" in routes
    settings.pose_weights.write_bytes(b"mock")
    settings.fire_weights.write_bytes(b"mock")
    perception = Perception(settings)
    perception.load()
    image = np.zeros((16, 16, 3), np.uint8)
    for _ in range(2):
        if "pose" in routes:
            assert perception.people(image, 0) == []
        if "fire" in routes:
            assert perception.hazards(image, 0) == []
    expected = 4 if device == "cpu" else backend_workers
    for route in routes:
        model = models[route]
        assert perception.states[route] == "ready"
        assert model.warmup_workers == model.inference_workers == [expected, expected]
        assert model.existing_callback_calls == 2
