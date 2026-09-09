#!/usr/bin/env python3
"""Download the MediaPipe models if not already present."""

import os
import ssl
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Two models: hand landmarker (for custom gestures) and gesture recognizer
# (for accurate built-in gesture classification).
MODELS = {
    "gesture_recognizer": {
        "file": "gesture_recognizer.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "gesture_recognizer/gesture_recognizer/float16/latest/"
            "gesture_recognizer.task"
        ),
    },
    "hand_landmarker": {
        "file": "hand_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task"
        ),
    },
}


def _make_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that works on macOS fresh installs."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    if not ctx.get_ca_certs():
        ctx = ssl._create_unverified_context()
    return ctx


def ensure_model(name: str = "gesture_recognizer") -> str:
    """Download the model if missing; return the local path."""
    info = MODELS[name]
    path = os.path.join(MODEL_DIR, info["file"])

    if os.path.exists(path):
        print(f"[model] Already exists: {path}")
        return path

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[model] Downloading {info['file']} …")
    ctx = _make_ssl_context()
    with urllib.request.urlopen(info["url"], context=ctx) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    print(f"[model] Saved to {path}")
    return path


if __name__ == "__main__":
    for model_name in MODELS:
        ensure_model(model_name)
