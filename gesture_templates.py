"""
gesture_templates.py — Custom gesture recording, storage, and matching.

Records hand landmark poses as normalised 63-dimensional vectors
(21 landmarks × 3 coords), stores them in ``custom_gestures.json``,
and matches incoming hand poses against saved templates using
Euclidean distance.

Normalisation makes templates invariant to hand position, size, and
(to a degree) camera distance.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional


_DIR  = os.path.dirname(os.path.abspath(__file__))
_FILE = os.path.join(_DIR, "custom_gestures.json")

# How close a pose must be to a template to count as a match.
# Lower = stricter.  2.5 works well for static hand poses.
MATCH_THRESHOLD = 2.5

# Number of frames to sample when recording a gesture (~2 s at 30 fps).
RECORD_FRAMES = 50


# ── Normalisation ─────────────────────────────────────────────────────

def normalize_landmarks(landmarks) -> list[float]:
    """
    Flatten 21 MediaPipe hand landmarks into a 63-element vector,
    normalised relative to the wrist (landmark 0) and scaled by
    the wrist-to-middle-MCP distance so hand size doesn't matter.
    """
    wrist = landmarks[0]

    # Reference distance: wrist → middle finger MCP (landmark 9)
    ref = math.sqrt(
        (landmarks[9].x - wrist.x) ** 2
        + (landmarks[9].y - wrist.y) ** 2
        + (landmarks[9].z - wrist.z) ** 2
    )
    if ref < 1e-6:
        ref = 1e-6

    vec: list[float] = []
    for lm in landmarks:
        vec.extend([
            (lm.x - wrist.x) / ref,
            (lm.y - wrist.y) / ref,
            (lm.z - wrist.z) / ref,
        ])
    return vec


def _euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ── Template store ────────────────────────────────────────────────────

def load_templates() -> dict[str, list[float]]:
    """Load custom gesture templates from disk.  Returns name → vector."""
    if not os.path.exists(_FILE):
        return {}
    try:
        with open(_FILE, "r") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(v, list)}
    except (json.JSONDecodeError, ValueError):
        return {}


def save_templates(templates: dict[str, list[float]]) -> None:
    """Persist custom gesture templates to disk."""
    with open(_FILE, "w") as f:
        json.dump(templates, f)
    print(f"[templates] Saved {len(templates)} custom gesture(s).")


def add_template(name: str, vector: list[float]) -> None:
    """Add or overwrite a single template and save."""
    templates = load_templates()
    templates[name] = vector
    save_templates(templates)


def delete_template(name: str) -> None:
    """Remove a template by name and save."""
    templates = load_templates()
    templates.pop(name, None)
    save_templates(templates)


# ── Matching ──────────────────────────────────────────────────────────

def match_custom_gesture(
    landmarks,
    templates: dict[str, list[float]] | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> Optional[str]:
    """
    Compare current hand landmarks against all saved custom templates.

    Returns the name of the closest match if within *threshold*,
    or ``None`` if nothing matches.
    """
    if templates is None:
        templates = load_templates()
    if not templates:
        return None

    current = normalize_landmarks(landmarks)

    best_name: Optional[str] = None
    best_dist = threshold

    for name, template_vec in templates.items():
        if len(template_vec) != len(current):
            continue
        dist = _euclidean(current, template_vec)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


# ── Recording helper ──────────────────────────────────────────────────

class GestureRecorder:
    """
    Accumulates landmark samples over multiple frames and produces
    an averaged template vector.

    Usage:
        rec = GestureRecorder()
        while not rec.is_done():
            rec.add_sample(landmarks)
        template = rec.get_template()
    """

    def __init__(self, num_frames: int = RECORD_FRAMES):
        self._target = num_frames
        self._samples: list[list[float]] = []

    @property
    def progress(self) -> float:
        """0.0 → 1.0"""
        return min(len(self._samples) / self._target, 1.0)

    def is_done(self) -> bool:
        return len(self._samples) >= self._target

    def add_sample(self, landmarks) -> None:
        """Add one frame's worth of normalised landmarks."""
        vec = normalize_landmarks(landmarks)
        self._samples.append(vec)

    def get_template(self) -> list[float]:
        """Return the averaged template vector."""
        if not self._samples:
            return []
        dim = len(self._samples[0])
        avg = [0.0] * dim
        for sample in self._samples:
            for i in range(dim):
                avg[i] += sample[i]
        n = len(self._samples)
        return [v / n for v in avg]

    def reset(self) -> None:
        self._samples.clear()
