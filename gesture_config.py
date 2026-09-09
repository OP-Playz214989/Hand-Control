"""
gesture_config.py — Persistent gesture-to-action configuration.

Manages a ``gesture_map.json`` file that stores the mapping from
recognised hand gestures to action IDs (from actions.py).

Functions:
    load_config()        → dict[str, str]   current mapping
    save_config(mapping) → None             persist to disk
    get_default_config() → dict[str, str]   factory defaults

The config file is auto-created with defaults on first run.
"""

from __future__ import annotations

import json
import os

_CONFIG_DIR  = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "gesture_map.json")

# Available gesture names (from MediaPipe's GestureRecognizer model)
AVAILABLE_GESTURES = [
    "open_palm",
    "fist",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "pointing_up",
    "i_love_you",
]


def get_default_config() -> dict[str, str]:
    """Return the factory-default gesture → action mapping."""
    return {
        "open_palm":   "stop",
        "fist":        "mute_jarvis",
        "thumbs_up":   "volume_up",
        "thumbs_down": "volume_down",
        "peace":       "screenshot",
        "pointing_up": "media_next",
        "i_love_you":  "media_play_pause",
    }


def load_config() -> dict[str, str]:
    """
    Load the gesture mapping from disk.

    If the config file doesn't exist yet, create it with defaults.
    If it's corrupted, fall back to defaults and overwrite.
    """
    if not os.path.exists(_CONFIG_FILE):
        defaults = get_default_config()
        save_config(defaults)
        return defaults

    try:
        with open(_CONFIG_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Config is not a dict")
        return data
    except (json.JSONDecodeError, ValueError):
        print("[config] ⚠  gesture_map.json is corrupted — resetting to defaults.")
        defaults = get_default_config()
        save_config(defaults)
        return defaults


def save_config(mapping: dict[str, str]) -> None:
    """Persist the gesture → action mapping to disk."""
    with open(_CONFIG_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[config] Saved gesture mappings to {_CONFIG_FILE}")
