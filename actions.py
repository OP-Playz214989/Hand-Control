"""
actions.py — Computer-control action registry for Mini Jarvis.

Every action is a callable registered in ACTION_REGISTRY.  Both the brain
(voice commands) and the gesture listener (hand gestures) resolve commands
through this single registry.

Actions use AppleScript (osascript) on macOS for system-level control.
Each action function receives an optional TTSEngine and returns a short
status string.

To add a new action:
    1. Write a function:  def my_action(tts=None) -> str
    2. Register it:       ACTION_REGISTRY["my_action"] = { ... }
"""

from __future__ import annotations

import platform
import subprocess
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tts_engine import TTSEngine


# ── Helpers ───────────────────────────────────────────────────────────

def _osascript(script: str) -> str:
    """Run an AppleScript snippet and return stdout."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _key_code(code: int, modifiers: str = "") -> None:
    """Simulate a keypress via AppleScript key code."""
    mod_clause = f" using {modifiers}" if modifiers else ""
    _osascript(
        f'tell application "System Events" to key code {code}{mod_clause}'
    )


def _keystroke(key: str, modifiers: str = "") -> None:
    """Simulate a keystroke via AppleScript."""
    mod_clause = f" using {modifiers}" if modifiers else ""
    _osascript(
        f'tell application "System Events" to keystroke "{key}"{mod_clause}'
    )


def _speak(tts: Optional["TTSEngine"], text: str) -> None:
    """Speak text if a TTS engine is available."""
    if tts:
        tts.speak(text)


# ── Volume ────────────────────────────────────────────────────────────

def volume_up(tts: Optional["TTSEngine"] = None) -> str:
    _osascript("set volume output volume ((output volume of (get volume settings)) + 10)")
    _speak(tts, "Volume up.")
    return "Volume increased."


def volume_down(tts: Optional["TTSEngine"] = None) -> str:
    _osascript("set volume output volume ((output volume of (get volume settings)) - 10)")
    _speak(tts, "Volume down.")
    return "Volume decreased."


def volume_mute(tts: Optional["TTSEngine"] = None) -> str:
    _osascript(
        "set curMute to output muted of (get volume settings)\n"
        "set volume output muted (not curMute)"
    )
    _speak(tts, "Toggled system mute.")
    return "System mute toggled."


# ── Brightness ────────────────────────────────────────────────────────

def brightness_up(tts: Optional["TTSEngine"] = None) -> str:
    # F2 key code = 144 (brightness up media key)
    _key_code(144)
    _speak(tts, "Brightness up.")
    return "Brightness increased."


def brightness_down(tts: Optional["TTSEngine"] = None) -> str:
    # F1 key code = 145 (brightness down media key)
    _key_code(145)
    _speak(tts, "Brightness down.")
    return "Brightness decreased."


# ── Media ─────────────────────────────────────────────────────────────

def media_play_pause(tts: Optional["TTSEngine"] = None) -> str:
    # Media play/pause key code = 16 with command
    # More reliable: use NX key simulation via osascript
    _key_code(16, "command down")  # Cmd+P as fallback
    _speak(tts, "Play pause.")
    return "Media play/pause toggled."


def media_next(tts: Optional["TTSEngine"] = None) -> str:
    _key_code(124, "command down")  # Cmd+Right (next track in many players)
    _speak(tts, "Next track.")
    return "Next track."


def media_previous(tts: Optional["TTSEngine"] = None) -> str:
    _key_code(123, "command down")  # Cmd+Left (previous track)
    _speak(tts, "Previous track.")
    return "Previous track."


# ── Window Management ─────────────────────────────────────────────────

def minimize_window(tts: Optional["TTSEngine"] = None) -> str:
    _keystroke("m", "command down")  # Cmd+M
    _speak(tts, "Window minimized.")
    return "Window minimized."


def close_window(tts: Optional["TTSEngine"] = None) -> str:
    _keystroke("w", "command down")  # Cmd+W
    _speak(tts, "Window closed.")
    return "Window closed."


def fullscreen(tts: Optional["TTSEngine"] = None) -> str:
    # Cmd+Ctrl+F toggles macOS fullscreen
    _keystroke("f", "{command down, control down}")
    _speak(tts, "Toggled fullscreen.")
    return "Fullscreen toggled."


# ── System ────────────────────────────────────────────────────────────

def lock_screen(tts: Optional["TTSEngine"] = None) -> str:
    _speak(tts, "Locking screen.")
    _keystroke("q", "{command down, control down}")  # Cmd+Ctrl+Q
    return "Screen locked."


def show_desktop(tts: Optional["TTSEngine"] = None) -> str:
    # F11 key code = 103, or Cmd+F3 (key code 99)
    _key_code(99, "command down")
    _speak(tts, "Showing desktop.")
    return "Show desktop."


def mission_control(tts: Optional["TTSEngine"] = None) -> str:
    # Control+Up (key code 126)
    _key_code(126, "control down")
    _speak(tts, "Mission Control.")
    return "Mission Control opened."


def spotlight(tts: Optional["TTSEngine"] = None) -> str:
    _keystroke(" ", "command down")  # Cmd+Space
    _speak(tts, "Spotlight.")
    return "Spotlight opened."


def screenshot_action(tts: Optional["TTSEngine"] = None) -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["screencapture", "-ix"])
        elif system == "Linux":
            subprocess.Popen(["gnome-screenshot", "-i"])
        elif system == "Windows":
            subprocess.Popen(["snippingtool"])
        _speak(tts, "Screenshot tool opened.")
        return "Screenshot tool opened."
    except FileNotFoundError:
        _speak(tts, "Screenshot tool not found.")
        return "Screenshot tool not found."


def notification_center(tts: Optional["TTSEngine"] = None) -> str:
    # Click the date/time area or use a trackpad gesture isn't easy via
    # AppleScript.  We'll use Notification Center's bundle ID.
    _osascript(
        'tell application "System Events" to tell process "ControlCenter"\n'
        '  click menu bar item "Clock" of menu bar 1\n'
        'end tell'
    )
    _speak(tts, "Notification Center.")
    return "Notification Center toggled."


def scroll_up(tts: Optional["TTSEngine"] = None) -> str:
    _osascript(
        'tell application "System Events" to key code 126 using {option down}'
    )  # Option+Up for page up behavior
    return "Scrolled up."


def scroll_down(tts: Optional["TTSEngine"] = None) -> str:
    _osascript(
        'tell application "System Events" to key code 125 using {option down}'
    )  # Option+Down for page down behavior
    return "Scrolled down."


# ── Jarvis Internal ───────────────────────────────────────────────────

def stop_action(tts: Optional["TTSEngine"] = None) -> str:
    _speak(tts, "Stopping.")
    return "Stopped."


def confirm_action(tts: Optional["TTSEngine"] = None) -> str:
    _speak(tts, "Confirmed.")
    return "Confirmed."


def cancel_action(tts: Optional["TTSEngine"] = None) -> str:
    _speak(tts, "Cancelled.")
    return "Cancelled."


def do_nothing(tts: Optional["TTSEngine"] = None) -> str:
    """Placeholder — used when a gesture slot is intentionally unassigned."""
    return ""


# ── Open App (voice-only, needs argument) ─────────────────────────────

def open_app(app_name: str, tts: Optional["TTSEngine"] = None) -> str:
    """Launch an application by name, OS-aware."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", "-a", app_name])
        elif system == "Linux":
            subprocess.Popen([app_name.lower()])
        elif system == "Windows":
            subprocess.Popen(["start", app_name], shell=True)
        reply = f"Opening {app_name}."
        _speak(tts, reply)
        return reply
    except Exception as exc:
        reply = f"Could not open {app_name}: {exc}"
        _speak(tts, reply)
        return reply


# ═══════════════════════════════════════════════════════════════════════
# ACTION REGISTRY
#
# Maps action IDs to metadata + callable.  The config menu and brain
# both discover available actions from this registry.
# ═══════════════════════════════════════════════════════════════════════

ActionEntry = dict  # {"name": str, "category": str, "fn": Callable}

ACTION_REGISTRY: dict[str, ActionEntry] = {
    # ── Volume ─────────────────────────────────
    "volume_up":        {"name": "Volume Up",         "category": "Volume",     "fn": volume_up},
    "volume_down":      {"name": "Volume Down",       "category": "Volume",     "fn": volume_down},
    "volume_mute":      {"name": "System Mute",       "category": "Volume",     "fn": volume_mute},
    # ── Brightness ─────────────────────────────
    "brightness_up":    {"name": "Brightness Up",     "category": "Brightness", "fn": brightness_up},
    "brightness_down":  {"name": "Brightness Down",   "category": "Brightness", "fn": brightness_down},
    # ── Media ──────────────────────────────────
    "media_play_pause": {"name": "Play / Pause",      "category": "Media",      "fn": media_play_pause},
    "media_next":       {"name": "Next Track",        "category": "Media",      "fn": media_next},
    "media_previous":   {"name": "Previous Track",    "category": "Media",      "fn": media_previous},
    # ── Window ─────────────────────────────────
    "minimize_window":  {"name": "Minimize Window",   "category": "Window",     "fn": minimize_window},
    "close_window":     {"name": "Close Window",      "category": "Window",     "fn": close_window},
    "fullscreen":       {"name": "Toggle Fullscreen", "category": "Window",     "fn": fullscreen},
    # ── System ─────────────────────────────────
    "lock_screen":      {"name": "Lock Screen",       "category": "System",     "fn": lock_screen},
    "show_desktop":     {"name": "Show Desktop",      "category": "System",     "fn": show_desktop},
    "mission_control":  {"name": "Mission Control",   "category": "System",     "fn": mission_control},
    "spotlight":        {"name": "Spotlight",          "category": "System",     "fn": spotlight},
    "screenshot":       {"name": "Screenshot",        "category": "System",     "fn": screenshot_action},
    "notification_center": {"name": "Notification Center", "category": "System", "fn": notification_center},
    "scroll_up":        {"name": "Scroll Up",         "category": "System",     "fn": scroll_up},
    "scroll_down":      {"name": "Scroll Down",       "category": "System",     "fn": scroll_down},
    # ── Jarvis ─────────────────────────────────
    "stop":             {"name": "Stop",              "category": "Jarvis",     "fn": stop_action},
    "mute_jarvis":      {"name": "Mute Jarvis",       "category": "Jarvis",     "fn": None},  # handled by brain
    "confirm":          {"name": "Confirm",           "category": "Jarvis",     "fn": confirm_action},
    "cancel":           {"name": "Cancel",            "category": "Jarvis",     "fn": cancel_action},
    "do_nothing":       {"name": "Do Nothing",        "category": "Jarvis",     "fn": do_nothing},
}

# ── Voice keyword → action ID mapping ────────────────────────────────
# These let the user say a natural phrase and resolve to an action.
VOICE_KEYWORDS: dict[str, str] = {
    "stop":               "stop",
    "mute":               "mute_jarvis",
    "unmute":             "mute_jarvis",
    "confirm":            "confirm",
    "cancel":             "cancel",
    "screenshot":         "screenshot",
    "next":               "media_next",
    "previous":           "media_previous",
    "volume up":          "volume_up",
    "volume down":        "volume_down",
    "louder":             "volume_up",
    "quieter":            "volume_down",
    "brighter":           "brightness_up",
    "dimmer":             "brightness_down",
    "brightness up":      "brightness_up",
    "brightness down":    "brightness_down",
    "play":               "media_play_pause",
    "pause":              "media_play_pause",
    "play pause":         "media_play_pause",
    "minimize":           "minimize_window",
    "close window":       "close_window",
    "fullscreen":         "fullscreen",
    "full screen":        "fullscreen",
    "lock":               "lock_screen",
    "lock screen":        "lock_screen",
    "show desktop":       "show_desktop",
    "desktop":            "show_desktop",
    "mission control":    "mission_control",
    "spotlight":          "spotlight",
    "search":             "spotlight",
    "scroll up":          "scroll_up",
    "scroll down":        "scroll_down",
    "notification center": "notification_center",
    "notifications":      "notification_center",
}
