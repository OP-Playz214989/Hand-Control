"""
side_panel.py — Status panel for the hand-tracking virtual mouse.

Displayed alongside the webcam feed. Shows:
  • Current hand state (Moving / Click / Scroll / Drag / Idle)
  • Gesture legend with colour-coded labels
  • Scroll sensitivity slider (draggable)
  • Toggle: enable/disable hand mouse
  • Instructions
"""

from __future__ import annotations

import math
import numpy as np
import cv2
from typing import Optional

from hand_mouse import HandMouse, HandState


# ── Layout constants ──────────────────────────────────────────────────
PANEL_WIDTH     = 320
_HEADER_H       = 55
_PAD            = 14
_FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ── Colours (BGR) ─────────────────────────────────────────────────────
_BG             = (25, 25, 30)
_HEADER_BG      = (45, 35, 55)
_SECTION_BG     = (35, 32, 40)
_TEXT_WHITE      = (240, 240, 240)
_TEXT_DIM        = (140, 140, 150)
_TEXT_ACCENT     = (130, 255, 130)
_DIVIDER         = (60, 55, 70)
_STATE_BG        = (40, 38, 48)
_BTN_BG          = (60, 55, 80)
_BTN_HOV         = (90, 80, 120)
_BTN_ACTIVE      = (60, 130, 60)
_BTN_PAUSED      = (60, 60, 130)
_SLIDER_TRACK    = (55, 50, 65)
_SLIDER_FILL     = (130, 100, 200)
_SLIDER_KNOB     = (200, 180, 255)

# State → colour + icon
_STATE_COLORS: dict[HandState, tuple[str, tuple[int, int, int]]] = {
    HandState.IDLE:      ("IDLE",        (120, 120, 120)),
    HandState.MOVING:    ("MOVING",  (0, 255, 100)),
    HandState.L_CLICK:   ("LEFT CLICK",  (0, 200, 255)),
    HandState.R_CLICK:   ("RIGHT CLICK", (255, 100, 100)),
    HandState.SCROLLING: ("SCROLL",  (255, 200, 0)),
    HandState.DRAGGING:  ("DRAGGING",    (200, 100, 255)),
}

# Gesture legend entries: (description, gesture, colour)
_LEGEND = [
    ("Move cursor",   "Index finger up",        (0, 255, 100)),
    ("Left click",    "Thumb + Index pinch",     (0, 200, 255)),
    ("Right click",   "Thumb + Middle pinch",    (255, 100, 100)),
    ("Scroll",        "Index + Middle up/down",  (255, 200, 0)),
    ("Drag & drop",   "Pinch + hold + move",     (200, 100, 255)),
]

# ── Slider range (log scale) ─────────────────────────────────────────
_SENS_MIN  = 0.01    # most sensitive (fastest scroll)
_SENS_MAX  = 1.0     # least sensitive (slowest scroll)


def _sens_to_pct(val: float) -> float:
    """Convert sensitivity value to slider percentage [0, 1] via log scale."""
    val = max(_SENS_MIN, min(_SENS_MAX, val))
    return (math.log(val) - math.log(_SENS_MIN)) / (math.log(_SENS_MAX) - math.log(_SENS_MIN))


def _pct_to_sens(pct: float) -> float:
    """Convert slider percentage [0, 1] to sensitivity value via log scale."""
    pct = max(0.0, min(1.0, pct))
    return math.exp(math.log(_SENS_MIN) + pct * (math.log(_SENS_MAX) - math.log(_SENS_MIN)))


class SidePanel:
    """Renders the status/legend panel alongside the webcam feed."""

    def __init__(self):
        self._state: HandState = HandState.IDLE
        self._status_msg: str = "WAITING"
        self._hover_btn: str = ""
        self._toggle_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

        # Slider state
        self._slider_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._dragging_slider: bool = False

        # Click flash animation
        self._flash_state: Optional[HandState] = None
        self._flash_until: float = 0.0

    # ── Public API ────────────────────────────────────────────────

    def update_state(self, state: HandState, msg: str) -> None:
        """Called each frame by gesture_listener."""
        self._state = state
        self._status_msg = msg

    def render(self, height: int, current_time: float,
               mouse: Optional[HandMouse] = None) -> np.ndarray:
        """Return a (height × PANEL_WIDTH × 3) uint8 image of the panel."""
        panel = np.full((height, PANEL_WIDTH, 3), _BG, dtype=np.uint8)

        y = 0

        # ── Header ────────────────────────────────────────────────
        cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, _HEADER_H), _HEADER_BG, -1)
        cv2.putText(panel, "HAND MOUSE", (_PAD, 24),
                    _FONT, 0.6, _TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(panel, "Virtual cursor control", (_PAD, 44),
                    _FONT, 0.35, _TEXT_DIM, 1, cv2.LINE_AA)
        y = _HEADER_H + 8

        # ── Current State Display ─────────────────────────────────
        state_h = 70
        cv2.rectangle(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + state_h), _STATE_BG, -1)
        cv2.rectangle(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + state_h), _DIVIDER, 1)

        label, color = _STATE_COLORS.get(self._state, ("?", (200, 200, 200)))

        # State indicator dot
        cv2.circle(panel, (_PAD + 18, y + 28), 8, color, -1)
        cv2.circle(panel, (_PAD + 18, y + 28), 10, color, 2)

        # State label (large)
        cv2.putText(panel, label, (_PAD + 36, y + 34),
                    _FONT, 0.7, color, 2, cv2.LINE_AA)

        # Status message (small)
        cv2.putText(panel, self._status_msg, (_PAD + 36, y + 56),
                    _FONT, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)

        y += state_h + 16

        # ── Gesture Legend ────────────────────────────────────────
        cv2.putText(panel, "GESTURES", (_PAD + 2, y + 2),
                    _FONT, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)
        y += 14

        # Divider line
        cv2.line(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y), _DIVIDER, 1)
        y += 8

        for action, gesture, color in _LEGEND:
            # Colour bar
            cv2.rectangle(panel, (_PAD + 2, y + 2), (_PAD + 6, y + 34), color, -1)

            # Action name
            cv2.putText(panel, action, (_PAD + 14, y + 16),
                        _FONT, 0.45, _TEXT_WHITE, 1, cv2.LINE_AA)

            # Gesture description
            cv2.putText(panel, gesture, (_PAD + 14, y + 32),
                        _FONT, 0.33, _TEXT_DIM, 1, cv2.LINE_AA)

            y += 42

        y += 8

        # ── Scroll Sensitivity Slider ─────────────────────────────
        cv2.putText(panel, "SCROLL SPEED", (_PAD + 2, y + 2),
                    _FONT, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)

        # Show current multiplier value
        if mouse:
            val_text = f"{1.0 / mouse.scroll_sensitivity:.0f}x"
        else:
            val_text = ""
        tw = cv2.getTextSize(val_text, _FONT, 0.38, 1)[0][0]
        cv2.putText(panel, val_text, (PANEL_WIDTH - _PAD - tw - 2, y + 2),
                    _FONT, 0.38, _SLIDER_KNOB, 1, cv2.LINE_AA)
        y += 14
        cv2.line(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y), _DIVIDER, 1)
        y += 10

        # Track
        track_x1 = _PAD + 8
        track_x2 = PANEL_WIDTH - _PAD - 8
        track_y  = y + 8
        track_h  = 6
        self._slider_rect = (track_x1, track_y - 8, track_x2, track_y + track_h + 8)

        cv2.rectangle(panel, (track_x1, track_y),
                      (track_x2, track_y + track_h), _SLIDER_TRACK, -1)
        cv2.rectangle(panel, (track_x1, track_y),
                      (track_x2, track_y + track_h), _DIVIDER, 1)

        # Filled portion + knob
        if mouse:
            pct = 1.0 - _sens_to_pct(mouse.scroll_sensitivity)  # invert: left=slow, right=fast
        else:
            pct = 0.5
        fill_x = int(track_x1 + pct * (track_x2 - track_x1))
        cv2.rectangle(panel, (track_x1, track_y),
                      (fill_x, track_y + track_h), _SLIDER_FILL, -1)

        # Knob
        knob_cy = track_y + track_h // 2
        cv2.circle(panel, (fill_x, knob_cy), 9, _SLIDER_KNOB, -1)
        cv2.circle(panel, (fill_x, knob_cy), 9, _DIVIDER, 2)

        # Labels
        cv2.putText(panel, "Slow", (track_x1 - 2, track_y + track_h + 18),
                    _FONT, 0.3, _TEXT_DIM, 1, cv2.LINE_AA)
        fast_tw = cv2.getTextSize("Fast", _FONT, 0.3, 1)[0][0]
        cv2.putText(panel, "Fast", (track_x2 - fast_tw + 2, track_y + track_h + 18),
                    _FONT, 0.3, _TEXT_DIM, 1, cv2.LINE_AA)

        y += 40

        # ── Divider ──────────────────────────────────────────────
        cv2.line(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y), _DIVIDER, 1)
        y += 16

        # ── Toggle Button ────────────────────────────────────────
        btn_h = 38
        btn_y1 = y
        btn_y2 = y + btn_h
        self._toggle_rect = (_PAD, btn_y1, PANEL_WIDTH - _PAD, btn_y2)

        bg = _BTN_HOV if self._hover_btn == "toggle" else _BTN_BG
        cv2.rectangle(panel, (_PAD, btn_y1), (PANEL_WIDTH - _PAD, btn_y2), bg, -1)
        cv2.rectangle(panel, (_PAD, btn_y1), (PANEL_WIDTH - _PAD, btn_y2), _DIVIDER, 1)

        toggle_text = "Press [P] to pause"
        cv2.putText(panel, toggle_text,
                    (PANEL_WIDTH // 2 - 70, btn_y2 - 12),
                    _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)

        y = btn_y2 + 16

        # ── Keyboard shortcuts ────────────────────────────────────
        cv2.putText(panel, "SHORTCUTS", (_PAD + 2, y + 2),
                    _FONT, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)
        y += 14
        cv2.line(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y), _DIVIDER, 1)
        y += 12

        shortcuts = [
            ("[P]", "Pause / resume mouse"),
            ("[Q]", "Quit application"),
            ("[+]", "Increase scroll speed"),
            ("[-]", "Decrease scroll speed"),
        ]
        for key, desc in shortcuts:
            cv2.putText(panel, key, (_PAD + 6, y + 4),
                        _FONT, 0.4, _TEXT_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(panel, desc, (_PAD + 40, y + 4),
                        _FONT, 0.38, _TEXT_DIM, 1, cv2.LINE_AA)
            y += 22

        return panel

    # ── Mouse interaction ─────────────────────────────────────────

    def on_mouse(self, event: int, x: int, y: int, cam_width: int,
                 current_time: float, mouse: HandMouse) -> None:
        """Handle mouse events on the panel area."""
        px = x - cam_width
        if px < 0:
            self._hover_btn = ""
            if event == cv2.EVENT_LBUTTONUP:
                self._dragging_slider = False
            return

        # ── Slider drag ──────────────────────────────────────────
        sr = self._slider_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            if sr[0] <= px <= sr[2] and sr[1] <= y <= sr[3]:
                self._dragging_slider = True
                self._apply_slider(px, mouse)
                return

        if event == cv2.EVENT_MOUSEMOVE and self._dragging_slider:
            self._apply_slider(px, mouse)
            return

        if event == cv2.EVENT_LBUTTONUP:
            self._dragging_slider = False

        # ── Button hover / click ─────────────────────────────────
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover_btn = ""
            tr = self._toggle_rect
            if tr[0] <= px <= tr[2] and tr[1] <= y <= tr[3]:
                self._hover_btn = "toggle"

        elif event == cv2.EVENT_LBUTTONDOWN:
            tr = self._toggle_rect
            if tr[0] <= px <= tr[2] and tr[1] <= y <= tr[3]:
                mouse.enabled = not mouse.enabled
                print(f"[panel] Mouse {'enabled' if mouse.enabled else 'paused'}")

    def _apply_slider(self, px: int, mouse: HandMouse) -> None:
        """Update scroll sensitivity from slider knob position."""
        sr = self._slider_rect
        track_x1 = sr[0]
        track_x2 = sr[2]
        pct = (px - track_x1) / max(1, track_x2 - track_x1)
        pct = max(0.0, min(1.0, pct))
        # Invert: right = fast (low sensitivity value)
        mouse.scroll_sensitivity = _pct_to_sens(1.0 - pct)
