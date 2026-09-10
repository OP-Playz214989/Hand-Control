"""
side_panel.py — Professional HUD status panel for Mini Jarvis.

Displays a modern, sleek dark-mode interface alongside the webcam feed:
  • Header with live status beacon and Groq AI badge
  • Active tracking state card with dynamic glowing accents
  • Gesture control legend with color-coded badges
  • Interactive scroll speed precision slider
  • AI voice assistant status card
  • Interactive pause/resume button and keyboard hotkey guide
"""

from __future__ import annotations

import math
import numpy as np
import cv2
from typing import Optional

from hand_mouse import HandMouse, HandState


# ── Dimensions & Layout ───────────────────────────────────────────────
PANEL_WIDTH     = 340
_PAD            = 14
_FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ── Modern Dark Palette (BGR) ─────────────────────────────────────────
_BG             = (18, 15, 13)       # Deep graphite obsidian
_CARD_BG        = (30, 24, 21)       # Glass card surface
_CARD_BORDER    = (50, 42, 38)       # Subtle card outline
_CARD_HOVER     = (45, 36, 32)       # Card hover state
_DIVIDER         = (42, 35, 32)       # Divider rule
_TEXT_WHITE      = (250, 250, 250)    # Primary headers & text
_TEXT_MUTED      = (165, 160, 155)    # Secondary text
_TEXT_DIM        = (115, 110, 105)    # Small labels
_TEXT_EMERALD    = (100, 235, 0)      # Ready / Success
_TEXT_VIOLET     = (230, 120, 190)    # AI Agent badge

# Slider track & knob
_SLIDER_TRACK    = (40, 32, 28)
_SLIDER_FILL     = (0, 180, 240)      # Glowing amber
_SLIDER_KNOB     = (255, 230, 180)    # Bright knob
_SLIDER_BORDER   = (60, 50, 45)

# Buttons
_BTN_BG          = (40, 32, 28)
_BTN_HOV         = (60, 48, 42)
_BTN_ACTIVE_BORDER = (100, 235, 0)
_BTN_PAUSED_BORDER = (80, 80, 230)

# State → (Header Label, Accent BGR, Subtitle)
_STATE_THEME: dict[HandState, tuple[str, tuple[int, int, int], str]] = {
    HandState.IDLE:      ("STANDBY",      (130, 125, 120), "Awaiting hand in camera view"),
    HandState.MOVING:    ("TRACKING",     (100, 235, 0),   "Following index fingertip"),
    HandState.L_CLICK:   ("LEFT CLICK",   (255, 210, 0),   "Pinch click registered"),
    HandState.R_CLICK:   ("RIGHT CLICK",  (100, 100, 255), "Secondary click registered"),
    HandState.SCROLLING: ("4-WAY SCROLL", (0, 195, 255),   "Two-finger smooth scroll"),
    HandState.DRAGGING:  ("DRAGGING",     (235, 95, 195),  "Pinch held — moving item"),
}

# Gesture legend entries: (description, gesture, colour)
_LEGEND = [
    ("Move Cursor",   "Index finger pointing",    (100, 235, 0)),
    ("Left Click",    "Thumb + Index pinch",     (255, 210, 0)),
    ("Right Click",   "Thumb + Middle pinch",    (100, 100, 255)),
    ("4-Way Scroll",  "Two fingers swipe",       (0, 195, 255)),
    ("Drag & Drop",   "Pinch + hold + drag",     (235, 95, 195)),
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


def draw_rounded_rect(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int],
                      color: tuple[int, int, int], radius: int = 8,
                      thickness: int = -1) -> None:
    """Draw an anti-aliased rounded rectangle in OpenCV."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    r = min(radius, w // 2, h // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


class SidePanel:
    """Renders the sleek status/legend panel alongside the webcam feed."""

    def __init__(self):
        self._state: HandState = HandState.IDLE
        self._status_msg: str = "WAITING"
        self._hover_btn: str = ""
        self._toggle_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

        # Slider state
        self._slider_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._dragging_slider: bool = False

    # ── Public API ────────────────────────────────────────────────

    def update_state(self, state: HandState, msg: str) -> None:
        """Called each frame by gesture_listener."""
        self._state = state
        self._status_msg = msg

    def render(self, height: int, current_time: float,
               mouse: Optional[HandMouse] = None) -> np.ndarray:
        """Return a (height × PANEL_WIDTH × 3) uint8 image of the panel."""
        panel = np.full((height, PANEL_WIDTH, 3), _BG, dtype=np.uint8)

        # Determine scale & layout padding based on window height
        is_compact = height < 550
        y = 12

        # ══════════════════════════════════════════════════════════
        #  1. HEADER CARD: Brand, Status Beacon & Groq Badge
        # ══════════════════════════════════════════════════════════
        header_h = 48 if is_compact else 54
        draw_rounded_rect(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + header_h), _CARD_BG, radius=10, thickness=-1)
        draw_rounded_rect(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + header_h), _CARD_BORDER, radius=10, thickness=1)

        # Pulsing beacon dot
        pulse_alpha = (math.sin(current_time * 4.0) + 1.0) / 2.0
        beacon_color = _TEXT_EMERALD if (mouse and mouse.enabled) else (80, 80, 230)
        beacon_cx = _PAD + 16
        beacon_cy = y + header_h // 2
        # Outer halo
        halo_r = int(7 + 3 * pulse_alpha)
        halo_color = (int(beacon_color[0] * 0.35), int(beacon_color[1] * 0.35), int(beacon_color[2] * 0.35))
        cv2.circle(panel, (beacon_cx, beacon_cy), halo_r, halo_color, -1, cv2.LINE_AA)
        cv2.circle(panel, (beacon_cx, beacon_cy), 5, beacon_color, -1, cv2.LINE_AA)

        # Brand Title
        cv2.putText(panel, "MINI JARVIS", (_PAD + 28, y + 21 if is_compact else y + 23),
                    _FONT, 0.55, _TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(panel, "AI Voice & Hand Control", (_PAD + 28, y + 37 if is_compact else y + 42),
                    _FONT, 0.32, _TEXT_MUTED, 1, cv2.LINE_AA)

        # Groq Agent Badge (Pill)
        badge_w, badge_h = 68, 20
        badge_x2 = PANEL_WIDTH - _PAD - 10
        badge_x1 = badge_x2 - badge_w
        badge_y1 = y + (header_h - badge_h) // 2
        badge_y2 = badge_y1 + badge_h
        draw_rounded_rect(panel, (badge_x1, badge_y1), (badge_x2, badge_y2), (48, 28, 44), radius=6, thickness=-1)
        draw_rounded_rect(panel, (badge_x1, badge_y1), (badge_x2, badge_y2), (90, 50, 80), radius=6, thickness=1)
        cv2.putText(panel, "GROQ AI", (badge_x1 + 10, badge_y2 - 6),
                    _FONT, 0.33, (240, 160, 220), 1, cv2.LINE_AA)

        y += header_h + (8 if is_compact else 12)

        # ══════════════════════════════════════════════════════════
        #  2. ACTIVE STATE CARD
        # ══════════════════════════════════════════════════════════
        state_h = 58 if is_compact else 66
        label, accent, desc = _STATE_THEME.get(self._state, ("STANDBY", (130, 125, 120), "Waiting for hand"))

        # Card body
        draw_rounded_rect(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + state_h), _CARD_BG, radius=10, thickness=-1)
        # Highlight border if active
        border_col = accent if self._state != HandState.IDLE else _CARD_BORDER
        draw_rounded_rect(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + state_h), border_col, radius=10, thickness=1)

        # Left accent vertical indicator strip
        draw_rounded_rect(panel, (_PAD + 2, y + 6), (_PAD + 6, y + state_h - 6), accent, radius=2, thickness=-1)

        # State Pill Badge
        cv2.putText(panel, label, (_PAD + 16, y + 26 if is_compact else y + 30),
                    _FONT, 0.62, accent, 2, cv2.LINE_AA)

        # Subtitle / dynamic status
        sub_text = self._status_msg if self._status_msg and self._status_msg != label else desc
        cv2.putText(panel, sub_text, (_PAD + 16, y + 46 if is_compact else y + 52),
                    _FONT, 0.33, _TEXT_MUTED, 1, cv2.LINE_AA)

        y += state_h + (8 if is_compact else 12)

        # ══════════════════════════════════════════════════════════
        #  3. GESTURE CONTROLS LEGEND
        # ══════════════════════════════════════════════════════════
        cv2.putText(panel, "GESTURE CONTROLS", (_PAD + 4, y + 2),
                    _FONT, 0.34, _TEXT_DIM, 1, cv2.LINE_AA)
        y += 10
        cv2.line(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y), _DIVIDER, 1)
        y += 8

        row_h = 24 if is_compact else 27
        for action, gesture, color in _LEGEND:
            # Indicator dot
            dot_y = y + row_h // 2 - 1
            cv2.circle(panel, (_PAD + 8, dot_y), 4, color, -1, cv2.LINE_AA)

            # Action name
            cv2.putText(panel, action, (_PAD + 20, dot_y + 4),
                        _FONT, 0.40, _TEXT_WHITE, 1, cv2.LINE_AA)

            # Gesture description
            tw = cv2.getTextSize(gesture, _FONT, 0.31, 1)[0][0]
            cv2.putText(panel, gesture, (PANEL_WIDTH - _PAD - tw - 4, dot_y + 4),
                        _FONT, 0.31, _TEXT_MUTED, 1, cv2.LINE_AA)

            y += row_h

        y += 4 if is_compact else 8

        # ══════════════════════════════════════════════════════════
        #  4. SCROLL SPEED PRECISION SLIDER
        # ══════════════════════════════════════════════════════════
        cv2.putText(panel, "SCROLL SENSITIVITY", (_PAD + 4, y + 2),
                    _FONT, 0.34, _TEXT_DIM, 1, cv2.LINE_AA)

        # Current multiplier pill badge
        if mouse:
            multiplier = 1.0 / mouse.scroll_sensitivity
            val_text = f"{multiplier:.0f}x"
            if abs(multiplier - 10.0) < 1.5:
                val_text += " (DEFAULT)"
            elif multiplier >= 18:
                val_text += " (SWIFT)"
        else:
            val_text = "10x"

        tw = cv2.getTextSize(val_text, _FONT, 0.33, 1)[0][0]
        cv2.putText(panel, val_text, (PANEL_WIDTH - _PAD - tw - 4, y + 2),
                    _FONT, 0.33, _SLIDER_FILL, 1, cv2.LINE_AA)

        y += 10
        cv2.line(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y), _DIVIDER, 1)
        y += 8

        # Slider track
        track_x1 = _PAD + 10
        track_x2 = PANEL_WIDTH - _PAD - 10
        track_y  = y + 6
        track_h  = 6
        self._slider_rect = (track_x1, track_y - 8, track_x2, track_y + track_h + 8)

        # Background track
        draw_rounded_rect(panel, (track_x1, track_y), (track_x2, track_y + track_h), _SLIDER_TRACK, radius=3, thickness=-1)

        # Filled progress portion
        if mouse:
            pct = 1.0 - _sens_to_pct(mouse.scroll_sensitivity)
        else:
            pct = 0.5
        fill_x = int(track_x1 + pct * (track_x2 - track_x1))
        fill_x = max(track_x1, min(track_x2, fill_x))
        if fill_x > track_x1:
            draw_rounded_rect(panel, (track_x1, track_y), (fill_x, track_y + track_h), _SLIDER_FILL, radius=3, thickness=-1)

        # Slider Knob (glowing dual ring)
        knob_cy = track_y + track_h // 2
        cv2.circle(panel, (fill_x, knob_cy), 8, _SLIDER_KNOB, -1, cv2.LINE_AA)
        cv2.circle(panel, (fill_x, knob_cy), 9, (80, 60, 40), 1, cv2.LINE_AA)

        # Speed Labels
        cv2.putText(panel, "1x (Fine)", (track_x1, track_y + track_h + 14),
                    _FONT, 0.28, _TEXT_DIM, 1, cv2.LINE_AA)
        fast_tw = cv2.getTextSize("25x (Fast)", _FONT, 0.28, 1)[0][0]
        cv2.putText(panel, "25x (Fast)", (track_x2 - fast_tw, track_y + track_h + 14),
                    _FONT, 0.28, _TEXT_DIM, 1, cv2.LINE_AA)

        y += 32 if is_compact else 38

        # ══════════════════════════════════════════════════════════
        #  5. VOICE ASSISTANT CARD
        # ══════════════════════════════════════════════════════════
        voice_card_h = 36 if is_compact else 42
        draw_rounded_rect(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + voice_card_h), _CARD_BG, radius=8, thickness=-1)
        draw_rounded_rect(panel, (_PAD, y), (PANEL_WIDTH - _PAD, y + voice_card_h), _CARD_BORDER, radius=8, thickness=1)

        # Microphone indicator
        cv2.circle(panel, (_PAD + 14, y + voice_card_h // 2), 5, (230, 140, 100), -1, cv2.LINE_AA)
        cv2.putText(panel, "WAKE WORD: \"Jarvis\"", (_PAD + 26, y + 16 if is_compact else y + 18),
                    _FONT, 0.36, _TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(panel, "Agent: Groq Cloud (Llama 3.3)", (_PAD + 26, y + 29 if is_compact else y + 33),
                    _FONT, 0.30, _TEXT_MUTED, 1, cv2.LINE_AA)

        y += voice_card_h + (8 if is_compact else 12)

        # ══════════════════════════════════════════════════════════
        #  6. PAUSE BUTTON & SHORTCUTS FOOTER
        # ══════════════════════════════════════════════════════════
        btn_h = 32 if is_compact else 36
        btn_y1 = y
        btn_y2 = y + btn_h
        self._toggle_rect = (_PAD, btn_y1, PANEL_WIDTH - _PAD, btn_y2)

        is_enabled = mouse.enabled if mouse else True
        btn_bg = _CARD_HOVER if self._hover_btn == "toggle" else _BTN_BG
        btn_border = _BTN_ACTIVE_BORDER if is_enabled else _BTN_PAUSED_BORDER

        draw_rounded_rect(panel, (_PAD, btn_y1), (PANEL_WIDTH - _PAD, btn_y2), btn_bg, radius=8, thickness=-1)
        draw_rounded_rect(panel, (_PAD, btn_y1), (PANEL_WIDTH - _PAD, btn_y2), btn_border, radius=8, thickness=1)

        btn_status = "[P] Tracking Active" if is_enabled else "[P] Tracking PAUSED"
        btn_color = _TEXT_EMERALD if is_enabled else (100, 100, 255)
        btn_tw = cv2.getTextSize(btn_status, _FONT, 0.38, 1)[0][0]
        cv2.putText(panel, btn_status,
                    ((PANEL_WIDTH - btn_tw) // 2, btn_y2 - 11 if is_compact else btn_y2 - 13),
                    _FONT, 0.38, btn_color, 1, cv2.LINE_AA)

        y = btn_y2 + (8 if is_compact else 12)

        # Hotkeys hint
        hotkeys_text = "[P] Pause   [Q] Quit   [+/-] Speed"
        hkw = cv2.getTextSize(hotkeys_text, _FONT, 0.30, 1)[0][0]
        cv2.putText(panel, hotkeys_text, ((PANEL_WIDTH - hkw) // 2, y + 10),
                    _FONT, 0.30, _TEXT_DIM, 1, cv2.LINE_AA)

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
