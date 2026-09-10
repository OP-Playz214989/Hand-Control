"""
gesture_listener.py — Hand-tracking virtual mouse via webcam.

Uses MediaPipe HandLandmarker for raw landmark detection and the
HandMouse class to translate finger positions into OS mouse actions:
    Index finger pointing   → move cursor
    Thumb + index pinch     → left click
    Thumb + middle pinch    → right click
    Index + middle up/down  → scroll
    Pinch hold + move       → drag & drop

The window displays: webcam feed (left) + status panel (right).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from download_model import ensure_model
from hand_mouse import HandMouse, HandState, _is_finger_extended
from side_panel import SidePanel, PANEL_WIDTH, draw_rounded_rect


# Hand landmark connections for drawing skeleton
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# State → display info (Theme matching side_panel)
_STATE_DISPLAY: dict[HandState, tuple[str, tuple[int, int, int]]] = {
    HandState.IDLE:      ("STANDBY",     (130, 125, 120)),
    HandState.MOVING:    ("TRACKING",    (100, 235, 0)),
    HandState.L_CLICK:   ("LEFT CLICK",  (255, 210, 0)),
    HandState.R_CLICK:   ("RIGHT CLICK", (100, 100, 255)),
    HandState.SCROLLING: ("4-WAY SCROLL",(0, 195, 255)),
    HandState.DRAGGING:  ("DRAGGING",    (235, 95, 195)),
}


_WINDOW_NAME = "Mini Jarvis"


# ── Drawing helpers ──────────────────────────────────────────────────

def _draw_active_zone(frame: np.ndarray, w: int, h: int) -> None:
    """Draw high-tech HUD corner brackets marking the active screen-mapping zone."""
    margin = 0.125  # Matches _ACTIVE_ZONE = 0.75 in hand_mouse
    x1 = int(margin * w)
    y1 = int(margin * h)
    x2 = int((1.0 - margin) * w)
    y2 = int((1.0 - margin) * h)

    bracket_len = 20
    col = (70, 60, 50)  # Subtle dark slate

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), col, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), col, 2, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), col, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), col, 2, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), col, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), col, 2, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), col, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), col, 2, cv2.LINE_AA)


def _draw_landmarks(frame: np.ndarray, landmarks, w: int, h: int) -> None:
    """Draw sleek anti-aliased hand skeleton and glowing joint nodes."""
    points = []
    for lm in landmarks:
        px, py = int((1.0 - lm.x) * w), int(lm.y * h)
        points.append((px, py))

    # Bone connections (smooth cyan-slate)
    for s, e in _HAND_CONNECTIONS:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], (180, 150, 60), 2, cv2.LINE_AA)

    # Joint nodes
    for idx, (px, py) in enumerate(points):
        if idx == 8:
            continue  # Index tip has dedicated reticle
        cv2.circle(frame, (px, py), 4, (30, 24, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, (230, 215, 120), -1, cv2.LINE_AA)


def _draw_fingertip_highlight(frame: np.ndarray, landmarks, w: int, h: int) -> None:
    """Highlight the index fingertip with a precision HUD reticle."""
    tip = landmarks[8]
    px, py = int((1.0 - tip.x) * w), int(tip.y * h)

    # Outer precision ring
    reticle_color = (255, 215, 0)  # Bright cyan
    cv2.circle(frame, (px, py), 13, reticle_color, 2, cv2.LINE_AA)

    # Crosshair ticks (4px)
    cv2.line(frame, (px, py - 13), (px, py - 18), reticle_color, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py + 13), (px, py + 18), reticle_color, 1, cv2.LINE_AA)
    cv2.line(frame, (px - 13, py), (px - 18, py), reticle_color, 1, cv2.LINE_AA)
    cv2.line(frame, (px + 13, py), (px + 18, py), reticle_color, 1, cv2.LINE_AA)

    # Center dot
    cv2.circle(frame, (px, py), 3, reticle_color, -1, cv2.LINE_AA)


def _draw_pinch_indicator(frame: np.ndarray, landmarks, w: int, h: int, state: HandState) -> None:
    """Draw a dynamic pinch connection between fingers when clicking/dragging."""
    if state in (HandState.L_CLICK, HandState.DRAGGING):
        t = landmarks[4]   # thumb tip
        i = landmarks[8]   # index tip
        pt1 = (int((1.0 - t.x) * w), int(t.y * h))
        pt2 = (int((1.0 - i.x) * w), int(i.y * h))
        col = (255, 210, 0) if state == HandState.L_CLICK else (235, 95, 195)
        cv2.line(frame, pt1, pt2, col, 3, cv2.LINE_AA)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.circle(frame, mid, 6, col, -1, cv2.LINE_AA)
        cv2.circle(frame, mid, 10, col, 1, cv2.LINE_AA)
    elif state == HandState.R_CLICK:
        t = landmarks[4]   # thumb tip
        m = landmarks[12]  # middle tip
        pt1 = (int((1.0 - t.x) * w), int(t.y * h))
        pt2 = (int((1.0 - m.x) * w), int(m.y * h))
        col = (100, 100, 255)
        cv2.line(frame, pt1, pt2, col, 3, cv2.LINE_AA)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.circle(frame, mid, 6, col, -1, cv2.LINE_AA)
        cv2.circle(frame, mid, 10, col, 1, cv2.LINE_AA)


def _draw_scroll_indicator(frame: np.ndarray, state: HandState, landmarks, w: int, h: int) -> None:
    """Draw a frosted glass 4-way compass HUD in scroll mode."""
    if state != HandState.SCROLLING:
        return
    cx = int((1.0 - landmarks[9].x) * w)
    cy = int(landmarks[9].y * h)

    # Frosted circular background overlay
    radius = 36
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (24, 18, 15), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Outer accent ring
    col = (0, 195, 255)  # Electric amber
    cv2.circle(frame, (cx, cy), radius, col, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, col, -1, cv2.LINE_AA)

    # 4 directional chevrons
    cv2.arrowedLine(frame, (cx, cy - 8), (cx, cy - 28), col, 2, tipLength=0.45)
    cv2.arrowedLine(frame, (cx, cy + 8), (cx, cy + 28), col, 2, tipLength=0.45)
    cv2.arrowedLine(frame, (cx - 8, cy), (cx - 28, cy), col, 2, tipLength=0.45)
    cv2.arrowedLine(frame, (cx + 8, cy), (cx + 28, cy), col, 2, tipLength=0.45)


def _draw_state_label(frame: np.ndarray, state: HandState) -> None:
    """Draw a glassmorphic pill badge in the top-left showing current mode."""
    label, color = _STATE_DISPLAY.get(state, ("STANDBY", (130, 125, 120)))

    # Pill dimensions
    pill_w = 145
    pill_h = 32
    x1, y1 = 14, 14
    x2, y2 = x1 + pill_w, y1 + pill_h

    draw_rounded_rect(frame, (x1, y1), (x2, y2), (28, 22, 19), radius=8, thickness=-1)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), color, radius=8, thickness=1)

    # Glowing status dot
    cv2.circle(frame, (x1 + 14, y1 + pill_h // 2), 5, color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 26, y1 + pill_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (250, 250, 250), 2, cv2.LINE_AA)


def _draw_waiting_screen(frame: np.ndarray, w: int, h: int) -> None:
    """Draw an elegant standby guide card when no hand is in view."""
    card_w = 270
    card_h = 56
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2 = x1 + card_w
    y2 = y1 + card_h

    draw_rounded_rect(frame, (x1, y1), (x2, y2), (28, 22, 19), radius=10, thickness=-1)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), (55, 45, 40), radius=10, thickness=1)

    cv2.putText(frame, "TRACKING STANDBY", (x1 + 42, y1 + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (250, 250, 250), 2, cv2.LINE_AA)
    cv2.putText(frame, "Place your hand in view to control", (x1 + 26, y1 + 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 155, 150), 1, cv2.LINE_AA)


# ── Main listener class ─────────────────────────────────────────────

class GestureListener:
    """
    Webcam-based hand-tracking virtual mouse.

    Uses MediaPipe HandLandmarker for raw landmark detection and
    HandMouse for OS mouse control.
    """

    def __init__(self, cmd_queue: queue.Queue, stop_event: threading.Event):
        self._queue = cmd_queue
        self._stop = stop_event
        self._panel = SidePanel()
        self._mouse = HandMouse()

        # ── HandLandmarker (detect 2 hands to prevent flipping between hands) ──
        hl_path = ensure_model("hand_landmarker")
        hl_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=hl_path,
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(hl_options)

        # Sticky hand lock-in (locks onto one index finger and prevents switching)
        self._tracked_handedness: Optional[str] = None
        self._tracked_index_tip: Optional[tuple[float, float]] = None
        self._hand_last_seen: float = 0.0

        # Real timestamp tracking
        self._start_time: float = time.monotonic()
        self._cam_width: int = 640

        # Click flash timer (for visual feedback)
        self._click_flash_until: float = 0.0
        self._click_flash_color: tuple[int, int, int] = (0, 0, 0)

    def _select_hand(self, result, now: float):
        """
        Select and lock onto a single hand's index finger.
        Once acquired, refuses to switch to another hand or index finger.
        Never drops frames during fast hand movements.
        """
        if not result.hand_landmarks:
            if self._tracked_index_tip is not None and (now - self._hand_last_seen) > 0.8:
                self._tracked_index_tip = None
                self._tracked_handedness = None
            return None

        # ── Case 1: Only 1 hand detected — always track without dropping
        if len(result.hand_landmarks) == 1:
            chosen_lm = result.hand_landmarks[0]
            self._tracked_index_tip = (chosen_lm[8].x, chosen_lm[8].y)
            self._hand_last_seen = now
            if result.handedness and len(result.handedness) > 0:
                self._tracked_handedness = result.handedness[0][0].category_name
            return chosen_lm

        # ── Case 2: Multiple hands detected — choose the locked hand
        if self._tracked_index_tip is None:
            best_idx = 0
            best_score = -1.0
            for idx, lm in enumerate(result.hand_landmarks):
                conf = 0.5
                if result.handedness and idx < len(result.handedness):
                    conf = result.handedness[idx][0].score
                ext_bonus = 1.0 if _is_finger_extended(lm, 8, 6) else 0.0
                score = conf + ext_bonus
                if score > best_score:
                    best_score = score
                    best_idx = idx

            chosen_lm = result.hand_landmarks[best_idx]
            if result.handedness and best_idx < len(result.handedness):
                self._tracked_handedness = result.handedness[best_idx][0].category_name
            self._tracked_index_tip = (chosen_lm[8].x, chosen_lm[8].y)
            self._hand_last_seen = now
            return chosen_lm

        # Multiple hands and already locked — match by proximity and handedness
        best_idx = 0
        best_dist = float("inf")
        tx, ty = self._tracked_index_tip

        for idx, lm in enumerate(result.hand_landmarks):
            ix, iy = lm[8].x, lm[8].y
            dist = math.hypot(ix - tx, iy - ty)

            # Check handedness consistency
            h_name = None
            if result.handedness and idx < len(result.handedness):
                h_name = result.handedness[idx][0].category_name

            if h_name is not None and self._tracked_handedness is not None:
                if h_name == self._tracked_handedness:
                    dist -= 0.20  # bonus for matching the locked hand
                else:
                    dist += 0.20  # penalty for opposing hand

            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        chosen_lm = result.hand_landmarks[best_idx]
        self._tracked_index_tip = (chosen_lm[8].x, chosen_lm[8].y)
        self._hand_last_seen = now
        if result.handedness and best_idx < len(result.handedness):
            self._tracked_handedness = result.handedness[best_idx][0].category_name
        return chosen_lm

    def run(self) -> None:
        """Entry point — must run on the main thread (macOS)."""
        cap = cv2.VideoCapture(0)
        # Force 640×480 — default is often 1080p which makes
        # MediaPipe hand detection too slow and laggy.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            print("[gesture] ⚠  Could not open webcam. Gesture listener disabled.")
            try:
                while not self._stop.is_set():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            return

        print("[gesture] Webcam opened. Hand-tracking mouse active.")
        print("[gesture] Index finger → cursor | Pinch → click | Two fingers → scroll")
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(_WINDOW_NAME, self._mouse_callback)

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]

            # ── Process the ORIGINAL (unflipped) frame with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            # ── Flip the frame AFTER detection for mirror display
            frame = cv2.flip(frame, 1)
            self._cam_width = w

            now = time.monotonic()

            hand_lm = self._select_hand(result, now)

            # Draw active control zone corner brackets
            _draw_active_zone(frame, w, h)

            if hand_lm is not None:
                # Draw sleek hand skeleton
                _draw_landmarks(frame, hand_lm, w, h)
                _draw_fingertip_highlight(frame, hand_lm, w, h)

                # Update virtual mouse
                state = self._mouse.update(hand_lm, w, h)

                # Update panel
                self._panel.update_state(state, self._mouse.status_msg)

                # Draw overlays
                _draw_pinch_indicator(frame, hand_lm, w, h, state)
                _draw_scroll_indicator(frame, state, hand_lm, w, h)
                _draw_state_label(frame, state)

                # Click flash
                if state == HandState.L_CLICK:
                    self._click_flash_until = now + 0.15
                    self._click_flash_color = (255, 210, 0)
                elif state == HandState.R_CLICK:
                    self._click_flash_until = now + 0.15
                    self._click_flash_color = (100, 100, 255)

            else:
                self._mouse.reset()
                self._panel.update_state(HandState.IDLE, "NO HAND")
                _draw_waiting_screen(frame, w, h)
                _draw_state_label(frame, HandState.IDLE)

            # ── Click flash border ────────────────────────────────
            if now < self._click_flash_until:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                              self._click_flash_color, 3)

            # ── Combine + display ─────────────────────────────────
            panel_img = self._panel.render(h, now, self._mouse)
            combined = np.hstack([frame, panel_img])
            cv2.imshow(_WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._stop.set()
                break
            elif key == ord("p"):
                # Toggle pause
                self._mouse.enabled = not self._mouse.enabled
                print(f"[gesture] Mouse {'enabled' if self._mouse.enabled else 'paused'}")
            elif key in (ord("+"), ord("=")):  # = is unshifted +
                self._mouse.scroll_sensitivity = max(0.01, self._mouse.scroll_sensitivity / 1.25)
                print(f"[gesture] Scroll speed: {1.0 / self._mouse.scroll_sensitivity:.0f}x")
            elif key == ord("-"):
                self._mouse.scroll_sensitivity = min(1.0, self._mouse.scroll_sensitivity * 1.25)
                print(f"[gesture] Scroll speed: {1.0 / self._mouse.scroll_sensitivity:.0f}x")

        self._mouse.reset()
        cap.release()
        cv2.destroyAllWindows()
        print("[gesture] Listener stopped.")

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle real mouse clicks on the panel area."""
        self._panel.on_mouse(event, x, y, self._cam_width, time.monotonic(),
                             self._mouse)
