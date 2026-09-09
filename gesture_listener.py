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
from side_panel import SidePanel, PANEL_WIDTH


# Hand landmark connections for drawing skeleton
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# State → display info
_STATE_DISPLAY: dict[HandState, tuple[str, tuple[int, int, int]]] = {
    HandState.IDLE:      ("IDLE",        (120, 120, 120)),
    HandState.MOVING:    ("MOVING",  (0, 255, 100)),
    HandState.L_CLICK:   ("LEFT CLICK",  (0, 200, 255)),
    HandState.R_CLICK:   ("RIGHT CLICK", (255, 100, 100)),
    HandState.SCROLLING: ("SCROLL",  (255, 200, 0)),
    HandState.DRAGGING:  ("DRAGGING",    (200, 100, 255)),
}


_WINDOW_NAME = "Mini Jarvis"


# ── Drawing helpers ──────────────────────────────────────────────────

def _draw_landmarks(frame, landmarks, w: int, h: int) -> None:
    """Draw hand landmarks and bone connections on the display frame."""
    points = []
    for lm in landmarks:
        # Mirror the x-coordinate since we display flipped but detect unflipped
        px, py = int((1.0 - lm.x) * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
    for s, e in _HAND_CONNECTIONS:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], (255, 255, 255), 2)


def _draw_fingertip_highlight(frame, landmarks, w: int, h: int) -> None:
    """Highlight the index fingertip (cursor source) with a larger ring."""
    tip = landmarks[8]
    px, py = int((1.0 - tip.x) * w), int(tip.y * h)
    cv2.circle(frame, (px, py), 14, (0, 255, 255), 2)
    cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)


def _draw_pinch_indicator(frame, landmarks, w: int, h: int, state: HandState) -> None:
    """Draw a line between pinching fingers when clicking/dragging."""
    if state in (HandState.L_CLICK, HandState.DRAGGING):
        t = landmarks[4]   # thumb tip
        i = landmarks[8]   # index tip
        pt1 = (int((1.0 - t.x) * w), int(t.y * h))
        pt2 = (int((1.0 - i.x) * w), int(i.y * h))
        cv2.line(frame, pt1, pt2, (0, 200, 255), 3)
    elif state == HandState.R_CLICK:
        t = landmarks[4]   # thumb tip
        m = landmarks[12]  # middle tip
        pt1 = (int((1.0 - t.x) * w), int(t.y * h))
        pt2 = (int((1.0 - m.x) * w), int(m.y * h))
        cv2.line(frame, pt1, pt2, (255, 100, 100), 3)


def _draw_state_label(frame, state: HandState) -> None:
    """Draw the current hand state as a large label on the feed."""
    label, color = _STATE_DISPLAY.get(state, ("?", (200, 200, 200)))
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def _draw_scroll_indicator(frame, state: HandState, landmarks, w: int, h: int) -> None:
    """Draw 4-way arrows when in scroll mode."""
    if state != HandState.SCROLLING:
        return
    # Draw arrows near the middle of the hand
    cx = int((1.0 - landmarks[9].x) * w)
    cy = int(landmarks[9].y * h)
    # Up / Down arrows
    cv2.arrowedLine(frame, (cx, cy - 10), (cx, cy - 50), (255, 200, 0), 3, tipLength=0.4)
    cv2.arrowedLine(frame, (cx, cy + 10), (cx, cy + 50), (255, 200, 0), 3, tipLength=0.4)
    # Left / Right arrows
    cv2.arrowedLine(frame, (cx - 10, cy), (cx - 50, cy), (255, 200, 0), 3, tipLength=0.4)
    cv2.arrowedLine(frame, (cx + 10, cy), (cx + 50, cy), (255, 200, 0), 3, tipLength=0.4)


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

            if hand_lm is not None:
                # Draw hand skeleton
                _draw_landmarks(frame, hand_lm, w, h)
                _draw_fingertip_highlight(frame, hand_lm, w, h)

                # Update virtual mouse
                state = self._mouse.update(hand_lm, w, h)

                # Update panel
                self._panel.update_state(state, self._mouse.status_msg)

                # Draw overlays
                _draw_pinch_indicator(frame, hand_lm, w, h, state)
                _draw_state_label(frame, state)
                _draw_scroll_indicator(frame, state, hand_lm, w, h)

                # Click flash
                if state == HandState.L_CLICK:
                    self._click_flash_until = now + 0.15
                    self._click_flash_color = (0, 200, 255)
                elif state == HandState.R_CLICK:
                    self._click_flash_until = now + 0.15
                    self._click_flash_color = (255, 100, 100)

            else:
                self._mouse.reset()
                self._panel.update_state(HandState.IDLE, "NO HAND")
                cv2.putText(frame, "Show your hand", (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

            # ── Click flash border ────────────────────────────────
            if now < self._click_flash_until:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                              self._click_flash_color, 4)

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
