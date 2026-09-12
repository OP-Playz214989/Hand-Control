"""
hand_mouse.py — Virtual mouse controlled by hand tracking.

Uses MediaPipe hand landmarks to:
  • Move the OS cursor via index-finger tip position
  • Left-click  via thumb + index pinch
  • Right-click via thumb + middle-finger pinch
  • Scroll      via two-finger vertical movement
  • Drag        via pinch-hold + movement

Uses native macOS Quartz CoreGraphics for zero-overhead mouse control.
Falls back to pyautogui on non-macOS systems.

Cursor tracks the index fingertip ALWAYS when hand is visible,
regardless of gesture state. Actions are layered on top.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import math
import platform
import threading
import time
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
#  Native macOS mouse control via Quartz CoreGraphics (ctypes).
#  ~50x faster than pyautogui — no Python wrapper overhead.
# ══════════════════════════════════════════════════════════════════════

class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _NativeMouse:
    """
    Low-level mouse control using macOS CoreGraphics via ctypes.
    Uses kCGEventSourceStateHIDSystemState so the operating system and all
    applications recognize the cursor as the genuine system trackpad/mouse,
    firing native mouse-moved, hover, click, and scroll events.
    Falls back to pyautogui on non-macOS.
    """

    # CG event type constants
    _kCGEventMouseMoved        = 5
    _kCGEventLeftMouseDown     = 1
    _kCGEventLeftMouseUp       = 2
    _kCGEventLeftMouseDragged  = 6
    _kCGEventRightMouseDown    = 3
    _kCGEventRightMouseUp      = 4
    _kCGHIDEventTap            = 0
    _kCGMouseButtonLeft        = 0
    _kCGMouseButtonRight       = 1
    _kCGScrollEventUnitPixel   = 0
    _kCGMouseEventClickState   = 1

    def __init__(self):
        self._use_quartz = False
        if platform.system() == "Darwin":
            try:
                self._cg = ctypes.cdll.LoadLibrary(
                    "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
                )
                self._cf = ctypes.cdll.LoadLibrary(
                    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
                )

                # Event source from hardware HID system state (1 = native trackpad/mouse)
                self._cg.CGEventSourceCreate.argtypes = [ctypes.c_int32]
                self._cg.CGEventSourceCreate.restype = ctypes.c_void_p
                self._kCGEventSourceStateHIDSystemState = 1
                self._event_source = self._cg.CGEventSourceCreate(self._kCGEventSourceStateHIDSystemState)

                # Mouse event creation and dispatch
                self._cg.CGEventCreateMouseEvent.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, _CGPoint, ctypes.c_uint32
                ]
                self._cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p

                self._cg.CGEventSetIntegerValueField.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64
                ]
                self._cg.CGEventSetIntegerValueField.restype = None

                self._cg.CGEventCreateScrollWheelEvent.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32
                ]
                self._cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p

                if hasattr(self._cg, "CGEventCreateScrollWheelEvent2"):
                    self._cg.CGEventCreateScrollWheelEvent2.argtypes = [
                        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
                    ]
                    self._cg.CGEventCreateScrollWheelEvent2.restype = ctypes.c_void_p
                    self._has_scroll2 = True
                else:
                    self._has_scroll2 = False

                self._cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
                self._cg.CGEventPost.restype = None

                self._cf.CFRelease.argtypes = [ctypes.c_void_p]
                self._cf.CFRelease.restype = None

                # Query current system mouse location
                self._cg.CGEventCreate.argtypes = [ctypes.c_void_p]
                self._cg.CGEventCreate.restype = ctypes.c_void_p

                self._cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
                self._cg.CGEventGetLocation.restype = _CGPoint

                self._cg.CGWarpMouseCursorPosition.argtypes = [_CGPoint]
                self._cg.CGWarpMouseCursorPosition.restype = ctypes.c_int32

                self._cg.CGAssociateMouseAndMouseCursorPosition.argtypes = [ctypes.c_int]
                self._cg.CGAssociateMouseAndMouseCursorPosition.restype = ctypes.c_int32

                self._cg.CGMainDisplayID.argtypes = []
                self._cg.CGMainDisplayID.restype = ctypes.c_uint32
                self._cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
                self._cg.CGDisplayPixelsWide.restype = ctypes.c_uint64
                self._cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
                self._cg.CGDisplayPixelsHigh.restype = ctypes.c_uint64

                self._use_quartz = True
                print("[mouse] Using native Quartz CoreGraphics (system trackpad/mouse mode)")
            except (OSError, AttributeError) as e:
                print(f"[mouse] Quartz unavailable ({e}), falling back to pyautogui")

        if not self._use_quartz:
            import pyautogui
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = True
            self._pyautogui = pyautogui
            print("[mouse] Using pyautogui fallback")

    def screen_size(self) -> tuple[int, int]:
        if self._use_quartz:
            d = self._cg.CGMainDisplayID()
            return (int(self._cg.CGDisplayPixelsWide(d)),
                    int(self._cg.CGDisplayPixelsHigh(d)))
        return self._pyautogui.size()

    def get_position(self) -> tuple[float, float]:
        """Get the current system trackpad/mouse cursor position."""
        if self._use_quartz:
            ev = self._cg.CGEventCreate(None)
            if ev:
                loc = self._cg.CGEventGetLocation(ev)
                self._cf.CFRelease(ev)
                return (loc.x, loc.y)
        if hasattr(self, "_pyautogui"):
            pos = self._pyautogui.position()
            return (float(pos[0]), float(pos[1]))
        return (0.0, 0.0)

    def move_to(self, x: float, y: float) -> None:
        """
        Move the native system cursor and post genuine kCGEventMouseMoved
        events to kCGHIDEventTap so the OS, Dock, menus, and browser apps
        trigger native hover and cursor transformations without breaking
        hardware trackpad/mouse association.
        """
        if self._use_quartz:
            pt = _CGPoint(x, y)
            event = self._cg.CGEventCreateMouseEvent(
                self._event_source, self._kCGEventMouseMoved, pt, 0
            )
            if event:
                self._cg.CGEventPost(self._kCGHIDEventTap, event)
                self._cf.CFRelease(event)
        else:
            self._pyautogui.moveTo(int(x), int(y), _pause=False)

    def click(self, x: float, y: float, button: str = "left") -> None:
        """Post a native click event with clickState=1 recognized by all Cocoa/web apps."""
        if self._use_quartz:
            pt = _CGPoint(x, y)
            btn = self._kCGMouseButtonLeft if button == "left" else self._kCGMouseButtonRight
            down_type = self._kCGEventLeftMouseDown if button == "left" else self._kCGEventRightMouseDown
            up_type = self._kCGEventLeftMouseUp if button == "left" else self._kCGEventRightMouseUp

            ev_down = self._cg.CGEventCreateMouseEvent(self._event_source, down_type, pt, btn)
            if ev_down:
                self._cg.CGEventSetIntegerValueField(ev_down, self._kCGMouseEventClickState, 1)
                self._cg.CGEventPost(self._kCGHIDEventTap, ev_down)
                self._cf.CFRelease(ev_down)

            time.sleep(0.01)  # 10ms click duration matching physical click

            ev_up = self._cg.CGEventCreateMouseEvent(self._event_source, up_type, pt, btn)
            if ev_up:
                self._cg.CGEventSetIntegerValueField(ev_up, self._kCGMouseEventClickState, 1)
                self._cg.CGEventPost(self._kCGHIDEventTap, ev_up)
                self._cf.CFRelease(ev_up)
        else:
            self._pyautogui.click(int(x), int(y), button=button)

    def mouse_down(self, x: float, y: float) -> None:
        if self._use_quartz:
            pt = _CGPoint(x, y)
            ev = self._cg.CGEventCreateMouseEvent(self._event_source, self._kCGEventLeftMouseDown, pt, self._kCGMouseButtonLeft)
            if ev:
                self._cg.CGEventSetIntegerValueField(ev, self._kCGMouseEventClickState, 1)
                self._cg.CGEventPost(self._kCGHIDEventTap, ev)
                self._cf.CFRelease(ev)
        else:
            self._pyautogui.mouseDown(int(x), int(y), button='left')

    def mouse_up(self, x: float, y: float) -> None:
        if self._use_quartz:
            pt = _CGPoint(x, y)
            ev = self._cg.CGEventCreateMouseEvent(self._event_source, self._kCGEventLeftMouseUp, pt, self._kCGMouseButtonLeft)
            if ev:
                self._cg.CGEventSetIntegerValueField(ev, self._kCGMouseEventClickState, 1)
                self._cg.CGEventPost(self._kCGHIDEventTap, ev)
                self._cf.CFRelease(ev)
        else:
            self._pyautogui.mouseUp(int(x), int(y), button='left')

    def drag_move(self, x: float, y: float) -> None:
        """Move cursor while left button is held (drag)."""
        if self._use_quartz:
            pt = _CGPoint(x, y)
            ev = self._cg.CGEventCreateMouseEvent(self._event_source, self._kCGEventLeftMouseDragged, pt, self._kCGMouseButtonLeft)
            if ev:
                self._cg.CGEventSetIntegerValueField(ev, self._kCGMouseEventClickState, 1)
                self._cg.CGEventPost(self._kCGHIDEventTap, ev)
                self._cf.CFRelease(ev)
        else:
            self._pyautogui.moveTo(int(x), int(y), _pause=False)

    def scroll(self, pixels: int) -> None:
        """Scroll vertically using native trackpad continuous scroll event."""
        if self._use_quartz:
            if getattr(self, "_has_scroll2", False):
                event = self._cg.CGEventCreateScrollWheelEvent2(
                    self._event_source, self._kCGScrollEventUnitPixel, 1, pixels, 0, 0
                )
            else:
                event = self._cg.CGEventCreateScrollWheelEvent(
                    self._event_source, self._kCGScrollEventUnitPixel, 1, pixels
                )
            if event:
                self._cg.CGEventPost(self._kCGHIDEventTap, event)
                self._cf.CFRelease(event)
        else:
            self._pyautogui.scroll(pixels)

    def scroll_h(self, pixels: int) -> None:
        """Scroll horizontally using native trackpad continuous scroll event."""
        if self._use_quartz:
            if getattr(self, "_has_scroll2", False):
                event = self._cg.CGEventCreateScrollWheelEvent2(
                    self._event_source, self._kCGScrollEventUnitPixel, 2, 0, pixels, 0
                )
            else:
                event = self._cg.CGEventCreateScrollWheelEvent(
                    self._event_source, self._kCGScrollEventUnitPixel, 2, 0, pixels
                )
            if event:
                self._cg.CGEventPost(self._kCGHIDEventTap, event)
                self._cf.CFRelease(event)
        else:
            self._pyautogui.hscroll(pixels)


# ── Singleton native mouse instance ─────────────────────────────────
_mouse = _NativeMouse()


# ── Hand state enum ──────────────────────────────────────────────────

class HandState(enum.Enum):
    IDLE      = "idle"
    MOVING    = "moving"
    L_CLICK   = "left_click"
    R_CLICK   = "right_click"
    SCROLLING = "scrolling"
    DRAGGING  = "dragging"


# ── Finger detection helpers ─────────────────────────────────────────

_WRIST      = 0
_THUMB_TIP  = 4
_INDEX_PIP  = 6
_INDEX_TIP  = 8
_MIDDLE_PIP = 10
_MIDDLE_TIP = 12
_RING_PIP   = 14
_RING_TIP   = 16


def _dist_2d(lm, i: int, j: int) -> float:
    """2D Euclidean distance (ignoring depth)."""
    a, b = lm[i], lm[j]
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _is_finger_extended(lm, tip: int, pip: int) -> bool:
    """
    Orientation-agnostic finger extension check.

    A finger is extended when its tip is further from the wrist
    than its PIP joint is. Works with both palm and back-of-hand
    facing the camera.
    """
    tip_dist = _dist_2d(lm, tip, _WRIST)
    pip_dist = _dist_2d(lm, pip, _WRIST)
    return tip_dist > pip_dist * 1.05  # 5% margin


# ── Configuration ────────────────────────────────────────────────────

_PINCH_THRESHOLD    = 0.035   # normalised 2D distance — thumb+index pinch
_PINCH_RELEASE      = 0.039   # hysteresis — must exceed to release pinch
_MID_PINCH_THRESHOLD = 0.045  # middle finger is longer — wider threshold
_MID_PINCH_RELEASE   = 0.049  # matching hysteresis for middle pinch

# Adaptive smoothing:
# When hand is stationary / fixed, use low alpha (0.18) to eliminate jitter.
# When hand moves fast, scale up to 0.85 for responsive, lag-free tracking.
_MIN_ALPHA          = 0.18    # low-speed smoothing factor (anti-jitter)
_MAX_ALPHA          = 0.85    # high-speed smoothing factor (responsiveness)

# Deadband for stationary / fixed finger:
# Any movement under this radius (screen pixels) is treated as tremor / noise
# and will NOT move the cursor at all, keeping it rock-solid.
_FIXED_DEADBAND_PX  = 4.5     # screen pixels threshold to lock fixed finger
_DRAG_START_DIST_PX = 22.0    # pixel movement to break out of pinch-freeze into drag

_ACTIVE_ZONE        = 0.55    # central fraction of webcam mapped to screen

_DEAD_ZONE_PX       = 1       # OS cursor update threshold (sub-pixel suppression)

_SCROLL_SENSITIVITY = 0.10    # default scroll sensitivity (10x default)
_SCROLL_MAX_PX      = 350     # maximum scroll pixels per frame to allow fast page sweeps without flooding WindowServer

# ── Acceleration (macOS-style) ───────────────────────────────────────
# Power curve: slow hand movements → precise, fast → amplified.
_ACCEL_POWER        = 1.5     # exponent (1.0 = linear / no accel)
_ACCEL_BASE         = 100.0   # pixel distance for 1× gain (crossover point)
_ACCEL_MIN_GAIN     = 0.20    # minimum gain (precision at slow speeds)
_ACCEL_MAX_GAIN     = 3.2     # maximum gain (speed at fast movements)
_SCROLL_ACCEL       = 0.12    # scroll acceleration factor (0 = none)

_DRAG_HOLD_TIME     = 0.28    # seconds pinch must hold before drag starts

_CLICK_COOLDOWN     = 0.22    # seconds between consecutive clicks

_ATTRACT_RADIUS     = 35      # pixels — search radius for clickable elements
_ATTRACT_STRENGTH   = 0.28    # 0–1, base pull strength toward element center
_ATTRACT_SNAP_PX    = 3.0     # snap-lock distance — stop jitter when nearly centered
_ATTRACT_INTERVAL   = 0.08    # seconds between accessibility API queries
_ATTRACT_SETTLE_VEL = 12.0    # px — cursor must be slower than this for attraction to engage
_ATTRACT_HYST_IN    = 0.85    # radius multiplier to enter attraction (inner ring)
_ATTRACT_HYST_OUT   = 1.4     # radius multiplier to leave attraction (outer ring)


# ── Cursor attraction via macOS Accessibility API ────────────────────

class _CursorAttractor:
    """
    Uses macOS Accessibility API (HIServices via ctypes) to detect nearby
    clickable UI elements and provide attraction targets for the cursor.

    All AX queries run on a **dedicated background thread** so they never
    block the cursor-tracking loop.

    Requires Accessibility permission:
      System Settings → Privacy & Security → Accessibility
    """

    _CLICKABLE_ROLES = frozenset({
        b"AXButton", b"AXLink", b"AXCheckBox", b"AXRadioButton",
        b"AXPopUpButton", b"AXComboBox", b"AXMenuItem", b"AXMenuBarItem",
        b"AXTab", b"AXIncrementor", b"AXDisclosureTriangle",
        b"AXSlider", b"AXCell", b"AXTextField", b"AXTextArea",
    })

    # Probe center + four cardinal offsets (tight spacing avoids overshooting small buttons)
    _PROBE_DIRS = [(0, 0), (-0.6, 0), (0.6, 0), (0, -0.6), (0, 0.6)]

    def __init__(self):
        self._available = False
        # Cursor position written by main thread, read by BG thread
        self._cursor_lock = threading.Lock()
        self._cursor_pos: tuple[float, float] = (0.0, 0.0)
        # Target written by BG thread, read by main thread
        # Stores (center_x, center_y, width, height) so main thread can scale strength
        self._target: Optional[tuple[float, float, float, float]] = None

        if platform.system() != "Darwin":
            print("[mouse] Cursor attraction: macOS only — disabled")
            return

        try:
            self._hi = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/ApplicationServices.framework/"
                "Frameworks/HIServices.framework/HIServices"
            )
            self._cf = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/CoreFoundation.framework/"
                "CoreFoundation"
            )

            # ── HIServices bindings ──
            self._hi.AXIsProcessTrusted.argtypes = []
            self._hi.AXIsProcessTrusted.restype = ctypes.c_bool

            self._hi.AXUIElementCreateSystemWide.argtypes = []
            self._hi.AXUIElementCreateSystemWide.restype = ctypes.c_void_p

            self._hi.AXUIElementCopyElementAtPosition.argtypes = [
                ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._hi.AXUIElementCopyElementAtPosition.restype = ctypes.c_int32

            self._hi.AXUIElementCopyAttributeValue.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._hi.AXUIElementCopyAttributeValue.restype = ctypes.c_int32

            self._hi.AXValueGetValue.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ]
            self._hi.AXValueGetValue.restype = ctypes.c_bool

            # ── CoreFoundation bindings ──
            self._cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
            ]
            self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p

            self._cf.CFStringGetCStringPtr.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32,
            ]
            self._cf.CFStringGetCStringPtr.restype = ctypes.c_char_p

            self._cf.CFStringGetCString.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.c_long, ctypes.c_uint32,
            ]
            self._cf.CFStringGetCString.restype = ctypes.c_bool

            self._cf.CFRelease.argtypes = [ctypes.c_void_p]
            self._cf.CFRelease.restype = None

            self._kUTF8 = 0x08000100          # kCFStringEncodingUTF8
            self._kAXValueCGPoint = 1
            self._kAXValueCGSize = 2

            # ── Permission check ──
            if not self._hi.AXIsProcessTrusted():
                print("[mouse] ⚠️  Cursor attraction needs Accessibility permission.")
                print("[mouse]    System Settings → Privacy & Security → Accessibility")
                print("[mouse]    Add and enable your terminal / IDE.")
                return

            self._system = self._hi.AXUIElementCreateSystemWide()
            if not self._system:
                return

            # Cache CFStrings for attribute names (never released)
            self._cf_role = self._mk(b"AXRole")
            self._cf_pos  = self._mk(b"AXPosition")
            self._cf_size = self._mk(b"AXSize")

            self._available = True
            print("[mouse] ✅ Cursor attraction enabled (background thread)")

            # Start the background query thread
            self._thread = threading.Thread(target=self._bg_loop, daemon=True)
            self._thread.start()

        except (OSError, AttributeError) as exc:
            print(f"[mouse] Cursor attraction unavailable: {exc}")

    # ── helpers ──────────────────────────────────────────────────

    def _mk(self, name: bytes) -> ctypes.c_void_p:
        return ctypes.c_void_p(
            self._cf.CFStringCreateWithCString(None, name, self._kUTF8)
        )

    def _cfstr_bytes(self, ref) -> Optional[bytes]:
        ptr = self._cf.CFStringGetCStringPtr(ref, self._kUTF8)
        if ptr:
            return ptr
        buf = ctypes.create_string_buffer(256)
        if self._cf.CFStringGetCString(ref, buf, 256, self._kUTF8):
            return buf.value
        return None

    # ── public API (called from main tracking thread) ────────────

    def get_target(self, cx: float, cy: float) -> Optional[tuple[float, float, float, float]]:
        """
        Return (center_x, center_y, width, height) of a nearby clickable element.
        Non-blocking — just reads the latest result from the BG thread.
        """
        if not self._available:
            return None

        # Update cursor position for the BG thread
        with self._cursor_lock:
            self._cursor_pos = (cx, cy)

        # Read cached target (atomic on CPython thanks to GIL)
        target = self._target
        if target:
            tx, ty = target[0], target[1]
            if math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2) > _ATTRACT_RADIUS * _ATTRACT_HYST_OUT:
                return None
        return target

    # ── background thread ────────────────────────────────────────

    def _bg_loop(self) -> None:
        """Continuously probe for clickable elements in the background."""
        while True:
            time.sleep(_ATTRACT_INTERVAL)
            try:
                with self._cursor_lock:
                    cx, cy = self._cursor_pos

                best_dist = float("inf")
                best_target = None

                for dx, dy in self._PROBE_DIRS:
                    px = cx + dx * _ATTRACT_RADIUS * 0.45
                    py = cy + dy * _ATTRACT_RADIUS * 0.45
                    rect = self._query(px, py)
                    if rect is None:
                        continue
                    ex, ey, ew, eh = rect
                    ecx, ecy = ex + ew / 2.0, ey + eh / 2.0
                    dist = math.sqrt((cx - ecx) ** 2 + (cy - ecy) ** 2)
                    if dist < _ATTRACT_RADIUS * _ATTRACT_HYST_IN and dist < best_dist:
                        best_dist = dist
                        best_target = (ecx, ecy, ew, eh)

                self._target = best_target  # atomic write (GIL)
            except Exception:
                pass  # never crash the BG thread

    # ── internal AX query ────────────────────────────────────────

    def _query(self, x: float, y: float):
        """Return (x, y, w, h) of the clickable element at (x, y), or None."""
        elem = ctypes.c_void_p()
        err = self._hi.AXUIElementCopyElementAtPosition(
            self._system, ctypes.c_float(x), ctypes.c_float(y),
            ctypes.byref(elem),
        )
        if err != 0 or not elem.value:
            return None
        try:
            # Role
            role_ref = ctypes.c_void_p()
            if self._hi.AXUIElementCopyAttributeValue(
                elem, self._cf_role, ctypes.byref(role_ref)
            ) != 0 or not role_ref.value:
                return None
            role = self._cfstr_bytes(role_ref.value)
            self._cf.CFRelease(role_ref)
            if role not in self._CLICKABLE_ROLES:
                return None

            # Position
            pos_ref = ctypes.c_void_p()
            if self._hi.AXUIElementCopyAttributeValue(
                elem, self._cf_pos, ctypes.byref(pos_ref)
            ) != 0 or not pos_ref.value:
                return None
            pt = _CGPoint()
            ok = self._hi.AXValueGetValue(
                pos_ref, self._kAXValueCGPoint, ctypes.byref(pt)
            )
            self._cf.CFRelease(pos_ref)
            if not ok:
                return None

            # Size
            sz_ref = ctypes.c_void_p()
            if self._hi.AXUIElementCopyAttributeValue(
                elem, self._cf_size, ctypes.byref(sz_ref)
            ) != 0 or not sz_ref.value:
                return None
            sz = _CGSize()
            ok = self._hi.AXValueGetValue(
                sz_ref, self._kAXValueCGSize, ctypes.byref(sz)
            )
            self._cf.CFRelease(sz_ref)
            if not ok:
                return None

            return (pt.x, pt.y, sz.width, sz.height)
        except Exception:
            return None
        finally:
            self._cf.CFRelease(elem)


# ── Main class ───────────────────────────────────────────────────────

class HandMouse:
    """
    Translates MediaPipe hand landmarks into OS mouse actions.

    **Key design**: cursor ALWAYS follows the index fingertip when the
    hand is visible. Pinch/scroll/drag are layered on top — they never
    interrupt cursor tracking (except scroll, which intentionally
    freezes the cursor).
    """

    def __init__(self):
        self._screen_w, self._screen_h = _mouse.screen_size()

        # Seed smoothed cursor position directly from current OS trackpad/mouse
        cur_x, cur_y = _mouse.get_position()
        if cur_x <= 0 and cur_y <= 0:
            cur_x, cur_y = self._screen_w / 2.0, self._screen_h / 2.0

        # Smoothed cursor position (screen coords)
        self._sx: float = cur_x
        self._sy: float = cur_y

        # Last position we actually sent to the OS
        self._last_x: float = self._sx
        self._last_y: float = self._sy

        # Active hand tracking session flag
        self._hand_active: bool = False

        # State
        self.state: HandState = HandState.IDLE

        # Pinch tracking
        self._index_pinched: bool = False
        self._index_pinched_prev: bool = False
        self._middle_pinched: bool = False
        self._middle_pinched_prev: bool = False
        self._pinch_start: float = 0.0
        self._pinch_start_x: float = self._sx
        self._pinch_start_y: float = self._sy
        self._dragging: bool = False

        # Click cooldown
        self._last_click_time: float = 0.0

        # Scroll tracking
        self._scroll_anchor_x: Optional[float] = None
        self._scroll_anchor_y: Optional[float] = None
        self._scroll_accum: float = 0.0
        self._scroll_accum_h: float = 0.0   # horizontal accumulator
        self._scroll_vel: float = 0.0       # EMA-smoothed vertical velocity
        self._scroll_vel_h: float = 0.0     # EMA-smoothed horizontal velocity

        # For overlay drawing
        self.index_tip_norm: tuple[float, float] = (0.5, 0.5)
        self.cursor_screen: tuple[int, int] = (int(self._sx), int(self._sy))

        # Status message for the panel
        self.status_msg: str = ""

        # Enabled flag
        self.enabled: bool = True

        # Adjustable scroll sensitivity (slider-controlled)
        self.scroll_sensitivity: float = _SCROLL_SENSITIVITY

        # Cursor attraction to nearby clickable elements
        self._attractor = _CursorAttractor()
        self._attracted: bool = False
        self._attract_locked: bool = False       # hysteresis latch
        self._smooth_attract_x: float = 0.0      # smoothed attraction target
        self._smooth_attract_y: float = 0.0
        self._attract_blend: float = 0.0          # 0-1 blend factor (fades in/out)

    def update(self, landmarks, frame_w: int, frame_h: int) -> HandState:
        """Process one frame of hand landmarks and perform mouse actions."""
        if not self.enabled:
            self.state = HandState.IDLE
            self.status_msg = "PAUSED"
            return self.state

        lm = landmarks
        now = time.monotonic()

        # ══════════════════════════════════════════════════════════
        #  1. ALWAYS compute smoothed cursor from index fingertip
        # ══════════════════════════════════════════════════════════
        raw_x = lm[_INDEX_TIP].x
        raw_y = lm[_INDEX_TIP].y
        self.index_tip_norm = (raw_x, raw_y)

        margin = (1.0 - _ACTIVE_ZONE) / 2.0
        mapped_x = max(0.0, min(1.0, (raw_x - margin) / _ACTIVE_ZONE))
        mapped_y = max(0.0, min(1.0, (raw_y - margin) / _ACTIVE_ZONE))

        target_x = (1.0 - mapped_x) * self._screen_w  # mirror X
        target_y = mapped_y * self._screen_h

        # On hand first appearance, seamlessly acquire cursor from index fingertip
        # without slowly dragging across the screen from a stale point
        if not self._hand_active:
            self._sx = target_x
            self._sy = target_y
            self._last_x = target_x
            self._last_y = target_y
            self._hand_active = True

        # Save pre-EMA position so we can freeze cursor during scroll
        prev_sx, prev_sy = self._sx, self._sy

        # Distance from current smoothed cursor to raw target
        dx = target_x - self._sx
        dy = target_y - self._sy
        dist = math.hypot(dx, dy)

        # ── Fixed-finger stabilization (zero flickering when stationary) ──
        if dist < _FIXED_DEADBAND_PX:
            # Hand / finger is held fixed: freeze cursor, zero jitter
            effective_dx = 0.0
            effective_dy = 0.0
            effective_dist = 0.0
        else:
            # Intentional movement: subtract deadband so motion starts continuously
            ratio = (dist - _FIXED_DEADBAND_PX) / dist
            effective_dx = dx * ratio
            effective_dy = dy * ratio
            effective_dist = dist - _FIXED_DEADBAND_PX

            # Velocity-adaptive smoothing:
            # Low speed -> low alpha (~0.18) for rock-solid precision and anti-tremor
            # High speed -> high alpha (~0.85) for instant, lag-free responsiveness
            speed_factor = min(1.0, effective_dist / 35.0)
            alpha = _MIN_ALPHA + speed_factor * (_MAX_ALPHA - _MIN_ALPHA)

            self._sx += alpha * effective_dx
            self._sy += alpha * effective_dy

            # macOS-style acceleration: amplifies high-speed hand sweeps
            if effective_dist > 1.5:
                gain = (effective_dist / _ACCEL_BASE) ** (_ACCEL_POWER - 1.0)
                gain = max(_ACCEL_MIN_GAIN, min(gain, _ACCEL_MAX_GAIN))
                self._sx += effective_dx * gain * 0.20
                self._sy += effective_dy * gain * 0.20

        sx = self._sx
        sy = self._sy

        # ── Attract cursor toward nearby clickable UI elements ───
        # Uses: hysteresis, velocity gating, size-adaptive strength,
        #       smoothed target, snap-lock, and EMA writeback to prevent
        #       the oscillation/flickering that plagued small buttons.
        self._attracted = False
        _attract_target = None
        if not self._dragging:
            _attract_target = self._attractor.get_target(sx, sy)

        if _attract_target is not None:
            _tx, _ty, _tw, _th = _attract_target
            cursor_to_target = math.hypot(sx - _tx, sy - _ty)

            # Hysteresis: once locked, stay locked until cursor exits the outer ring
            if self._attract_locked:
                if cursor_to_target > _ATTRACT_RADIUS * _ATTRACT_HYST_OUT:
                    self._attract_locked = False
            else:
                if cursor_to_target < _ATTRACT_RADIUS * _ATTRACT_HYST_IN and effective_dist < _ATTRACT_SETTLE_VEL:
                    self._attract_locked = True

            if self._attract_locked:
                # Smooth the attraction target itself to prevent frame-to-frame jitter
                _at_alpha = 0.35
                if self._attract_blend < 0.01:  # first engagement — snap
                    self._smooth_attract_x = _tx
                    self._smooth_attract_y = _ty
                else:
                    self._smooth_attract_x += _at_alpha * (_tx - self._smooth_attract_x)
                    self._smooth_attract_y += _at_alpha * (_ty - self._smooth_attract_y)

                # Fade attraction in over ~5 frames so it doesn't pop
                self._attract_blend = min(1.0, self._attract_blend + 0.22)

                # Scale strength inversely with element size:
                # Large buttons (>60px) → full strength; tiny (≤16px) → reduced
                elem_size = max(_tw, _th)
                size_scale = min(1.0, max(0.4, elem_size / 50.0))
                strength = _ATTRACT_STRENGTH * size_scale * self._attract_blend

                # Snap-lock: if we're already nearly on center, lock perfectly
                dist_to_smooth = math.hypot(sx - self._smooth_attract_x, sy - self._smooth_attract_y)
                if dist_to_smooth < _ATTRACT_SNAP_PX:
                    sx = self._smooth_attract_x
                    sy = self._smooth_attract_y
                else:
                    sx += strength * (self._smooth_attract_x - sx)
                    sy += strength * (self._smooth_attract_y - sy)

                # CRITICAL: Write attracted position back to the internal EMA state
                # so next frame's filter starts from the attracted position rather
                # than the un-attracted raw position. This prevents the oscillation
                # that caused flickering.
                self._sx = sx
                self._sy = sy

                self._attracted = True
            else:
                # Not locked — decay blend smoothly
                self._attract_blend *= 0.7
        else:
            # No target found — release lock and decay smoothly
            self._attract_locked = False
            self._attract_blend *= 0.7

        self.cursor_screen = (int(sx), int(sy))

        cursor_moved = (abs(sx - self._last_x) >= _DEAD_ZONE_PX
                        or abs(sy - self._last_y) >= _DEAD_ZONE_PX)

        # ══════════════════════════════════════════════════════════
        #  2. Detect finger & pinch states
        # ══════════════════════════════════════════════════════════
        index_up  = _is_finger_extended(lm, _INDEX_TIP, _INDEX_PIP)
        middle_up = _is_finger_extended(lm, _MIDDLE_TIP, _MIDDLE_PIP)
        ring_up   = _is_finger_extended(lm, _RING_TIP, _RING_PIP)

        index_dist  = _dist_2d(lm, _THUMB_TIP, _INDEX_TIP)
        middle_dist = _dist_2d(lm, _THUMB_TIP, _MIDDLE_TIP)

        # Save previous frame's state
        self._index_pinched_prev = self._index_pinched
        self._middle_pinched_prev = self._middle_pinched

        # Index pinch with hysteresis
        if not self._index_pinched and index_dist < _PINCH_THRESHOLD:
            self._index_pinched = True
            self._pinch_start = now
        elif self._index_pinched and index_dist > _PINCH_RELEASE:
            self._index_pinched = False

        # Middle pinch with hysteresis (wider threshold — longer finger)
        if not self._middle_pinched and middle_dist < _MID_PINCH_THRESHOLD:
            self._middle_pinched = True
        elif self._middle_pinched and middle_dist > _MID_PINCH_RELEASE:
            self._middle_pinched = False

        # Edge events
        idx_pinch_started  = self._index_pinched and not self._index_pinched_prev
        idx_pinch_released = not self._index_pinched and self._index_pinched_prev
        mid_pinch_started  = self._middle_pinched and not self._middle_pinched_prev
        hold_time = (now - self._pinch_start) if self._index_pinched else 0.0

        # Lock pinch position at initiation
        if idx_pinch_started:
            self._pinch_start_x = sx
            self._pinch_start_y = sy

        pinch_travel = math.hypot(sx - self._pinch_start_x, sy - self._pinch_start_y) if self._index_pinched else 0.0

        # ══════════════════════════════════════════════════════════
        #  3. State machine — actions layered on top of cursor
        # ══════════════════════════════════════════════════════════

        # ── SCROLL: index + middle up, no pinch, not dragging ────
        # Latch scroll mode so rapid movement doesn't drop out of scroll if ring finger flexes
        is_scrolling = False
        if not self._index_pinched and not self._dragging:
            if self.state == HandState.SCROLLING:
                is_scrolling = (index_up and middle_up)
            else:
                is_scrolling = (index_up and middle_up and not ring_up)

        if is_scrolling:
            # Freeze cursor position — undo the EMA update so the
            # cursor stays exactly where it was when scroll started.
            self._sx, self._sy = prev_sx, prev_sy
            self.cursor_screen = (int(prev_sx), int(prev_sy))
            self.state = HandState.SCROLLING
            self._handle_scroll(raw_x, raw_y)
            self.status_msg = "SCROLL"
            return self.state

        # ── DRAG RELEASE: was dragging, pinch released ───────────
        if self._dragging and not self._index_pinched:
            _mouse.mouse_up(sx, sy)
            self._dragging = False
            self.state = HandState.MOVING
            self.status_msg = "DROP"
            self._move_cursor(sx, sy)
            return self.state

        # ── DRAGGING: pinch held past threshold OR moved intentionally ──
        if self._index_pinched and (hold_time > _DRAG_HOLD_TIME or pinch_travel > _DRAG_START_DIST_PX):
            if not self._dragging:
                # Start drag at current position
                _mouse.move_to(sx, sy)
                _mouse.mouse_down(sx, sy)
                self._dragging = True
                self._last_x = sx
                self._last_y = sy
            self.state = HandState.DRAGGING
            if cursor_moved:
                _mouse.drag_move(sx, sy)
                self._last_x = sx
                self._last_y = sy
            self.status_msg = "DRAGGING"
            return self.state

        # ── PINCH HELD (deciding click vs drag) ──────────────────
        if self._index_pinched and hold_time <= _DRAG_HOLD_TIME:
            # Hold cursor fixed at pinch start position so click doesn't slip or flicker
            sx = self._pinch_start_x
            sy = self._pinch_start_y
            self.cursor_screen = (int(sx), int(sy))
            self.state = HandState.MOVING
            self.status_msg = "..."
            return self.state

        # ── LEFT CLICK: short pinch just released ────────────────
        if idx_pinch_released:
            elapsed = now - self._pinch_start
            if elapsed < _DRAG_HOLD_TIME and (now - self._last_click_time) > _CLICK_COOLDOWN:
                # Click at the fixed pinch position
                click_x = self._pinch_start_x
                click_y = self._pinch_start_y
                _mouse.move_to(click_x, click_y)
                _mouse.click(click_x, click_y, "left")
                self._last_click_time = now
                self._last_x = click_x
                self._last_y = click_y
                self.state = HandState.L_CLICK
                self.status_msg = "LEFT CLICK"
            else:
                self.state = HandState.MOVING
                self.status_msg = "MOVING"
            return self.state

        # ── RIGHT CLICK: middle pinch just started ───────────────
        if mid_pinch_started and not self._dragging:
            if (now - self._last_click_time) > _CLICK_COOLDOWN:
                _mouse.move_to(sx, sy)
                _mouse.click(sx, sy, "right")
                self._last_click_time = now
                self._last_x = sx
                self._last_y = sy
                self.state = HandState.R_CLICK
                self.status_msg = "RIGHT CLICK"
            else:
                self.state = HandState.MOVING
            return self.state

        # ── NORMAL MOVEMENT ──────────────────────────────────────
        # Always track cursor when hand is visible — works with
        # both front and back of hand facing the camera.
        self.state = HandState.MOVING
        self._move_cursor(sx, sy)
        self.status_msg = "ATTRACT" if self._attracted else "MOVING"
        self._scroll_anchor_y = None
        return self.state

    def reset(self) -> None:
        """Reset state when hand disappears."""
        if self._dragging:
            _mouse.mouse_up(self._last_x, self._last_y)
            self._dragging = False
        self.state = HandState.IDLE
        self._index_pinched = False
        self._index_pinched_prev = False
        self._middle_pinched = False
        self._middle_pinched_prev = False
        self._scroll_anchor_x = None
        self._scroll_anchor_y = None
        self._scroll_accum = 0.0
        self._scroll_accum_h = 0.0
        self._scroll_vel = 0.0
        self._scroll_vel_h = 0.0
        self.status_msg = "NO HAND"

        # Reseed position from current OS cursor so physical trackpad/mouse usage
        # while hand is idle is respected
        cur_x, cur_y = _mouse.get_position()
        if cur_x > 0 or cur_y > 0:
            self._sx = cur_x
            self._sy = cur_y
            self._last_x = cur_x
            self._last_y = cur_y
        self._hand_active = False

    # ── Internal helpers ─────────────────────────────────────────

    def _move_cursor(self, x: float, y: float) -> None:
        """Move the OS cursor if position has meaningfully changed."""
        if (abs(x - self._last_x) > _DEAD_ZONE_PX
                or abs(y - self._last_y) > _DEAD_ZONE_PX):
            _mouse.move_to(x, y)
            self._last_x = x
            self._last_y = y

    def _handle_scroll(self, raw_x: float, raw_y: float) -> None:
        """Track finger movement and emit smooth vertical + horizontal scroll."""
        if self._scroll_anchor_y is None:
            self._scroll_anchor_x = raw_x
            self._scroll_anchor_y = raw_y
            self._scroll_accum = 0.0
            self._scroll_accum_h = 0.0
            self._scroll_vel = 0.0
            self._scroll_vel_h = 0.0
            return

        mult = 1.0 / max(0.005, self.scroll_sensitivity)

        # ── Vertical scroll ──────────────────────────────────────
        delta_y = (raw_y - self._scroll_anchor_y) * self._screen_h
        # Instant responsiveness so fast swipes register immediately
        self._scroll_vel += 0.65 * (delta_y - self._scroll_vel)

        speed_v = abs(self._scroll_vel)
        if speed_v < 1.0:
            # Stop promptly when hand halts
            self._scroll_vel *= 0.3
            self._scroll_accum *= 0.3
        else:
            # MacBook-style non-linear velocity curve:
            # Slow movement (<= 5.0): dampened gain (0.35 - 1.0) for gradual line-by-line scrolling
            # Fast movement (> 5.0): power curve acceleration (20x scrolls one full page per swipe)
            if speed_v <= 5.0:
                gain_v = 0.35 + 0.13 * speed_v
            else:
                gain_v = 1.0 + 0.085 * ((speed_v - 5.0) ** 1.35)
            gain_v = min(12.0, gain_v)

            step_v = self._scroll_vel * gain_v * (mult / 10.0) * 0.72
            self._scroll_accum += step_v

        pixels_v = int(self._scroll_accum)
        if pixels_v != 0:
            clamped_v = max(-_SCROLL_MAX_PX, min(_SCROLL_MAX_PX, pixels_v))
            _mouse.scroll(clamped_v)
            self._scroll_accum -= pixels_v

        # ── Horizontal scroll ────────────────────────────────────
        delta_x = -(raw_x - self._scroll_anchor_x) * self._screen_w
        self._scroll_vel_h += 0.65 * (delta_x - self._scroll_vel_h)

        speed_h = abs(self._scroll_vel_h)
        if speed_h < 1.0:
            self._scroll_vel_h *= 0.3
            self._scroll_accum_h *= 0.3
        else:
            if speed_h <= 5.0:
                gain_h = 0.35 + 0.13 * speed_h
            else:
                gain_h = 1.0 + 0.085 * ((speed_h - 5.0) ** 1.35)
            gain_h = min(12.0, gain_h)

            step_h = self._scroll_vel_h * gain_h * (mult / 10.0) * 0.72
            self._scroll_accum_h += step_h

        pixels_h = int(self._scroll_accum_h)
        if pixels_h != 0:
            clamped_h = max(-_SCROLL_MAX_PX, min(_SCROLL_MAX_PX, pixels_h))
            _mouse.scroll_h(clamped_h)
            self._scroll_accum_h -= pixels_h

        self._scroll_anchor_x = raw_x
        self._scroll_anchor_y = raw_y
