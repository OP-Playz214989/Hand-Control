#!/usr/bin/env python3
"""
main.py — Entry point for Mini Jarvis.

Usage:
    python main.py            Run the assistant (voice + gesture)
    python main.py --config   Open the gesture configuration menu

Starts the voice listener and queue-processor as daemon threads, speaks
a startup greeting, and runs the gesture listener on the MAIN thread
(required by macOS for camera access and OpenCV GUI windows).

Handles KeyboardInterrupt for clean shutdown.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time

# Disable MediaPipe GPU/Metal before importing mediapipe — fixes
# crashes on macOS where the Metal delegate is unavailable.
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read os.getenv at import time.
load_dotenv()

from actions import ACTION_REGISTRY              # noqa: E402
from brain import Brain                           # noqa: E402
from config_menu import run_config_menu           # noqa: E402
from gesture_config import load_config            # noqa: E402
from gesture_listener import GestureListener      # noqa: E402
from tts_engine import TTSEngine                  # noqa: E402
from voice_listener import VoiceListener          # noqa: E402


def _queue_processor(cmd_queue: queue.Queue, brain: Brain, stop_event: threading.Event) -> None:
    """Drain the command queue through brain.py (runs in a daemon thread)."""
    while not stop_event.is_set():
        try:
            command = cmd_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        print(f"[main] Command received: {command}")
        result = brain.process(command)
        if result:
            print(f"[main] Result: {result}")


def _print_mappings() -> None:
    """Print the active gesture → action mappings on startup."""
    mapping = load_config()
    print("\n  Active gesture mappings:")
    print("  ─────────────────────────────────────────")
    for gesture, action_id in mapping.items():
        action_name = ACTION_REGISTRY.get(action_id, {}).get("name", action_id)
        print(f"    {gesture:<15s} → {action_name}")
    print("  ─────────────────────────────────────────")
    print("  Run 'python main.py --config' to change.\n")


def main() -> None:
    # ── CLI arguments ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Mini Jarvis — Voice & Gesture Assistant")
    parser.add_argument(
        "--config", action="store_true",
        help="Open the gesture configuration menu instead of running the assistant.",
    )
    args = parser.parse_args()

    if args.config:
        run_config_menu()
        return

    # ── Shared resources ──────────────────────────────────────────
    cmd_queue  = queue.Queue()
    stop_event = threading.Event()

    # ── TTS engine (background thread started internally) ─────────
    tts = TTSEngine()

    # ── Brain (command processor) ─────────────────────────────────
    brain = Brain(tts)

    # ── Voice listener (daemon thread) ────────────────────────────
    voice = VoiceListener(cmd_queue, stop_event)
    voice_thread = threading.Thread(target=voice.run, daemon=True, name="voice")
    voice_thread.start()

    # ── Queue processor (daemon thread) ───────────────────────────
    proc_thread = threading.Thread(
        target=_queue_processor,
        args=(cmd_queue, brain, stop_event),
        daemon=True,
        name="processor",
    )
    proc_thread.start()

    # ── Startup greeting ──────────────────────────────────────────
    tts.speak("Mini Jarvis online. Voice and gesture control active.")
    print("\n╔══════════════════════════════════════════════╗")
    print("║       🤖  Mini Jarvis is now online.        ║")
    print("║  Say the wake word or use hand gestures.     ║")
    print("║  Press Ctrl+C or 'q' in the camera to quit.  ║")
    print("╚══════════════════════════════════════════════╝")

    _print_mappings()

    # ── Gesture listener on the MAIN thread ───────────────────────
    # macOS requires camera access and OpenCV GUI (imshow/waitKey)
    # to run on the main thread.
    gesture = GestureListener(cmd_queue, stop_event)
    try:
        gesture.run()  # blocks until 'q' pressed or stop_event set
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt — shutting down …")

    # ── Cleanup ───────────────────────────────────────────────────
    stop_event.set()
    tts.speak("Goodbye.")
    time.sleep(1)       # let the goodbye utterance finish
    tts.shutdown()
    print("[main] Mini Jarvis offline. 👋")


if __name__ == "__main__":
    main()
