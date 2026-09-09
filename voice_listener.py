"""
voice_listener.py — Continuous wake-word voice listener.

Runs in a background thread.  Continuously listens to the default
microphone via SpeechRecognition + the free Google Web Speech API.
When the configurable wake word is detected the listener captures the
*next* spoken phrase and pushes it onto a shared queue as:

    {"source": "voice", "text": "<transcribed command>"}

Design notes
────────────
• Uses ``sr.Recognizer.listen()`` with ambient noise adjustment so it
  works in typical room conditions.
• A short ``pause_threshold`` keeps latency tight.
• All configuration (wake word) comes from environment variables loaded
  by the caller (main.py) via python-dotenv.
"""

from __future__ import annotations

import os
import queue
import threading

import speech_recognition as sr


# ── defaults ──────────────────────────────────────────────────────────
_DEFAULT_WAKE_WORD = "jarvis"


class VoiceListener:
    """Listens for a wake word, then captures one command phrase."""

    def __init__(self, cmd_queue: queue.Queue, stop_event: threading.Event):
        self._queue = cmd_queue
        self._stop = stop_event
        self._wake_word = os.getenv("WAKE_WORD", _DEFAULT_WAKE_WORD).lower()
        self._recognizer = sr.Recognizer()
        # Shorter pause so the user doesn't have to wait too long.
        self._recognizer.pause_threshold = 1.0
        self._recognizer.dynamic_energy_threshold = True

    # ── public API ────────────────────────────────────────────────

    def run(self) -> None:
        """
        Entry point — call this from a daemon thread.

        Loop: adjust for noise → listen → recognise → check for wake
        word → if found, listen again for the actual command → push onto
        the shared queue.
        """
        mic = sr.Microphone()

        # One-time ambient noise calibration
        with mic as source:
            print("[voice] Calibrating for ambient noise …")
            self._recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("[voice] Calibration done.  Listening …")

        while not self._stop.is_set():
            try:
                # ── Phase 1: listen for the wake word ─────────────
                with mic as source:
                    audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=4)

                text = self._recognizer.recognize_google(audio).lower()  # type: ignore[arg-type]

                if self._wake_word not in text:
                    continue  # not the wake word → keep listening

                print(f"[voice] Wake word detected (\"{self._wake_word}\").")

                # ── Phase 2: capture the command ──────────────────
                print("[voice] Listening for command …")
                with mic as source:
                    cmd_audio = self._recognizer.listen(source, timeout=6, phrase_time_limit=8)

                command = self._recognizer.recognize_google(cmd_audio)  # type: ignore[arg-type]
                command_str: str = str(command).strip()

                if command_str:
                    print(f"[voice] Heard: \"{command_str}\"")
                    self._queue.put({"source": "voice", "text": command_str})

            except sr.WaitTimeoutError:
                # No speech detected within the timeout — just loop.
                continue
            except sr.UnknownValueError:
                # Speech was unintelligible.
                continue
            except sr.RequestError as exc:
                print(f"[voice] Google Speech API error: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[voice] Unexpected error: {exc}")
                continue

        print("[voice] Listener stopped.")
