"""
tts_engine.py — Thread-safe text-to-speech wrapper around pyttsx3.

pyttsx3's engine is NOT thread-safe: all calls must happen on the thread
that created it.  This module runs the engine on a dedicated daemon thread
and exposes a simple `speak(text)` function that any thread may call.
"""

import threading
import queue
import pyttsx3


class TTSEngine:
    """Wraps pyttsx3 in a dedicated background thread for thread safety."""

    def __init__(self, rate: int = 175, volume: float = 0.9):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Enqueue *text* to be spoken.  Safe to call from any thread."""
        if text:
            self._queue.put(text)

    def shutdown(self) -> None:
        """Signal the background thread to stop."""
        self._queue.put(None)

    # ── internals ─────────────────────────────────────────────────

    def _run(self) -> None:
        """Loop on the dedicated TTS thread — creates & owns the engine."""
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        while True:
            text = self._queue.get()
            if text is None:          # shutdown sentinel
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except RuntimeError:
                # Occasionally pyttsx3 raises if the engine is busy;
                # swallow it and keep going.
                pass
