"""
brain.py — Core command-processing logic for Mini Jarvis.

Responsibilities:
1. Match incoming commands against the ACTION_REGISTRY and VOICE_KEYWORDS
   from actions.py for instant, offline resolution (no API call needed).
2. Forward unmatched *voice* commands to a locally-running Ollama server
   for a conversational LLM reply.  Gesture-sourced commands that don't
   match a registered action are silently ignored (no LLM fallback).
3. Maintain a short rolling conversation history for context.
4. Enforce a MUTED flag — when muted, only the "mute" (toggle) command is
   processed; everything else is silently dropped.

All LLM traffic stays local via Ollama — **zero paid API keys required**.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from actions import ACTION_REGISTRY, VOICE_KEYWORDS, open_app
from tts_engine import TTSEngine


# ── Configuration ────────────────────────────────────────────────────
_OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
_CHAT_ENDPOINT = f"{_OLLAMA_URL}/api/chat"

# Maximum number of messages kept in the rolling conversation history.
_MAX_HISTORY = 10

# System prompt sent to the LLM for conversational replies.
_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful and concise personal assistant. "
    "Answer in one or two short sentences unless asked for more detail."
)


# ── Brain ─────────────────────────────────────────────────────────────

class Brain:
    """
    Core command processor.

    Resolves system commands locally via ACTION_REGISTRY; forwards
    unmatched voice commands to Ollama for an LLM-powered conversational
    reply.
    """

    def __init__(self, tts: TTSEngine):
        self.tts = tts
        self.muted: bool = False
        self._history: list[dict[str, str]] = []

    # ── public API ────────────────────────────────────────────────

    def process(self, command: dict) -> Optional[str]:
        """
        Process a single command dict (``{"source": ..., "text": ...}``).

        Returns a response string, or *None* if the command was silently
        ignored (e.g. unmatched gesture, or muted).
        """
        source: str = command.get("source", "")
        text: str   = command.get("text", "").strip()
        if not text:
            return None

        text_lower = text.lower()

        # ── Mute gate ─────────────────────────────────────────────
        # When muted, only allow the mute/unmute toggle through.
        if self.muted and text_lower not in ("mute", "unmute", "mute_jarvis"):
            print(f"[brain] (muted) Ignoring: \"{text}\"")
            return None

        # ── Check for "open <app>" (voice only, needs argument) ───
        if text_lower.startswith("open "):
            app_name = text[5:].strip()
            if app_name:
                return open_app(app_name, self.tts)

        # ── Direct action ID match (from gesture commands) ────────
        if text_lower in ACTION_REGISTRY:
            return self._execute_action(text_lower)

        # ── Voice keyword matching ────────────────────────────────
        # Check longest keywords first for better matching
        # (e.g. "volume up" before "volume")
        for keyword in sorted(VOICE_KEYWORDS, key=len, reverse=True):
            if keyword in text_lower:
                action_id = VOICE_KEYWORDS[keyword]
                return self._execute_action(action_id)

        # ── Fallback: LLM via Ollama (voice only) ────────────────
        if source == "voice":
            return self._ask_ollama(text)

        # Gesture commands with no action match → silently ignore.
        return None

    def toggle_mute(self) -> None:
        """Toggle the MUTED flag."""
        self.muted = not self.muted

    # ── internals ─────────────────────────────────────────────────

    def _execute_action(self, action_id: str) -> str:
        """Look up and execute an action from the registry."""
        # Special case: mute_jarvis is handled internally
        if action_id == "mute_jarvis":
            self.toggle_mute()
            state = "muted" if self.muted else "unmuted"
            self.tts.speak(f"I am now {state}.")
            return f"Now {state}."

        entry = ACTION_REGISTRY.get(action_id)
        if not entry:
            return ""

        fn = entry.get("fn")
        if fn is None:
            return ""

        result = fn(tts=self.tts)
        action_name = entry.get("name", action_id)
        print(f"[brain] Action: {action_name}")
        return result or ""

    def _ask_ollama(self, user_text: str) -> str:
        """
        Send *user_text* to the local Ollama server and return the
        assistant's reply.  Maintains a short rolling chat history.
        """
        # Append user message to history
        self._history.append({"role": "user", "content": user_text})

        # Trim history if it exceeds the cap
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
        ]

        try:
            resp = requests.post(
                _CHAT_ENDPOINT,
                json={
                    "model": _OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("message", {}).get("content", "").strip()

            if not reply:
                reply = "I'm not sure how to respond to that."

        except requests.ConnectionError:
            reply = (
                "I can't reach the Ollama server. "
                "Make sure it's running on " + _OLLAMA_URL
            )
        except requests.Timeout:
            reply = "The Ollama server took too long to respond."
        except Exception as exc:  # noqa: BLE001
            reply = f"LLM error: {exc}"

        # Store assistant reply in history
        self._history.append({"role": "assistant", "content": reply})

        # Speak the reply aloud
        self.tts.speak(reply)
        print(f"[brain] Jarvis: {reply}")
        return reply
