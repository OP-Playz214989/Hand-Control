"""
brain.py — Core command-processing logic for Mini Jarvis.

Responsibilities:
1. Match incoming gestures against ACTION_REGISTRY for instant, offline resolution.
2. Route voice commands to a fast, cloud-based Groq AI Agent with full tool-calling
   capabilities (controlling applications, volume, brightness, media, and macOS).
3. If Groq API key is not configured, optionally fall back to a local Ollama server.
4. Maintain rolling conversation history for multi-turn voice interaction.
5. Enforce a MUTED flag — when muted, only the "mute" (toggle) command is processed.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import requests

from actions import (
    ACTION_REGISTRY,
    VOICE_KEYWORDS,
    brightness_down,
    brightness_up,
    close_window,
    fullscreen,
    lock_screen,
    media_next,
    media_play_pause,
    media_previous,
    minimize_window,
    mission_control,
    open_app,
    screenshot_action,
    scroll_down,
    scroll_up,
    show_desktop,
    spotlight,
    volume_down,
    volume_mute,
    volume_up,
)
from tts_engine import TTSEngine

try:
    import groq
except ImportError:
    groq = None


# ── Configuration ────────────────────────────────────────────────────
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
_GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
_CHAT_ENDPOINT = f"{_OLLAMA_URL}/api/chat"

# Maximum messages in rolling conversation history
_MAX_HISTORY = 10

# System prompt defining Jarvis personality and tool usage
_SYSTEM_PROMPT = (
    "You are Jarvis, a fast, intelligent, and proactive personal voice assistant for macOS. "
    "You can control the computer using your tools (e.g. open apps, adjust volume or brightness, "
    "control media playback, manage windows, lock the screen, take screenshots). "
    "When the user asks to perform an action on their computer, invoke the appropriate tool. "
    "Keep all spoken responses concise (1-2 short sentences max), clear, and natural for text-to-speech."
)

# ── Groq Tools Declaration ───────────────────────────────────────────
_GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open or launch any application installed on macOS (e.g. Spotify, Safari, Chrome, Notes, Terminal, Calculator, Visual Studio Code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "The common name of the macOS application to launch.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_volume",
            "description": "Adjust or mute macOS system output volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down", "mute"],
                        "description": "Whether to turn volume up, down, or toggle mute.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_brightness",
            "description": "Adjust display screen brightness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Whether to increase or decrease screen brightness.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_media",
            "description": "Control playback of music or videos (Spotify, Apple Music, YouTube, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play_pause", "next", "previous"],
                        "description": "Media command to execute.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_window",
            "description": "Control the current active window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["minimize", "close", "fullscreen"],
                        "description": "Window management action.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_control",
            "description": "Execute macOS system controls such as locking the screen, viewing desktop, or taking a screenshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["lock_screen", "show_desktop", "mission_control", "spotlight", "screenshot"],
                        "description": "System action to trigger.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_page",
            "description": "Scroll the active document, window, or web page up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction.",
                    }
                },
                "required": ["direction"],
            },
        },
    },
]


# ── Brain ─────────────────────────────────────────────────────────────

class Brain:
    """
    Core command processor.

    Resolves quick system gestures locally; forwards voice commands to
    an intelligent Groq Agent equipped with macOS tool execution.
    """

    def __init__(self, tts: TTSEngine):
        self.tts = tts
        self.muted: bool = False
        self._history: list[dict[str, str]] = []
        self._groq_client = None
        self._init_groq()

    def _init_groq(self) -> None:
        """Initialize the Groq client if an API key is present."""
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key and groq is not None:
            try:
                self._groq_client = groq.Groq(api_key=api_key)
                print(f"[brain] Groq Agent initialized with model {_GROQ_MODEL}")
            except Exception as exc:
                print(f"[brain] Error initializing Groq client: {exc}")
                self._groq_client = None

    # ── public API ────────────────────────────────────────────────

    def process(self, command: dict) -> Optional[str]:
        """
        Process a single command dict (``{"source": ..., "text": ...}``).

        Returns a response string, or *None* if silently ignored.
        """
        source: str = command.get("source", "")
        text: str   = command.get("text", "").strip()
        if not text:
            return None

        text_lower = text.lower()

        # ── Mute gate ─────────────────────────────────────────────
        if self.muted and text_lower not in ("mute", "unmute", "mute_jarvis"):
            print(f"[brain] (muted) Ignoring: \"{text}\"")
            return None

        # ── Gestures: Instant action ID match (<1ms) ─────────────
        if source == "gesture" and text_lower in ACTION_REGISTRY:
            return self._execute_action(text_lower)

        # ── Quick local commands (mute / stop) ────────────────────
        if text_lower in ("mute", "unmute", "mute jarvis", "mute_jarvis"):
            return self._execute_action("mute_jarvis")

        # ── Voice: Process via Groq Agent (with tools & intelligence) ───
        if source == "voice":
            # Direct quick keyword match as fast path
            for keyword in sorted(VOICE_KEYWORDS, key=len, reverse=True):
                if text_lower == keyword:
                    action_id = VOICE_KEYWORDS[keyword]
                    return self._execute_action(action_id)

            return self._ask_agent(text)

        return None

    def toggle_mute(self) -> None:
        """Toggle the MUTED flag."""
        self.muted = not self.muted

    # ── internals ─────────────────────────────────────────────────

    def _execute_action(self, action_id: str) -> str:
        """Look up and execute an action from the registry."""
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
        print(f"[brain] Action executed: {action_name}")
        return result or ""

    def _ask_agent(self, user_text: str) -> str:
        """
        Forward user text to Groq Agent (or local Ollama if configured).
        """
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        provider = os.getenv("LLM_PROVIDER", _LLM_PROVIDER).lower()

        if provider == "groq" or (provider != "ollama" and api_key):
            return self._ask_groq(user_text)

        return self._ask_ollama(user_text)

    def _ask_groq(self, user_text: str) -> str:
        """
        Execute voice command through the Groq Agent with macOS tool execution.
        """
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            msg = "Please set your GROQ_API_KEY in the .env file to use the Groq agent."
            print(f"[brain] {msg}")
            self.tts.speak("Please add your Groq API key to the .env file.")
            return msg

        if self._groq_client is None:
            self._init_groq()

        if self._groq_client is None:
            msg = "Groq library not available. Please run: pip install groq"
            print(f"[brain] {msg}")
            self.tts.speak(msg)
            return msg

        # Append user message to rolling history
        self._history.append({"role": "user", "content": user_text})
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
        ]

        try:
            response = self._groq_client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=messages,
                tools=_GROQ_TOOLS,
                tool_choice="auto",
                temperature=0.6,
                max_tokens=250,
            )

            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None)

            action_results = []
            if tool_calls:
                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except Exception:
                        args = {}

                    res = self._execute_tool(fn_name, args)
                    if res:
                        action_results.append(res)

            # Determine response text
            reply_text = ""
            if message.content and message.content.strip():
                reply_text = message.content.strip()
            elif action_results:
                reply_text = action_results[-1]
            else:
                reply_text = "Done."

            # Update assistant reply in history
            self._history.append({"role": "assistant", "content": reply_text})

            # Speak and return
            self.tts.speak(reply_text)
            print(f"[brain] Jarvis (Groq): {reply_text}")
            return reply_text

        except Exception as exc:
            error_msg = str(exc)
            print(f"[brain] Groq Agent error: {error_msg}")
            if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                reply = "Your Groq API key appears to be invalid. Please check your .env file."
            elif "rate_limit" in error_msg.lower():
                reply = "Groq rate limit reached. Please wait a moment."
            else:
                reply = "I encountered an error processing that request with Groq."

            self.tts.speak(reply)
            return reply

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute macOS actions called by the Groq Agent."""
        print(f"[brain] Groq Agent calling tool '{tool_name}' with args: {args}")

        if tool_name == "open_application":
            app_name = args.get("app_name", "").strip()
            if app_name:
                return open_app(app_name, tts=None)

        elif tool_name == "control_volume":
            action = args.get("action", "")
            if action == "up":
                return volume_up(tts=None)
            elif action == "down":
                return volume_down(tts=None)
            elif action == "mute":
                return volume_mute(tts=None)

        elif tool_name == "control_brightness":
            action = args.get("action", "")
            if action == "up":
                return brightness_up(tts=None)
            elif action == "down":
                return brightness_down(tts=None)

        elif tool_name == "control_media":
            action = args.get("action", "")
            if action == "play_pause":
                return media_play_pause(tts=None)
            elif action == "next":
                return media_next(tts=None)
            elif action == "previous":
                return media_previous(tts=None)

        elif tool_name == "manage_window":
            action = args.get("action", "")
            if action == "minimize":
                return minimize_window(tts=None)
            elif action == "close":
                return close_window(tts=None)
            elif action == "fullscreen":
                return fullscreen(tts=None)

        elif tool_name == "system_control":
            action = args.get("action", "")
            if action == "lock_screen":
                return lock_screen(tts=None)
            elif action == "show_desktop":
                return show_desktop(tts=None)
            elif action == "mission_control":
                return mission_control(tts=None)
            elif action == "spotlight":
                return spotlight(tts=None)
            elif action == "screenshot":
                return screenshot_action(tts=None)

        elif tool_name == "scroll_page":
            direction = args.get("direction", "down")
            if direction == "up":
                return scroll_up(tts=None)
            else:
                return scroll_down(tts=None)

        return ""

    def _ask_ollama(self, user_text: str) -> str:
        """
        Fallback: Send *user_text* to local Ollama server if Groq is not configured.
        """
        self._history.append({"role": "user", "content": user_text})
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
                "Neither Groq API key nor local Ollama server is available. "
                "Add GROQ_API_KEY to your .env file."
            )
        except requests.Timeout:
            reply = "The local server took too long to respond."
        except Exception as exc:  # noqa: BLE001
            reply = f"LLM error: {exc}"

        self._history.append({"role": "assistant", "content": reply})
        self.tts.speak(reply)
        print(f"[brain] Jarvis (Ollama): {reply}")
        return reply
