# 🤖 Mini Jarvis — Voice & Gesture Personal Assistant

A modular, offline-first personal assistant you control with **voice commands**
and **hand gestures** via your webcam.  Everything runs locally — **zero paid
API keys required**.

---

## Features

| Input   | How it works |
|---------|-------------|
| 🎤 Voice | Wake-word activation → Google Web Speech API (free) → command processing |
| ✋ Gesture | MediaPipe Hands via webcam → gesture classification → command mapping |
| 🧠 Agent | Voice commands & general queries → ultra-fast cloud Groq AI Agent (Llama 3.3 70B) with tool execution |
| 🔇 Mute  | Global mute flag — silences all responses except the unmute toggle |

### Gesture-to-Command Table

| Gesture        | Command      | Description |
|----------------|-------------|-------------|
| ✋ Open palm    | `stop`       | Stop current action |
| ✊ Fist         | `mute`       | Toggle mute on/off |
| 👍 Thumbs up   | `confirm`    | Confirm action |
| 👎 Thumbs down | `cancel`     | Cancel action |
| ✌️ Peace        | `screenshot` | Open native screenshot tool |
| ☝️ Pointing up  | `next`       | Skip to next |

Gestures require a **~0.6 s hold** and have a **1.5 s cooldown** to prevent
accidental spam.

### Voice System Commands

| Phrase               | Action |
|----------------------|--------|
| `stop`               | Stop current action |
| `mute` / `unmute`    | Toggle mute |
| `confirm`            | Confirm |
| `cancel`             | Cancel |
| `screenshot`         | Take a screenshot |
| `next`               | Skip to next |
| `open <app name>`    | Launch an app (macOS/Linux/Windows) |
| *anything else*      | Forwarded to Ollama LLM for a conversational reply |

---

## Prerequisites

### 1. System audio library (PortAudio)

SpeechRecognition's PyAudio backend needs PortAudio installed at the OS level.

**macOS** (Homebrew):
```bash
brew install portaudio
```

**Ubuntu / Debian**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows**: PyAudio ships pre-built wheels — no extra step needed.

### 2. Python ≥ 3.10

### 3. Ollama (for LLM conversational replies)

Install Ollama from <https://ollama.com> and pull a model:

```bash
# Install Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull llama3.2

# Start the server (runs on http://localhost:11434 by default)
ollama serve
```

> **Note:** If you prefer a different model, change `OLLAMA_MODEL` in your
> `.env` file.  Any Ollama-compatible model works.

---

## Installation

```bash
# 1. Clone or navigate into the project
cd assistant

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create your .env from the example
cp .env.example .env
# Edit .env if you want a different wake word or model
```

---

## Running

```bash
python main.py
```

You should see:

```
╔══════════════════════════════════════════════╗
║       🤖  Mini Jarvis is now online.        ║
║  Say the wake word or use hand gestures.     ║
║  Press Ctrl+C or 'q' in the camera to quit.  ║
╚══════════════════════════════════════════════╝
```

- **Voice**: Say *"Jarvis"* (or your configured wake word), wait for the
  prompt, then speak your command.
- **Gesture**: Hold a gesture in front of the webcam for ~0.6 seconds.
- **Quit**: Press `Ctrl+C` in the terminal, or press `q` in the camera window.

---

## Project Structure

```
assistant/
├── main.py              # Entry point — wires everything together
├── voice_listener.py    # Wake-word + command capture (background thread)
├── gesture_listener.py  # Webcam gesture recognition (background thread)
├── brain.py             # Command routing + Ollama LLM integration
├── tts_engine.py        # Thread-safe text-to-speech (pyttsx3)
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
└── README.md            # You are here
```

---

## Extending

### Add a new gesture

1. Define the classification logic in
   [`gesture_listener.py`](gesture_listener.py) → `classify_gesture()`.
2. Add the mapping in `GESTURE_COMMAND_MAP`.
3. If the new command needs a handler, add it to
   [`brain.py`](brain.py) → `SYSTEM_COMMANDS` and write a `_handle_*` function.

### Add a new system command

1. Add the keyword to `SYSTEM_COMMANDS` in [`brain.py`](brain.py).
2. Write a `_handle_*` function and wire it in `_dispatch_system()`.

---

## FAQ

**Q: Do I need an API key?**
A: You only need a free Groq API key from [console.groq.com](https://console.groq.com/keys) added as `GROQ_API_KEY` in `.env`. Everything else (gesture tracking, voice recognition, TTS, system actions) requires zero paid keys.

**Q: Can I use a different model?**
A: Yes — set `GROQ_MODEL` in `.env` to any Groq-supported model (e.g. `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`). You can also switch to local Ollama by setting `LLM_PROVIDER=ollama`.

---

## License

MIT — do whatever you want with it. 🚀
