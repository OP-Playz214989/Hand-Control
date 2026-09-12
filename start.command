#!/usr/bin/env bash

# Resolve script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

cd "$DIR"

# Print banner
[ -n "$TERM" ] && [ "$TERM" != "dumb" ] && clear 2>/dev/null || true
echo "=========================================================="
echo "          🤖 MINI JARVIS — ONE-CLICK LAUNCHER             "
echo "=========================================================="
echo "Project Path: $DIR"
echo ""

# Find Python executable (prefer .venv)
if [ -f "$DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python3"
    echo "✔ Using virtual environment (.venv)"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
    echo "✔ Using system Python ($PYTHON_BIN)"
else
    echo "❌ Error: python3 could not be found."
    echo ""
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

# Check .env
if [ ! -f "$DIR/.env" ]; then
    if [ -f "$DIR/.env.example" ]; then
        echo "⚠️ .env file not found. Creating from .env.example..."
        cp "$DIR/.env.example" "$DIR/.env"
        echo "✔ Created .env file."
    fi
fi

echo "🚀 Starting Mini Jarvis..."
echo "----------------------------------------------------------"
echo "Tips: "
echo " • Show hand to camera to control cursor & gestures."
echo " • Press 'q' in the camera window or Ctrl+C to quit."
echo "----------------------------------------------------------"
echo ""

"$PYTHON_BIN" main.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "----------------------------------------------------------"
    echo "⚠️ Application exited with code $EXIT_CODE"
    echo "----------------------------------------------------------"
    read -n 1 -s -r -p "Press any key to close this window..."
fi
