#!/usr/bin/env python3
"""
config_menu.py — Interactive terminal menu for gesture configuration.

Lets the user:
  1. View current gesture → action mappings
  2. Reassign any gesture to any action from the registry
  3. Reset all mappings to factory defaults
  4. Save and exit

Run standalone:   python config_menu.py
Or via main.py:   python main.py --config
"""

from __future__ import annotations

from actions import ACTION_REGISTRY
from gesture_config import (
    AVAILABLE_GESTURES,
    get_default_config,
    load_config,
    save_config,
)


# ── Pretty names for gestures ────────────────────────────────────────
_GESTURE_DISPLAY = {
    "open_palm":   "✋ Open Palm",
    "fist":        "✊ Fist",
    "thumbs_up":   "👍 Thumbs Up",
    "thumbs_down": "👎 Thumbs Down",
    "peace":       "✌️  Peace",
    "pointing_up": "☝️  Pointing Up",
}


def _print_header() -> None:
    print("\n╔═══════════════════════════════════════════════╗")
    print("║     🤖  Mini Jarvis — Gesture Configuration   ║")
    print("╚═══════════════════════════════════════════════╝\n")


def _print_current_map(mapping: dict[str, str]) -> None:
    """Display the current gesture → action table."""
    print("  ┌──────────────────────┬──────────────────────────────┐")
    print("  │  Gesture             │  Action                      │")
    print("  ├──────────────────────┼──────────────────────────────┤")
    for gesture in AVAILABLE_GESTURES:
        action_id = mapping.get(gesture, "do_nothing")
        action_entry = ACTION_REGISTRY.get(action_id, {})
        action_name = action_entry.get("name", action_id)
        category = action_entry.get("category", "")
        g_display = _GESTURE_DISPLAY.get(gesture, gesture)
        print(f"  │  {g_display:<19s}│  [{category}] {action_name:<21s}│")
    print("  └──────────────────────┴──────────────────────────────┘\n")


def _print_action_list() -> None:
    """Display all available actions grouped by category."""
    categories: dict[str, list[tuple[str, str]]] = {}
    for action_id, entry in ACTION_REGISTRY.items():
        cat = entry.get("category", "Other")
        categories.setdefault(cat, []).append((action_id, entry["name"]))

    print("  Available actions:\n")
    idx = 1
    action_index: list[str] = []
    for cat in ["Volume", "Brightness", "Media", "Window", "System", "Jarvis"]:
        if cat not in categories:
            continue
        print(f"    ── {cat} ──")
        for action_id, action_name in categories[cat]:
            print(f"      {idx:>2}. {action_name:<25s}  ({action_id})")
            action_index.append(action_id)
            idx += 1
        print()

    return action_index


def _reassign_gesture(mapping: dict[str, str]) -> dict[str, str]:
    """Interactive flow to reassign one gesture."""
    print("  Which gesture do you want to reassign?\n")
    for i, gesture in enumerate(AVAILABLE_GESTURES, 1):
        g_display = _GESTURE_DISPLAY.get(gesture, gesture)
        current = mapping.get(gesture, "do_nothing")
        current_name = ACTION_REGISTRY.get(current, {}).get("name", current)
        print(f"    {i}. {g_display:<20s}  (currently: {current_name})")

    print()
    try:
        choice = input("  Enter gesture number (or 'b' to go back): ").strip()
        if choice.lower() == "b":
            return mapping
        gesture_idx = int(choice) - 1
        if gesture_idx < 0 or gesture_idx >= len(AVAILABLE_GESTURES):
            print("  ⚠  Invalid choice.")
            return mapping
    except (ValueError, EOFError):
        return mapping

    gesture = AVAILABLE_GESTURES[gesture_idx]
    g_display = _GESTURE_DISPLAY.get(gesture, gesture)
    print(f"\n  Reassigning: {g_display}\n")

    action_index = _print_action_list()

    try:
        action_choice = input("  Enter action number (or 'b' to go back): ").strip()
        if action_choice.lower() == "b":
            return mapping
        action_idx = int(action_choice) - 1
        if action_idx < 0 or action_idx >= len(action_index):
            print("  ⚠  Invalid choice.")
            return mapping
    except (ValueError, EOFError):
        return mapping

    action_id = action_index[action_idx]
    action_name = ACTION_REGISTRY[action_id]["name"]
    mapping[gesture] = action_id
    print(f"\n  ✅  {g_display} → {action_name}")
    return mapping


def run_config_menu() -> None:
    """Main config menu loop."""
    mapping = load_config()

    while True:
        _print_header()
        _print_current_map(mapping)

        print("  Options:")
        print("    1. Reassign a gesture")
        print("    2. Reset all to defaults")
        print("    3. Save and exit")
        print("    4. Exit without saving")
        print()

        try:
            choice = input("  Choose [1-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "1":
            mapping = _reassign_gesture(mapping)
        elif choice == "2":
            mapping = get_default_config()
            print("\n  ✅  Reset to factory defaults.\n")
        elif choice == "3":
            save_config(mapping)
            print("\n  ✅  Configuration saved! Restart Mini Jarvis to apply.\n")
            break
        elif choice == "4":
            print("\n  Exited without saving.\n")
            break
        else:
            print("\n  ⚠  Invalid option.\n")


if __name__ == "__main__":
    run_config_menu()
