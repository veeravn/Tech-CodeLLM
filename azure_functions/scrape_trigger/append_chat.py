import json
import os
from datetime import datetime

HISTORY_FILE = "chat_history.jsonl"

def append_chat_entry(instruction, response, input_text=""):
    if not instruction or not response:
        print("Instruction and response are required.")
        return False

    # Build new entry
    entry = {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "response": response.strip(),
        "timestamp": datetime.now().isoformat() + "Z"
    }

    seen_instructions = set()

    # Load existing chat history to detect duplicates
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing = json.loads(line.strip())
                    seen_instructions.add(existing.get("instruction", "").strip())
                except json.JSONDecodeError:
                    continue

    if entry["instruction"] in seen_instructions:
        print("Duplicate instruction found — entry not added.")
        return False

    # Append new entry
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print("✅ Chat entry added.")
    return True
