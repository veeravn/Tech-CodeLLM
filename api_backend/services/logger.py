import json
import os
from datetime import datetime
from ..config import CHAT_HISTORY_PATH


def append_chat_entry(instruction, response, input_text=""):
    if not instruction or not response:
        return False

    entry = {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "response": response.strip(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    seen_instructions = set()
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing = json.loads(line.strip())
                    seen_instructions.add(existing.get("instruction", "").strip())
                except json.JSONDecodeError:
                    continue

    if entry["instruction"] in seen_instructions:
        return False

    with open(CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return True