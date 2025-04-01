import json
from datetime import datetime
from pathlib import Path

CHAT_HISTORY_PATH = Path("chat_history.jsonl")

def rotate_chat_history():
    if not CHAT_HISTORY_PATH.exists():
        print("⚠️ chat_history.jsonl not found")
        return

    cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
    retained = []

    with open(CHAT_HISTORY_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                if timestamp >= cutoff:
                    retained.append(record)
            except Exception as e:
                print(f"Skipping malformed line: {e}")

    with open(CHAT_HISTORY_PATH, "w") as f:
        for entry in retained:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Chat history rotated. Remaining entries: {len(retained)}")