import json
import os
from datetime import datetime, timedelta

HISTORY_FILE = "chat_history.jsonl"
ARCHIVE_FILE = "chat_history_archive.jsonl"
RETENTION_DAYS = 7

def rotate_chat_history():
    if not os.path.exists(HISTORY_FILE):
        print("No chat history found.")
        return

    recent_entries = []
    archived_entries = []

    now = datetime.utcnow()
    cutoff = now - timedelta(days=RETENTION_DAYS)

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                timestamp = datetime.fromisoformat(entry.get("timestamp", "").replace("Z", ""))
                if timestamp >= cutoff:
                    recent_entries.append(entry)
                else:
                    archived_entries.append(entry)
            except Exception as e:
                print(f"Skipping invalid line: {e}")
                continue

    # Rewrite recent entries to chat_history.jsonl
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        for entry in recent_entries:
            f.write(json.dumps(entry) + "\n")

    # Append archived entries to archive file
    if archived_entries:
        with open(ARCHIVE_FILE, "a", encoding="utf-8") as f:
            for entry in archived_entries:
                f.write(json.dumps(entry) + "\n")

    print(f"Kept {len(recent_entries)} recent entries.")
    print(f"Archived {len(archived_entries)} old entries.")

if __name__ == "__main__":
    rotate_chat_history()
