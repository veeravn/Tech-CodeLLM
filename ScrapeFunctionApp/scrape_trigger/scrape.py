import requests, os, json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient
from .rotate_chat_history import rotate_chat_history

AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER_NAME = os.environ["CONTAINER_NAME"]

def scrape_stackoverflow():
    tags = ["azure", "aws", "cloud", "devops", "kubernetes"]
    dataset = []
    for tag in tags:
        resp = requests.get("https://api.stackexchange.com/2.3/questions", params={
            "order": "desc", "sort": "votes", "tagged": tag,
            "site": "stackoverflow", "filter": "!9_bDDxJY5", "pagesize": 5
        })
        if resp.status_code != 200:
            continue
        for q in resp.json().get("items", []):
            q_id = q["question_id"]
            title = q.get("title", "")
            q_body = BeautifulSoup(q.get("body", ""), "html.parser").text.strip()
            a_resp = requests.get(f"https://api.stackexchange.com/2.3/questions/{q_id}/answers", params={
                "order": "desc", "sort": "votes", "site": "stackoverflow", "filter": "!9_bDDxJY5"
            })
            answers = a_resp.json().get("items", []) if a_resp.status_code == 200 else []
            a_body = BeautifulSoup(answers[0]["body"], "html.parser").text.strip() if answers else "No answer"
            dataset.append({"instruction": title, "input": q_body, "response": a_body})
    return dataset

def load_recent_chat_history(path="chat_history.jsonl", window_days=7):
    if not os.path.exists(path): return []
    recent = []
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = datetime.fromisoformat(entry["timestamp"].replace("Z", ""))
                if ts >= cutoff:
                    recent.append(entry)
            except:
                continue
    return recent

def upload_to_blob(file_path, blob_name):
    client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob = client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    with open(file_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)
    print(f"Uploaded {blob_name} to Azure Blob.")

def scrape_and_trigger():
    rotate_chat_history()

    # Load previous entries if the file exists
    filename = "new_data_today.jsonl"
    seen_instructions = set()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    seen_instructions.add(entry.get("instruction", "").strip())
                except json.JSONDecodeError:
                    continue

    # Scrape new data
    dataset = scrape_stackoverflow() + load_recent_chat_history()

    # Filter out duplicates based on 'instruction'
    unique_dataset = []
    for entry in dataset:
        instruction = entry.get("instruction", "").strip()
        if instruction and instruction not in seen_instructions:
            unique_dataset.append(entry)
            seen_instructions.add(instruction)  # Track it immediately

    # Write the deduplicated dataset
    with open(filename, "w", encoding="utf-8") as f:
        for entry in unique_dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"Final dataset contains {len(unique_dataset)} new unique entries.")

    # Upload to Azure Blob
    upload_to_blob(filename, filename)

    # Trigger fine-tuning
    os.system(f"python3 continuous_train_trigger.py --data_path {filename}")

