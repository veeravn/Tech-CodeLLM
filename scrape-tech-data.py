import requests
from bs4 import BeautifulSoup
import json
import time
import schedule
import threading
from azure.storage.blob import BlobServiceClient
import os

# Hugging Face API Configuration
HF_TOKEN = os.getevn("HF_TOKEN")  # Replace with your Hugging Face token
HF_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "bigcode/starcoderdata"
CONFIG = "default"
SPLIT = "train"
TOTAL_LENGTH = 1000000  # Max number of rows to fetch
BATCH_SIZE = 5000  # Fetch and save in batches

# Azure Blob Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "tech-instruction-dataset"
BLOB_NAME = "instruction_dataset.jsonl"


def fetch_starcoder_data(offset=0, length=TOTAL_LENGTH, batch_size=BATCH_SIZE):
    """Fetch StarCoder dataset in batches with checkpointing and retry on Hugging Face auth failure."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    dataset_file = "starcoder_dataset.jsonl"
    checkpoint_file = "starcoder_checkpoint.txt"
    MAX_RETRIES = 5
    RETRY_WAIT = 5  # seconds

    # Load last checkpoint if it exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            offset = int(f.read().strip())
        print(f"Resuming from checkpoint at row {offset}")
    else:
        print("No checkpoint found. Starting from row 0.")

    with open(dataset_file, "a", encoding="utf-8") as file:
        for start in range(offset, length, batch_size):
            retries = 0
            success = False

            while retries < MAX_RETRIES and not success:
                try:
                    params = {
                        "dataset": DATASET_NAME,
                        "config": CONFIG,
                        "split": SPLIT,
                        "offset": start,
                        "length": min(batch_size, length - start)
                    }

                    response = requests.get(HF_API_URL, headers=headers, params=params)

                    # Look for explicit failure message in text
                    if "Authentication check on the Hugging Face Hub failed" in response.text:
                        raise Exception("Authentication check on the Hugging Face Hub failed")
                    elif response.status_code == 200 and retries > 0:
                        retries = 0

                    response.raise_for_status()
                    rows = response.json().get("rows", [])

                    if not rows:
                        print("No more rows returned. Reached end of dataset.")
                        return  # Exit loop

                    # Write rows to file
                    for row in rows:
                        content = row.get("row", {}).get("content", "")
                        if content:
                            formatted_row = {
                                "instruction": "Explain the following code snippet.",
                                "input": "",
                                "response": content
                            }
                            file.write(json.dumps(formatted_row) + "\n")

                    # Save checkpoint
                    new_offset = start + len(rows)
                    with open(checkpoint_file, "w") as f:
                        f.write(str(new_offset))

                    print(f"Fetched and saved rows {start} to {new_offset}")
                    success = True

                except Exception as e:
                    retries += 1
                    print(f"Attempt {retries} failed for batch {start}: {e}")
                    if retries < MAX_RETRIES:
                        print(f"Retrying in {RETRY_WAIT} seconds...")
                        time.sleep(RETRY_WAIT)
                    else:
                        print(f"Max retries reached for batch {start}. Stopping.")
                        return

def scrape_docs(base_url, sitemap_url, source_name):
    """Scrape documentation from given sitemap"""
    response = requests.get(sitemap_url)
    dataset = []
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc") if base_url in loc.text]
        
        for url in urls:
            page_response = requests.get(url)
            if page_response.status_code == 200:
                page_soup = BeautifulSoup(page_response.text, "html.parser")
                title = page_soup.find("title").text.strip() if page_soup.find("title") else "No Title"
                paragraphs = page_soup.find_all("p")
                content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
                
                dataset.append({
                    "instruction": f"Explain the following {source_name} documentation: {title}",
                    "input": "",
                    "response": content
                })
            time.sleep(1)  # Avoid rate limiting
    
    return dataset


def scrape_stackoverflow():
    """Scrape top Azure questions from Stack Overflow"""
    api_url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=stackoverflow&tagged=azure"
    response = requests.get(api_url)
    dataset = []
    
    if response.status_code == 200:
        data = response.json().get("items", [])  # Get items, or return empty list
        for item in data:
            question = item.get("title", "No title provided")
            body = BeautifulSoup(item.get("body", "No content available"), "html.parser").text.strip()
            
            dataset.append({
                "instruction": question,
                "input": "",
                "response": body
            })
    return dataset

def scrape_github_issues():
    """Scrape issues from GitHub repositories"""
    repo = "microsoft/VirtualWAN"  # Example repository
    api_url = f"https://api.github.com/repos/{repo}/issues"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    dataset = []
    
    if response.status_code == 200:
        issues = response.json()
        for issue in issues:
            title = issue.get("title", "No title provided")
            body = issue.get("body", "No description available.")
            
            dataset.append({
                "instruction": f"Analyze the following GitHub issue: {title}",
                "input": "",
                "response": body
            })
    return dataset

def save_dataset_to_file(dataset, file_path="instruction_dataset.jsonl"):
    """Save dataset to a JSONL file"""
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in dataset:
            file.write(json.dumps(entry) + "\n")
    print(f"Dataset saved as {file_path}")

def upload_to_azure_blob(file_path):
    """Upload dataset file to Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print("Dataset uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        print(f"Failed to upload dataset: {e}")

def scrape_all_sources():
    """Fetch all data sources and upload to Azure"""
    print("Starting data scraping...")

    # Fetch StarCoder dataset
    fetch_starcoder_data()

    # Scrape documentation and other sources
    azure_dataset = scrape_docs("https://learn.microsoft.com/en-us/azure/", "https://learn.microsoft.com/sitemap.xml", "Microsoft Azure")
    aws_dataset = scrape_docs("https://docs.aws.amazon.com/", "https://docs.aws.amazon.com/sitemap.xml", "AWS")
    gcp_dataset = scrape_docs("https://cloud.google.com/", "https://cloud.google.com/sitemap.xml", "Google Cloud")
    gfg_dataset = scrape_docs("https://www.geeksforgeeks.org/", "https://www.geeksforgeeks.org/sitemap.xml", "GeeksforGeeks")
    
    dataset = azure_dataset + aws_dataset + gcp_dataset + gfg_dataset + scrape_stackoverflow() + scrape_github_issues()
    
    dataset_file = "instruction_dataset.jsonl"
    save_dataset_to_file(dataset, dataset_file)
    upload_to_azure_blob(dataset_file)

# Schedule the script to run daily at midnight
schedule.every().day.at("00:00").do(scrape_all_sources)

def run_scheduler():
    """Continuously run the scheduled task"""
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Run initial scrape
scrape_all_sources()

# Run the scheduler in a separate thread
thread = threading.Thread(target=run_scheduler)
thread.start()

print("Automated scraping system initialized. The dataset will update daily at midnight and upload to Azure Blob Storage.")
