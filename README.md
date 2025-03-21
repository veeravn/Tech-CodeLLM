# 🧠 Tech-CodeLLM: Fine-Tuned CodeLLaMA on Tech Industry Data

This project builds a domain-specific generative AI using [CodeLLaMA-13B](https://huggingface.co/codellama/CodeLlama-13b-hf), fine-tuned on technical documentation, developer Q&A, and GitHub issues related to cloud platforms (Azure, AWS, GCP) and programming resources (e.g., GeeksforGeeks, Stack Overflow). It uses LoRA for efficient fine-tuning and leverages Azure ML for training, storage, and model registry.

---

## 📁 Project Structure

```text
.
├── azure_ml_finetune.py       # Fine-tunes and registers the model on Azure ML
├── scrape-tech-data.py        # Scrapes tech-related sources and uploads dataset to Azure Blob Storage
├── instruction_dataset.jsonl  # (Generated) Dataset for instruction tuning
├── output_finetuned_model/    # (Generated) Directory with trained model and tokenizer
├── logs/                      # (Generated) Training logs
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start

### ✅ Prerequisites

- Azure Subscription
- Azure ML Workspace and Compute Cluster (ND A100 v4 recommended)
- Azure Blob Storage Account
- Python 3.8+
- HuggingFace Transformers, Datasets, PEFT
- Azure ML SDK
- BeautifulSoup, Requests

Install dependencies:

```bash
pip install transformers datasets peft azureml-core azure-storage-blob beautifulsoup4 schedule
```

---

## 🧾 Step 1: Scrape Instruction Data

Run the following script to collect, preprocess, and upload tech-related instruction data to Azure Blob Storage:

```bash
python scrape-tech-data.py
```

Sources include:

- Microsoft Learn (Azure)
- AWS Docs
- Google Cloud Docs
- GeeksforGeeks
- Stack Overflow (tagged: azure)
- GitHub Issues (e.g., microsoft/VirtualWAN)

The script runs daily via a background scheduler and uploads the dataset to:

```text
Azure Blob Storage -> your_container_name/instruction_dataset.jsonl
```

---

## 🏋️ Step 2: Fine-Tune CodeLLaMA with LoRA

Once the dataset is available in Azure ML, run:

```bash
python azure_ml_finetune.py
```

This will:

1. Connect to your Azure ML workspace.
2. Download the dataset from Azure ML Dataset Registry.
3. Load and tokenize the dataset.
4. Apply LoRA to the `q_proj` and `v_proj` modules of CodeLLaMA-13B.
5. Fine-tune the model using Hugging Face Trainer.
6. Save and register the trained model in Azure ML Model Registry.

---

## 🧠 Model Details

- **Base Model**: `codellama/CodeLlama-13b-hf`
- **Tuning Strategy**: LoRA
- **Dataset Format** (JSONL):
  ```json
  {
    "instruction": "Explain the following Azure documentation: ...",
    "input": "",
    "response": "..."
  }
  ```

---

## 📦 Azure ML Configuration

- **Compute Target**: `codegen-cluster` (must support A100 GPUs)
- **Dataset**: Registered as `instruction_dataset` in Azure ML
- **Environment**: Custom Conda environment with `transformers`, `datasets`, `peft`

---

## 📌 Notes

- Model evaluation currently uses a 100-sample slice of the training set. Replace with a validation set for production.
- Scraping respects rate limits via `time.sleep(1)` between requests.
- Customize the GitHub repo target in `scrape_github_issues()` if needed.

---

## 🔐 Authentication

Future versions of this project will include an API deployment with JWT authentication.

---

## 🛠 TODOs

- [ ] Add inference API with JWT-based access
- [ ] Set up CI/CD for retraining and redeploying
- [ ] Web-based interface for interactive usage

---

## 📬 Feedback & Contributions

Pull requests and issues are welcome! This project aims to support developers working with cloud platforms and modern software infrastructure.