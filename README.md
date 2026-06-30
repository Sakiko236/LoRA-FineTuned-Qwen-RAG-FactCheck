# LLM RAG FactCheck

This project implements a **Retrieval-Augmented Generation (RAG) pipeline combined with a LoRA fine-tuned Large Language Model (Qwen)** for factual verification and climate science fact-checking. 

The system retrieves relevant evidence for a given claim and generates a step-by-step reasoning chain (Chain-of-Thought) to classify the claim into one of four categories: `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or `DISPUTED`.

## 🚀 Key Features

- **RAG Pipeline**: Utilizes a dense retrieval setup with a Bi-encoder (`BAAI/bge-large-en-v1.5`) for candidate retrieval and a Cross-encoder (`Qwen/Qwen3-Reranker-0.6B`) for accurate reranking.
- **FAISS Integration**: High-performance similarity search for evidence retrieval using `faiss-cpu`.
- **LoRA Fine-Tuning**: Efficient instruction tuning of Qwen models (e.g., `Qwen/Qwen3.5-2B` or `Qwen2.5-7B-Instruct`) using Parameter-Efficient Fine-Tuning (PEFT) and 4-bit quantization.
- **Chain-of-Thought (CoT) Reasoning**: The model is trained to provide a step-by-step analysis before reaching a final classification, improving interpretability and accuracy.
- **Batch Processing & Evaluation**: Built-in scripts for batch prediction and comprehensive evaluation (F-score, Accuracy, Harmonic Mean).

## 📁 Project Structure

```text
├── data/
│   ├── cot.db                   # SQLite DB for Chain-of-Thought data
│   ├── cot.json                 # JSON formatted CoT data for fine-tuning
│   ├── train-claims.json        # Training dataset claims
│   ├── dev-claims.json          # Development dataset claims (with gold labels)
│   └── test-claims-unlabelled.json  # Unlabelled test set for prediction
├── fact_check/
│   ├── claim_verifier.py        # Core inference class wrapping the Model + LoRA + RAG
│   ├── evaluator.py             # Evaluation script comparing predictions vs ground truth
│   ├── predict.py               # Batch prediction script for generating test outputs
│   └── results/                 # Directory to store evaluation/prediction outputs
├── fine_tuning/
│   ├── build_cot.py             # Script to build Chain-of-Thought dataset
│   ├── build_evidence.py        # Script to prepare the SQLite evidence database
│   ├── fine_tuning.py           # Script to fine-tune Qwen using LoRA and 4-bit quantization
│   └── retrieve_evidence.py     # Utility to fetch evidence text
├── rag/
│   ├── faiss_builder.py         # Script to index the evidence database using FAISS
│   └── rag_pipeline.py          # Core RAG logic (Bi-encoder retrieval + Cross-encoder reranking)
├── model/                       # Directory where the LoRA adapters are saved (e.g., qwen-cot-lora-final)
├── .gitignore
└── requirements.txt             # Python dependencies
```

## 🛠️ Requirements & Installation

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment (optional but recommended).
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Main Dependencies**:
- `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`
- `sentence-transformers`, `faiss-cpu`
- `datasets`, `pandas`, `numpy`, `sqlite3`

## ⚙️ Usage Workflow

### 1. Data Preparation & RAG Setup

Before running the model, you need to prepare the evidence database and build the FAISS index.

```bash
# 1. Build the evidence SQLite database
python fine_tuning/build_evidence.py

# 2. Build the FAISS index for dense retrieval
python rag/faiss_builder.py
```
This will create `data/evidence.db`, `data/faiss_index.bin`, and `data/faiss_metadata.json`.

### 2. Model Fine-Tuning (Optional)

If you want to train your own LoRA adapter on the Chain-of-Thought dataset:

```bash
# Build the CoT dataset (if not already built)
python fine_tuning/build_cot.py

# Run the fine-tuning script
python fine_tuning/fine_tuning.py
```
This will train the model in 4-bit and save the LoRA weights into `model/qwen-cot-lora-final`.

### 3. Fact-Checking (Inference)

To run batch predictions on an unlabelled dataset (e.g., the test set) using the RAG pipeline and the fine-tuned model:

```bash
python fact_check/predict.py
```
The output, containing the claims, retrieved evidence IDs, and the predicted labels, will be saved in `fact_check/results/test-output.json`.

### 4. Evaluation

To evaluate the pipeline's performance on a labelled development set (measuring Evidence Retrieval F-score, Claim Classification Accuracy, and their Harmonic Mean):

```bash
python fact_check/evaluator.py
```
