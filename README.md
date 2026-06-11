# JKUAT-RAGAS Evaluation Suite

This repository implements an end-to-end evaluation workflow for the Veritas RAG system using a custom question set and RAGAS metrics.

## Project Overview

The project contains three main steps:

1. `1_parse_questions.py`
   - Parses selected questions and ground truth answers from `Veritas_Project_Evaluation_dataset.pdf`
   - Produces `questions.json`

2. `2_query_rag.js`
   - Sends each question to a local RAG server at `http://localhost:3001/ask`
   - Stores the server response in `rag_outputs.json`

3. `3_evaluate.py`
   - Loads the RAG output JSON
   - Computes RAGAS evaluation metrics using Groq's OpenAI-compatible API
   - Writes aggregated JSON results to `ragas_results.json`
   - Writes a human-readable report to `ragas_report.txt`

## Files in this repository

- `Veritas_Project_Evaluation_dataset.pdf` — source PDF with questions and ground truth answers
- `1_parse_questions.py` — extracts and saves selected questions in JSON format
- `questions.json` — parsed question set (generated)
- `2_query_rag.js` — queries the RAG server and saves server responses
- `rag_outputs.json` — RAG server outputs with question, answer, context, and ground truth
- `3_evaluate.py` — evaluates outputs using RAGAS metrics and Groq
- `ragas_results.json` — aggregated evaluation results
- `evaluation_summary.json` — additional result summary data (if present)
- `package.json` — config file for Node.js module type

## Requirements

### Python

- Python 3.10+ recommended
- Required Python packages:
  - `pypdf`
  - `ragas`
  - `datasets`
  - `openai`
  - `langchain_openai`
  - `langchain_community`
  - `sentence-transformers`
  - `pandas`

The `3_evaluate.py` script will automatically load values from `.env` if present.

### Node.js

- Node 18+ (bundled `fetch` is used)
- No additional npm dependencies are required for `2_query_rag.js` as written.

## Setup

1. Copy `.env.example` to `.env` in the project root:

```bash
cp .env.example .env
```

2. Add your Groq API key and optional Hugging Face token to `.env`.

3. If you prefer not to use `.env`, export the same values as environment variables before running the evaluation script.

4. Do not commit `.env` to GitHub. This repository includes `.gitignore` to ignore `.env`.

## Usage

### Step 1: Parse questions

```bash
python3 1_parse_questions.py
```

This generates `questions.json` from the source PDF.

### Step 2: Query the RAG server

Make sure a RAG server is running at `http://localhost:3001/ask`, then run:

```bash
node 2_query_rag.js
```

This generates `rag_outputs.json`.

### Step 3: Run RAGAS evaluation

```bash
python3 3_evaluate.py
```

This produces:

- `ragas_results.json`
- `ragas_report.txt`

## Environment Variables

The `.env` file supports these values:

- `GROQ_API_KEY` — Groq OpenAI-compatible API key used by `3_evaluate.py`
- `HUGGINGFACEHUB_API_TOKEN` — Optional Hugging Face token used for model downloads and authentication

> Note: `3_evaluate.py` reads `.env` at startup, so you can keep sensitive keys out of your shell.

## Notes

- `2_query_rag.js` currently expects the RAG server to run on `http://localhost:3001/ask`.
- If you change the RAG server location, update `RAG_SERVER_URL` inside `2_query_rag.js`.
- Keep `.env` private. Do not commit secrets to version control.
