# JKUAT-RAGAS Evaluation Suite

This repository implements an end-to-end evaluation workflow for the Veritas RAG system using a custom question set and RAGAS metrics, with interactive dashboard visualization.

## Project Overview

The project contains three main evaluation steps plus a dashboard:

1. `1_parse_questions.py`
   - Parses selected questions and ground truth answers from `Veritas_Project_Evaluation_dataset.pdf`
   - Produces `questions.json`

2. `2_query_rag.js`
   - Sends each question to a local RAG server at `http://localhost:3001/ask`
   - Stores the server response in `rag_outputs.json`

3. `3_evaluate.py`
   - Loads the RAG output JSON
   - Computes RAGAS evaluation metrics using Groq's OpenAI-compatible API
   - Writes full per-sample records to `ragas_results.json` (question, contexts, response, reference + all metric scores)
   - Writes a human-readable report to `ragas_report.txt`

4. `dashboard.py` (optional)
   - Interactive Streamlit dashboard for visualizing aggregated metrics and per-sample results
   - Displays metric summaries, bar charts, and detailed sample tables
   - Supports filtering and row selection

## Files in this repository

- `Veritas_Project_Evaluation_dataset.pdf` — source PDF with questions and ground truth answers
- `1_parse_questions.py` — extracts and saves selected questions in JSON format
- `questions.json` — parsed question set (generated)
- `2_query_rag.js` — queries the RAG server and saves server responses
- `rag_outputs.json` — RAG server outputs with question, answer, context, and ground truth
- `3_evaluate.py` — evaluates outputs using RAGAS metrics and Groq
- `dashboard.py` — Streamlit dashboard for visualization
- `ragas_results.json` — full per-sample evaluation results (generated)
- `ragas_report.txt` — human-readable evaluation report (generated)
- `evaluation_summary.json` — additional result summary data (if present)
- `package.json` — config file for Node.js module type
- `requirements.txt` — Python dependencies for dashboard and evaluation

## Requirements

### Python

- Python 3.10+ recommended
- Required Python packages (see `requirements.txt`):
  - `pypdf`
  - `ragas`
  - `datasets`
  - `openai`
  - `langchain_openai`
  - `langchain_community`
  - `sentence-transformers`
  - `pandas`
  - `numpy`
  - `streamlit` (for dashboard)

The `3_evaluate.py` script will automatically load values from `.env` if present.

### Node.js

- Node 18+ (bundled `fetch` is used)
- No additional npm dependencies are required for `2_query_rag.js` as written.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` in the project root:

```bash
cp .env.example .env
```

3. Add your Groq API key and optional Hugging Face token to `.env`.

4. If you prefer not to use `.env`, export the same values as environment variables before running the evaluation script.

5. Do not commit `.env` to GitHub. This repository includes `.gitignore` to ignore `.env`.

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

- `ragas_results.json` — contains full per-sample records with all metrics
- `ragas_report.txt` — human-readable summary report

### Step 4: View results on the dashboard (optional)

```bash
streamlit run dashboard.py
```

This launches an interactive dashboard showing:
- Aggregate metric scores with visual progress bars
- Bar chart of mean metric values
- Detailed per-sample table with filtering options
- Full question, response, contexts, and metrics for each sample

## Evaluation Metrics

The evaluation uses two core RAGAS metrics (token-efficient for Groq API):

- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Context Recall** — Were all needed information contexts retrieved?

(Previous runs may also include Answer Relevancy and Answer Correctness where available.)

## Output Format

`ragas_results.json` now contains full per-sample records:

```json
[
  {
    "question": "What is the Year 1 tuition fee for BSc Computer Science?",
    "retrieved_contexts": ["Context #1...", "Context #2..."],
    "response": "The Year 1 tuition fee is KES 148,000...",
    "reference": "KES 148,000.",
    "faithfulness": 1.0,
    "answer_relevancy": 0.9896,
    "context_recall": 1.0,
    "answer_correctness": NaN
  },
  ...
]
```

Each record includes the original RAG output plus computed metric scores.

## Environment Variables

The `.env` file supports these values:

- `GROQ_API_KEY` — Groq OpenAI-compatible API key used by `3_evaluate.py`
- `HUGGINGFACEHUB_API_TOKEN` — Optional Hugging Face token used for model downloads and authentication

> Note: `3_evaluate.py` reads `.env` at startup, so you can keep sensitive keys out of your shell.

## Notes

- `2_query_rag.js` currently expects the RAG server to run on `http://localhost:3001/ask`.
- If you change the RAG server location, update `RAG_SERVER_URL` inside `2_query_rag.js`.
- Keep `.env` private. Do not commit secrets to version control.
