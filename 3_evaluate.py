"""
Script 3: Run RAGAS evaluation on RAG server outputs
Input:  rag_outputs.json  (from 2_query_rag.js)
Output: ragas_results.json + ragas_report.txt

Usage:
    export GROQ_API_KEY="gsk_your_key_here"
    python3 3_evaluate.py

Metrics computed:
  - Faithfulness       : Is the answer grounded in the retrieved context?
  - Answer Relevancy   : Does the answer address the question? (strictness=1 for Groq)
  - Context Recall     : Was all needed info retrieved?
  - Answer Correctness : Is the answer factually correct vs ground truth?
"""

import json
import os
import sys
import time


def load_dotenv(dotenv_path=".env"):
    """Load environment variables from .env file if it exists."""
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAG_OUTPUTS_FILE  = "rag_outputs.json"
RESULTS_JSON      = "ragas_results.json"
RESULTS_REPORT    = "ragas_report.txt"
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"
GROQ_BASE_URL     = "https://api.groq.com/openai/v1"
REQUEST_DELAY     = 180          # Seconds between requests (increased for stability)
MAX_RETRIES       = 3           # Retry failed requests
# ────────────────────────────────────────────────────────────────────────────


def check_dependencies():
    """Check if all required packages are installed."""
    missing = []
    for pkg in ["ragas", "datasets", "openai", "langchain_openai"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        print("Run: pip install " + " ".join(missing))
        sys.exit(1)


def build_groq_llm():
    """Create a LangChain LLM wrapper pointing to Groq's OpenAI-compatible API."""
    from langchain_openai import ChatOpenAI

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Run: export GROQ_API_KEY='gsk_your_key_here'")
        sys.exit(1)

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0,
        max_retries=MAX_RETRIES,
        timeout=300,  # Increased from 60 to 300 seconds
        default_headers={"groq-n": "1"},
    )


def build_groq_embeddings():
    """
    Use HuggingFace embeddings (runs locally, no API key needed).
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception:
        print("Installing sentence-transformers for embeddings...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "sentence-transformers", "-q"
        ])
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


def load_rag_outputs(path: str) -> list[dict]:
    """Load RAG outputs from JSON file and filter out empty answers."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = [d for d in data if d.get("answer", "").strip()]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"⚠️  Skipping {skipped} entries with empty answers")

    return valid


def build_ragas_dataset(data: list[dict]):
    """Convert RAG outputs into a RAGAS EvaluationDataset."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    for d in data:
        contexts = d.get("contexts", [])
        if not contexts:
            contexts = [d.get("answer", "")]

        contexts = [str(c) for c in contexts if c]

        sample = SingleTurnSample(
            user_input=d["question"],
            response=d["answer"],
            retrieved_contexts=contexts,
            reference=d.get("ground_truth", ""),
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_evaluation_single_sample(sample, metrics, ragas_llm, ragas_embeddings, sample_idx, total):
    """
    Run evaluation on a single sample with retry logic.
    Returns evaluation result or None if failed.
    """
    from ragas import evaluate
    from ragas import EvaluationDataset

    single_dataset = EvaluationDataset(samples=[sample])

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    for attempt in range(MAX_RETRIES):
        try:
            result = evaluate(dataset=single_dataset, metrics=metrics)
            print(f"  ✓ Sample {sample_idx}/{total} completed")
            return result
        except Exception as e:
            error_msg = str(e)
            if "n' : number must be at most 1" in error_msg:
                print(f"  ⚠️ Sample {sample_idx}/{total}: Groq n=1 limit, retrying...")
            elif "timeout" in error_msg.lower():
                print(f"  ⚠️ Sample {sample_idx}/{total}: Timeout, retrying ({attempt+1}/{MAX_RETRIES})...")
            elif "connection" in error_msg.lower():
                print(f"  ⚠️ Sample {sample_idx}/{total}: Connection error, retrying ({attempt+1}/{MAX_RETRIES})...")
            elif "rate limit" in error_msg.lower():
                print(f"  ⚠️ Sample {sample_idx}/{total}: Rate limit, waiting 60s...")
                time.sleep(60)
            else:
                print(f"  ⚠️ Sample {sample_idx}/{total}: {error_msg[:100]}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                print(f"  ✗ Sample {sample_idx}/{total}: Failed after {MAX_RETRIES} attempts")
                return None


def run_evaluation(dataset, llm, embeddings):
    """Run RAGAS metrics one sample at a time to avoid rate limits."""
    # Import the new metric classes (not the deprecated functions)
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        ContextRecall,
        AnswerCorrectness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Define metrics with strictness=1 to avoid n=3 requests from Groq
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),  # Only 1 generation, not 3
        ContextRecall(),
        AnswerCorrectness(),
    ]

    print(f"\nRunning evaluation on {len(dataset)} samples (one at a time)...")
    print(f"Metrics: {[type(m).__name__ for m in metrics]}\n")

    all_results = []
    total = len(dataset)

    for idx, sample in enumerate(dataset, 1):
        print(f"\nProcessing sample {idx}/{total}:")
        print(f"  Question: {sample.user_input[:60]}...")

        result = run_evaluation_single_sample(
            sample, metrics, ragas_llm, ragas_embeddings, idx, total
        )

        all_results.append(result)

        if idx < total:
            print(f"  Waiting {REQUEST_DELAY} seconds before next sample...")
            time.sleep(REQUEST_DELAY)

    successful = [r for r in all_results if r is not None]
    print(f"\n✅ Successfully evaluated {len(successful)}/{total} samples")

    if successful:
        return combine_results(successful, total)
    else:
        return None


def combine_results(results_list, total_samples):
    """Combine multiple evaluation results into one aggregated result."""
    import pandas as pd

    all_scores = []
    for result in results_list:
        if result is not None:
            scores = result.scores if hasattr(result, 'scores') else {}
            all_scores.append(scores)

    if not all_scores:
        return None

    df = pd.DataFrame(all_scores)

    aggregated_scores = {}
    for col in df.columns:
        aggregated_scores[col] = df[col].mean()

    class MockResult:
        def __init__(self, scores, df):
            self.scores = scores
            self.to_pandas = lambda: df

    return MockResult(aggregated_scores, df)


def save_results(results, raw_data: list[dict]):
    """Save results to JSON and human-readable report."""
    import pandas as pd

    if results is None:
        print("No results to save.")
        return

    try:
        results_df = results.to_pandas()
        results_dict = results_df.to_dict(orient="records")

        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save JSON results: {e}")
        results_dict = []

    scores = results.scores if hasattr(results, 'scores') else {}

    report_lines = [
        "=" * 60,
        "  RAGAS EVALUATION REPORT — Veritas RAG System",
        "  JKUAT Institutional Knowledge Base",
        "=" * 60,
        f"  Total samples attempted : {len(raw_data)}",
        f"  Samples successfully evaluated : {len(results_dict) if results_dict else 0}",
        f"  LLM judge               : {GROQ_MODEL} (via Groq)",
        f"  Embeddings              : all-MiniLM-L6-v2 (local)",
        "=" * 60,
        "",
        "AGGREGATE SCORES",
        "-" * 40,
    ]

    metric_descriptions = {
        "faithfulness":       "Faithfulness       (answer grounded in context?)",
        "answer_relevancy":   "Answer Relevancy   (answer addresses question?)",
        "context_recall":     "Context Recall     (all needed info retrieved?)",
        "answer_correctness": "Answer Correctness (factually correct vs GT?)",
    }

    for key, label in metric_descriptions.items():
        if isinstance(scores, dict):
            score = scores.get(key, "N/A")
        else:
            score = getattr(scores, key, "N/A") if hasattr(scores, key) else "N/A"

        if isinstance(score, float):
            bar = "█" * int(score * 20)
            report_lines.append(f"  {label}: {score:.4f}  |{bar:<20}|")
        else:
            report_lines.append(f"  {label}: {score}")

    report_lines += [
        "",
        "SCORE INTERPRETATION",
        "-" * 40,
        "  0.0 – 0.4  : Poor  — significant issues",
        "  0.4 – 0.6  : Fair  — needs improvement",
        "  0.6 – 0.8  : Good  — working reasonably well",
        "  0.8 – 1.0  : Excellent",
        "",
        "=" * 60,
        f"  Full results saved to : {RESULTS_JSON}",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)

    with open(RESULTS_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✅ Report saved to: {RESULTS_REPORT}")
    print(f"✅ JSON results saved to: {RESULTS_JSON}")


def main():
    check_dependencies()

    print("Loading RAG outputs...")
    raw_data = load_rag_outputs(RAG_OUTPUTS_FILE)
    print(f"Loaded {len(raw_data)} valid samples")

    print("Setting up Groq LLM...")
    llm = build_groq_llm()

    print("Setting up embeddings (local MiniLM)...")
    embeddings = build_groq_embeddings()

    print("Building RAGAS dataset...")
    dataset = build_ragas_dataset(raw_data)

    results = run_evaluation(dataset, llm, embeddings)

    save_results(results, raw_data)


if __name__ == "__main__":
    main()