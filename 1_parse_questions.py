"""
Script 1: Parse test questions and ground truth from PDF
Output: questions.json (25 questions, 5 per category)

Usage:
    python3 1_parse_questions.py
"""

import json
import re
import sys

# ── CONFIG ─────────────────────────────────────────────────────────────────
PDF_PATH    = "Veritas_Project_Evaluation_dataset.pdf"
OUTPUT_PATH = "questions.json"

# 5 questions per category (25 total — safe for free tier)
SELECTED_IDS = (
    list(range(1,  6))  +   # Category 1: Fee Structures
    list(range(21, 26)) +   # Category 2: Academic Programmes
    list(range(41, 46)) +   # Category 3: Session Reporting
    list(range(61, 66)) +   # Category 4: Examinations & Policies
    list(range(81, 86))     # Category 5: Events & Timetable
)
# ───────────────────────────────────────────────────────────────────────────

try:
    from pypdf import PdfReader
except ImportError:
    print("Installing pypdf...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "-q"])
    from pypdf import PdfReader


def parse_qa_pdf(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    pattern = re.compile(
        r'(\d+)\.Q:(.*?)\nGT:(.*?)(?=\n\d+\.Q:|\Z)',
        re.DOTALL
    )

    qa_pairs = []
    for match in pattern.findall(full_text):
        number, question, ground_truth = match
        num = int(number)
        if num not in SELECTED_IDS:
            continue
        qa_pairs.append({
            "id": num,
            "question": question.strip().replace("\n", " "),
            "ground_truth": ground_truth.strip().replace("\n", " ")
        })

    # Sort by id to keep order
    qa_pairs.sort(key=lambda x: x["id"])
    return qa_pairs


def main():
    print(f"Parsing: {PDF_PATH}")
    qa_pairs = parse_qa_pdf(PDF_PATH)

    if not qa_pairs:
        print("ERROR: No Q&A pairs found. Check PDF format or path.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(qa_pairs)} Q&A pairs → {OUTPUT_PATH}")
    print(f"IDs selected: {[q['id'] for q in qa_pairs]}\n")

    categories = {
        "Fee Structures        (Q1–5)  ": list(range(1,  6)),
        "Academic Programmes   (Q21–25)": list(range(21, 26)),
        "Session Reporting     (Q41–45)": list(range(41, 46)),
        "Examinations          (Q61–65)": list(range(61, 66)),
        "Events & Timetable    (Q81–85)": list(range(81, 86)),
    }

    for cat_name, ids in categories.items():
        print(f"  {cat_name}")
        for q in qa_pairs:
            if q["id"] in ids:
                print(f"    Q{q['id']}: {q['question'][:65]}...")
        print()


if __name__ == "__main__":
    main()
