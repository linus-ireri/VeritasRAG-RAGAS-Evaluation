/**
 * Script 2: Query VeritasRAG server for each question and collect outputs
 * Input:  questions.json       (from 1_parse_questions.py)
 * Output: rag_outputs.json
 *
 * Usage:
 *   node 2_query_rag.js
 */

import fs from "fs";

// ── CONFIG ──────────────────────────────────────────────────────────────────
const QUESTIONS_FILE = "./questions.json";
const OUTPUT_FILE    = "./rag_outputs.json";
const RAG_SERVER_URL = "http://localhost:3001/ask";
const DELAY_MS       = 4000; // 4s between requests (safe for free tier)
// ────────────────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function queryRAG(question) {
  const response = await fetch(RAG_SERVER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  const data = await response.json();

  // Server returns: { answer: "...", context: ["Context #1:\n...", ...] }
  const answer   = data.answer  ?? "[No answer]";
  const contexts = Array.isArray(data.context) ? data.context : [answer];

  return { answer, contexts };
}

async function main() {
  if (!fs.existsSync(QUESTIONS_FILE)) {
    console.error(`ERROR: ${QUESTIONS_FILE} not found. Run 1_parse_questions.py first.`);
    process.exit(1);
  }

  const questions = JSON.parse(fs.readFileSync(QUESTIONS_FILE, "utf-8"));
  const totalTime = ((questions.length - 1) * DELAY_MS / 1000).toFixed(0);

  console.log(`Loaded ${questions.length} questions`);
  console.log(`Server: ${RAG_SERVER_URL}`);
  console.log(`Delay: ${DELAY_MS}ms — estimated time: ~${totalTime}s\n`);

  const results = [];
  const failed  = [];

  for (let i = 0; i < questions.length; i++) {
    const { id, question, ground_truth } = questions[i];
    process.stdout.write(`[${String(i+1).padStart(2)}/${questions.length}] Q${String(id).padStart(2)}: ${question.slice(0, 55).padEnd(55)} `);

    try {
      const { answer, contexts } = await queryRAG(question);
      results.push({ id, question, answer, contexts, ground_truth });
      console.log("✓");
    } catch (err) {
      console.log(`✗  ${err.message}`);
      failed.push({ id, error: err.message });
      results.push({ id, question, answer: "", contexts: [], ground_truth, error: err.message });
    }

    if (i < questions.length - 1) await sleep(DELAY_MS);
  }

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2), "utf-8");

  console.log(`\n─────────────────────────────────────────`);
  console.log(`✅ Done: ${results.length - failed.length}/${results.length} successful`);
  if (failed.length > 0) {
    console.log(`⚠️  Failed (${failed.length}):`);
    failed.forEach((f) => console.log(`   Q${f.id}: ${f.error}`));
  }
  console.log(`📄 Saved to: ${OUTPUT_FILE}`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
