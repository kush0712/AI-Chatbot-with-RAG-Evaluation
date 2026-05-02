"""
evaluator.py
RAGAS-style evaluation of the RAG pipeline using Gemma 4 as the judge LLM.

Metrics:
  Faithfulness      — Does the answer only use information from the retrieved chunks?
                      High score = no hallucination.
  Answer Relevance  — Does the answer actually address the question asked?
  Context Precision — Were the retrieved chunks relevant to the question?

Each metric is scored 0.0–1.0 using Gemma 4 as an LLM-as-judge.
Overall score = harmonic mean (penalises any weak dimension).
"""

import asyncio
import json
import re
import os
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GEMMA_MODEL    = os.getenv("GEMMA_MODEL", "google/gemma-4-31b-it")
INVOKE_URL     = os.getenv(
    "NVIDIA_INVOKE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions"
)


# ── LLM Judge (non-streaming, deterministic) ──────────────────────────────────

async def _judge(prompt: str) -> dict:
    """
    Call Gemma 4 in non-streaming mode as an evaluator judge.
    Returns {"score": float, "reason": str}.
    Temperature = 0 for deterministic, reproducible scores.
    """
    payload = {
        "model": GEMMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(INVOKE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    raw = data["choices"][0]["message"]["content"]

    # Parse JSON from model output (it may wrap it in markdown code fences)
    json_match = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0.5))
            return {
                "score": round(min(max(score, 0.0), 1.0), 3),
                "reason": str(result.get("reason", "")).strip()[:300],
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: pull first numeric value from output
    num = re.search(r'(\d+\.?\d*)', raw)
    score = float(num.group(1)) if num else 0.5
    if score > 1.0:
        score /= 10.0  # handle "8/10" style outputs
    return {"score": round(min(max(score, 0.0), 1.0), 3), "reason": raw[:200].strip()}


# ── Metric 1: Faithfulness ────────────────────────────────────────────────────

async def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Faithfulness measures whether every claim in the answer is supported
    by the retrieved contexts. Score 1.0 = fully grounded, no hallucination.
    """
    ctx_block = "\n\n".join(f"[Passage {i+1}]:\n{c}" for i, c in enumerate(contexts))

    prompt = f"""You are an expert evaluator of AI systems. Evaluate FAITHFULNESS of the given answer.

FAITHFULNESS = every factual claim in the Answer must be directly supported by the Source Passages.
A score of 1.0 means the answer contains NO information beyond what is in the passages.
A score of 0.0 means the answer is entirely hallucinated or contradicts the passages.

QUESTION:
{question}

SOURCE PASSAGES:
{ctx_block}

ANSWER TO EVALUATE:
{answer}

Identify each claim in the answer and check if it is supported by the passages.
Respond with ONLY valid JSON — no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence justification>"}}"""

    return await _judge(prompt)


# ── Metric 2: Answer Relevance ────────────────────────────────────────────────

async def evaluate_answer_relevance(question: str, answer: str) -> dict:
    """
    Answer Relevance measures whether the answer actually addresses the question.
    Score 1.0 = completely on-topic, Score 0.0 = completely off-topic.
    """
    prompt = f"""You are an expert evaluator of AI systems. Evaluate ANSWER RELEVANCE.

ANSWER RELEVANCE = how directly and completely the answer addresses the question asked.
A score of 1.0 means the answer fully addresses the question with no irrelevant content.
A score of 0.0 means the answer is completely off-topic or refuses to answer.

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Consider: Does the answer address what was asked? Is the response focused?
Respond with ONLY valid JSON — no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence justification>"}}"""

    return await _judge(prompt)


# ── Metric 3: Context Precision ───────────────────────────────────────────────

async def evaluate_context_precision(question: str, contexts: list[str]) -> dict:
    """
    Context Precision measures whether the retrieved passages are relevant
    to the question. Score 1.0 = all chunks useful, Score 0.0 = all noise.
    """
    ctx_block = "\n\n".join(f"[Passage {i+1}]:\n{c}" for i, c in enumerate(contexts))

    prompt = f"""You are an expert evaluator of AI retrieval systems. Evaluate CONTEXT PRECISION.

CONTEXT PRECISION = what fraction of the retrieved passages are actually useful for answering the question.
A score of 1.0 means every retrieved passage is relevant to the question.
A score of 0.0 means none of the passages contain useful information for the question.

QUESTION:
{question}

RETRIEVED PASSAGES:
{ctx_block}

For each passage, judge if it is relevant to the question. Compute the fraction.
Respond with ONLY valid JSON — no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence justification>"}}"""

    return await _judge(prompt)


# ── Full Evaluation ───────────────────────────────────────────────────────────

async def run_full_evaluation(
    question: str,
    answer: str,
    contexts: list[dict]   # [{"text": str, "source": str, "page": int}]
) -> dict:
    """
    Run all three RAGAS metrics concurrently and return a unified report.
    """
    texts = [c["text"] for c in contexts]

    faithfulness_r, relevance_r, precision_r = await asyncio.gather(
        evaluate_faithfulness(question, answer, texts),
        evaluate_answer_relevance(question, answer),
        evaluate_context_precision(question, texts),
    )

    # Harmonic mean: penalises any single weak metric
    scores = [faithfulness_r["score"], relevance_r["score"], precision_r["score"]]
    overall = (
        len(scores) / sum(1.0 / s for s in scores)
        if all(s > 0 for s in scores)
        else 0.0
    )

    return {
        "question": question,
        "answer":   answer,
        "overall_score": round(overall, 3),
        "metrics": {
            "faithfulness": {
                "score":       faithfulness_r["score"],
                "reason":      faithfulness_r["reason"],
                "label":       "Faithfulness",
                "description": "Does the answer contain ONLY information from the retrieved passages? Measures hallucination.",
            },
            "answer_relevance": {
                "score":       relevance_r["score"],
                "reason":      relevance_r["reason"],
                "label":       "Answer Relevance",
                "description": "Does the answer directly address the question that was asked?",
            },
            "context_precision": {
                "score":       precision_r["score"],
                "reason":      precision_r["reason"],
                "label":       "Context Precision",
                "description": "Were the retrieved document passages actually relevant to the question?",
            },
        },
        "chunks_used": len(contexts),
        "sources": [
            {"filename": c["source"], "page": c["page"]}
            for c in contexts
        ],
    }
