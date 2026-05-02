"""
gemma_client.py
Async wrapper around the NVIDIA Inference API for Gemma 4 (31B).

Endpoint : https://integrate.api.nvidia.com/v1/chat/completions
Model    : google/gemma-4-31b-it
Auth     : Bearer token (NVIDIA API key)
Streaming: SSE (text/event-stream) — collected into a single string
Thinking : enable_thinking=True for extended reasoning on legal text
"""

import os
import json
import re
import httpx
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GEMMA_MODEL    = os.getenv("GEMMA_MODEL", "google/gemma-4-31b-it")
INVOKE_URL     = os.getenv(
    "NVIDIA_INVOKE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions"
)

LEGAL_SYSTEM_PROMPT = """You are BarPal.ai, a precise and cautious legal document assistant for startups and SMEs.

You ONLY answer based on the document excerpts provided to you in the context. Follow these rules strictly:

1. **Ground every answer** in the provided context. Quote relevant clauses verbatim when helpful.
2. **Cite your sources**: always mention the document name and page number at the end of your answer.
3. **Be explicit about uncertainty**: if the context does not contain enough information to answer, say "The provided documents do not contain sufficient information to answer this question."
4. **Never provide general legal advice** not grounded in the specific documents.
5. **Highlight risk terms**: flag clauses that could be high-risk for a startup (e.g., broad IP assignment, perpetual non-competes, unlimited liability).
6. **Structure your answer** with clear headings when answering multi-part questions.
7. Do NOT hallucinate or invent contract terms.
"""


def _build_payload(user_prompt: str) -> dict:
    """Build the OpenAI-compatible request payload for NVIDIA API."""
    return {
        "model": GEMMA_MODEL,
        "messages": [
            {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 16384,
        "temperature": 0.10,   # Low temp for factual legal answers
        "top_p": 0.95,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _parse_sse_stream(raw_lines) -> str:
    """
    Parse NVIDIA's SSE stream (OpenAI-compatible format) and
    return the fully assembled answer text.

    Each line looks like:
        data: {"id":...,"choices":[{"delta":{"content":"..."},...}],...}
    Terminal line:
        data: [DONE]

    Gemma 4 thinking mode may emit a <think>...</think> block first.
    We strip the thinking block and return only the final answer.
    """
    full_text = []

    for raw in raw_lines:
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        payload_str = line[len("data:"):].strip()
        if payload_str == "[DONE]":
            break
        try:
            chunk = json.loads(payload_str)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            if content:
                full_text.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    combined = "".join(full_text)

    # Strip the <think>...</think> reasoning block emitted by thinking mode.
    # Keep only the final answer that follows the closing </think>.
    think_match = re.search(r"<think>.*?</think>\s*", combined, re.DOTALL)
    if think_match:
        combined = combined[think_match.end():]

    return combined.strip()


async def query_gemma(user_question: str, context_chunks: list[dict]) -> str:
    """
    Retrieve answer from Gemma 4 31B via NVIDIA Inference API.

    Args:
        user_question : The plain-English legal question.
        context_chunks: List of {"text": str, "source": str, "page": int}
                        retrieved by the RAG pipeline.

    Returns:
        Formatted answer string (markdown).
    """
    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is not set. "
            "Please add it to backend/.env  →  NVIDIA_API_KEY=nvapi-..."
        )

    # ── Build context block ────────────────────────────────────────────────
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}, Page {c.get('page', '?')}]\n{c['text']}"
        for c in context_chunks
    )

    user_prompt = (
        "Based ONLY on the following legal document excerpts, answer the question.\n\n"
        f"=== DOCUMENT EXCERPTS ===\n{context_text}\n\n"
        f"=== QUESTION ===\n{user_question}"
    )

    payload = _build_payload(user_prompt)

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    # ── Stream the response ────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST", INVOKE_URL, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            raw_lines = [line async for line in response.aiter_lines()]

    answer = _parse_sse_stream(raw_lines)

    if not answer:
        raise RuntimeError(
            "Received an empty response from the model. "
            "The context may have been too long or the request timed out."
        )

    return answer
