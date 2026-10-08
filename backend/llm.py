"""LLM calls: spelling correction and grounded answering with related questions (Resilient AI Studio Version)."""
import json
import os
import time

from . import config as C

_gemini = None
_openai = None


def _chat(system: str, user: str, fast: bool = False, max_tokens: int = 1200) -> str:
    global _gemini, _openai

    # OpenAI
    if C.LLM_PROVIDER == "openai":
        if _openai is None:
            from openai import OpenAI
            _openai = OpenAI()

        r = _openai.chat.completions.create(
            model=C.OPENAI_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        return r.choices[0].message.content or ""

    # Gemini via AI Studio
    if C.LLM_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or getattr(C, "GEMINI_API_KEY", None)

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set in environment or config.py."
            )

        # Re-initialize client dynamically if key changes
        if _gemini is None or getattr(_gemini, "_active_key", None) != api_key:
            from google import genai
            _gemini = genai.Client(api_key=api_key)
            _gemini._active_key = api_key

        # Model hierarchy: primary model, then secondary fallbacks for capacity spikes
        fast_model = getattr(C, "GEMINI_FAST_MODEL", "gemini-2.5-flash")
        std_model = getattr(C, "GEMINI_MODEL", "gemini-2.5-flash")
        primary_model = fast_model if fast else std_model

        candidate_models = [primary_model, "gemini-1.5-flash", "gemini-1.5-pro"]

        # Deduplicate while preserving order
        candidate_models = list(dict.fromkeys(candidate_models))

        last_exception = None

        for model in candidate_models:
            # Retry up to 3 times per model for temporary 503 glitches
            for attempt in range(3):
                try:
                    response = _gemini.models.generate_content(
                        model=model,
                        contents=user,
                        config={
                            "system_instruction": system,
                            "max_output_tokens": max_tokens,
                            "response_mime_type": "application/json",
                        },
                    )
                    return response.text or ""
                except Exception as e:
                    last_exception = e
                    err_msg = str(e)
                    # Check if error is high demand / 503 / 429
                    if "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "high demand" in err_msg.lower():
                        time.sleep(1 * (attempt + 1))  # Exponential backoff delay
                        continue
                    else:
                        raise e  # Fast-fail non-capacity errors

        # If all candidates/attempts failed, raise the last exception
        if last_exception:
            raise last_exception

    raise ValueError(f"Unsupported LLM provider: {C.LLM_PROVIDER}")


def _json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


def correct_query(question: str) -> tuple[str, bool]:
    system = (
        "You fix spelling mistakes in questions that will be searched in a document. "
        "Fix only clear typos. Never change proper nouns, product names, IDs, codes or numbers. "
        'Reply with JSON only: {"corrected": "<question>"}'
    )

    try:
        fixed = _json(
            _chat(
                system,
                question,
                fast=True,
                max_tokens=200,
            )
        ).get("corrected", "").strip()

    except Exception:
        return question, False

    if not fixed or fixed.lower() == question.strip().lower():
        return question, False

    return fixed, True


def answer(question: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(
        f"[Page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    system = (
        "You are a PDF research assistant. "
        "Answer using ONLY the document context provided. "
        "Do not invent information. "
        "If the answer is not in the context, say you could not find it "
        "in the document. "
        "Cite pages inline like [p.42]. "
        "Keep the answer clear and concise. "
        "Also suggest 3 short follow-up questions that the document context can answer. "
        'Reply with JSON only: '
        '{"answer": "...", "related_questions": ["...", "...", "..."]}'
    )

    raw = _chat(
        system,
        f"CONTEXT:\n{context}\n\nQUESTION: {question}",
    )

    data = _json(raw)

    if "answer" not in data:
        return {
            "answer": raw.strip(),
            "related_questions": [],
        }

    return {
        "answer": data["answer"],
        "related_questions": data.get("related_questions", [])[:3],
    }