"""
OpenRouter LLM client.
Wraps the OpenAI-compatible SDK to call models via OpenRouter.
"""

import json
import logging
import time
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    """Lazy-initialise the OpenAI client pointing at OpenRouter."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://explainability-metrics.gov.uk",
                "X-Title": "Explainability Metrics Governance",
            },
        )
    return _client


def call_llm(prompt: str, *, model: str | None = None, max_retries: int = 3) -> str:
    """
    Send a prompt to OpenRouter and return the raw text response.

    Parameters
    ----------
    prompt : str
        The user/system prompt.
    model : str, optional
        OpenRouter model identifier.  Defaults to REASONING_MODEL.
    max_retries : int
        Number of retry attempts on transient failures.
    """
    model = model or config.REASONING_MODEL
    client = _get_client()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


def call_llm_json(prompt: str, *, model: str | None = None) -> dict:
    """
    Call the LLM and parse the response as JSON.

    The prompt **must** instruct the model to reply with valid JSON.
    A lenient parser strips markdown fences before decoding.
    """
    raw = call_llm(prompt, model=model)

    # Strip common markdown fences the model may wrap around JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON response:\n%s", raw)
        raise
