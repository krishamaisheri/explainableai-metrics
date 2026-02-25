"""
OpenRouter LLM client.
Wraps the OpenAI-compatible SDK to call models via OpenRouter.
Includes LLMTracker for per-metric call analytics with full trace capture.
"""

import json
import logging
import re
import time
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

_client = None


# ── LLM Call Tracker ────────────────────────────────────────────
class LLMTracker:
    """Singleton tracker that records every LLM call with model, time, caller, and I/O."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._calls = []
        return cls._instance

    def record(
        self,
        model: str,
        duration: float,
        caller: str,
        success: bool,
        prompt: str = "",
        response: str = "",
    ):
        self._calls.append({
            "model": model,
            "duration": round(duration, 3),
            "caller": caller,
            "success": success,
            "prompt": prompt[:500],      # truncate prompts to save memory
            "response": response[:1000],  # truncate responses
        })

    def reset(self):
        self._calls = []

    def get_calls_for(self, caller: str) -> list[dict]:
        """Return all recorded calls for a given caller/metric."""
        return [c for c in self._calls if c["caller"] == caller]

    def get_stats(self) -> dict:
        """Return per-caller and global stats."""
        stats = {}
        for call in self._calls:
            caller = call["caller"]
            if caller not in stats:
                stats[caller] = {
                    "total_calls": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_time": 0.0,
                    "models_used": set(),
                }
            stats[caller]["total_calls"] += 1
            if call["success"]:
                stats[caller]["successful"] += 1
            else:
                stats[caller]["failed"] += 1
            stats[caller]["total_time"] += call["duration"]
            stats[caller]["models_used"].add(call["model"])

        # Convert sets to sorted lists for JSON serialization
        for caller in stats:
            stats[caller]["models_used"] = sorted(stats[caller]["models_used"])
            stats[caller]["total_time"] = round(stats[caller]["total_time"], 3)

        return stats

    def get_global_summary(self) -> dict:
        total = len(self._calls)
        success = sum(1 for c in self._calls if c["success"])
        total_time = sum(c["duration"] for c in self._calls)
        return {
            "total_calls": total,
            "successful": success,
            "failed": total - success,
            "total_time": round(total_time, 3),
        }


tracker = LLMTracker()


# ── Metric Trace Collector ──────────────────────────────────────
class MetricTraceCollector:
    """
    Collects computation trace details from each metric.
    Metrics write their intermediate values and formula steps here.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._traces = {}
        return cls._instance

    def reset(self):
        self._traces = {}

    def set_trace(self, metric_name: str, trace: dict):
        """Store the computation trace for a metric."""
        self._traces[metric_name] = trace

    def get_trace(self, metric_name: str) -> dict:
        return self._traces.get(metric_name, {})

    def get_all_traces(self) -> dict:
        return dict(self._traces)


trace_collector = MetricTraceCollector()


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


def call_llm(
    prompt: str,
    *,
    model: str | None = None,
    max_retries: int = 3,
    caller: str = "unknown",
) -> str:
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
    caller : str
        Name of the metric/component making this call (for tracking).
    """
    model = model or config.REASONING_MODEL
    client = _get_client()

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            duration = time.time() - t0
            tracker.record(model, duration, caller, success=True,
                           prompt=prompt, response=content)
            return content
        except Exception as exc:
            duration = time.time() - t0
            tracker.record(model, duration, caller, success=False,
                           prompt=prompt, response=str(exc))
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first complete JSON object from text using bracket-depth matching.
    Also handles markdown code fences (```json ... ```).
    """
    # Strip markdown code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.replace('```', '')

    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError("Incomplete JSON object in response")


def call_llm_json(
    prompt: str,
    *,
    model: str | None = None,
    caller: str = "unknown",
) -> dict:
    """
    Call the LLM and parse the response as JSON.

    Uses bracket-depth matching to extract the first complete JSON object,
    handling markdown fences and conversational filler text.
    Sanitizes control characters that LLMs often include inside JSON strings.
    """
    raw = call_llm(prompt, model=model, caller=caller)

    try:
        json_str = _extract_first_json_object(raw)
        # Sanitize control characters inside JSON string values
        json_str = _sanitize_json_string(json_str)
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse LLM JSON response: %s\nOriginal raw response:\n%s", exc, raw)
        raise


def _sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string by escaping control characters inside string values.
    LLMs often produce literal newlines/tabs inside JSON strings which are invalid.
    """
    result = []
    in_string = False
    escape_next = False

    for ch in json_str:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue

        if ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string and ch in ('\n', '\r', '\t'):
            # Replace control chars with escaped versions
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            continue

        result.append(ch)

    return ''.join(result)
