"""Static Anthropic-shaped model catalog for the claude-max proxy.

There is no REST /models catalog on the subscription path, so the proxy serves
a fixed list. hermes reads ``max_input_tokens`` from it for context-length
detection (agent/model_metadata.py) and lists ids in the /model picker. The
provider profile's ``fetch_models`` returns the same ids (no HTTP).

Leaf module: no intra-package imports, so the provider plugin can import it
cheaply at discovery time.
"""

from __future__ import annotations

# Claude context window (current Claude models). 200k is correct for
# opus/sonnet/haiku; the SDK/subscription enforces the real limit at turn time.
CONTEXT_WINDOW = 200_000

# (id, display_name). Full ids + SDK aliases; aliases resolve to the
# subscription's current model server-side.
_MODEL_ROWS: tuple[tuple[str, str], ...] = (
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-sonnet-5", "Claude Sonnet 5"),
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
    ("opus", "Claude Opus (latest)"),
    ("sonnet", "Claude Sonnet (latest)"),
    ("haiku", "Claude Haiku (latest)"),
)

MODEL_IDS: tuple[str, ...] = tuple(mid for mid, _ in _MODEL_ROWS)

# Stable synthetic created_at (Date.now unavailable / avoid churn in caches).
_CREATED_AT = "2026-01-01T00:00:00Z"


def _model_entry(model_id: str, display_name: str) -> dict:
    return {
        "type": "model",
        "id": model_id,
        "display_name": display_name,
        "created_at": _CREATED_AT,
        "max_input_tokens": CONTEXT_WINDOW,
    }


def models_response(limit: int | None = None) -> dict:
    """Anthropic /v1/models list response shape."""
    rows = [_model_entry(mid, name) for mid, name in _MODEL_ROWS]
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return {
        "data": rows,
        "has_more": False,
        "first_id": rows[0]["id"] if rows else None,
        "last_id": rows[-1]["id"] if rows else None,
    }
