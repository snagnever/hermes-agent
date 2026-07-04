"""/v1/models + /health contract for the claude-max server."""

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from hermes_cli.claude_max.server import create_app  # noqa: E402


async def _client() -> TestClient:
    client = TestClient(TestServer(create_app()))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_health_ok():
    client = await _client()
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert body.get("status") == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_models_list_shape():
    client = await _client()
    try:
        resp = await client.get("/v1/models?limit=1000")
        assert resp.status == 200
        body = await resp.json()
        assert body["has_more"] is False
        data = body["data"]
        assert data and all(m["type"] == "model" for m in data)
        assert all(m["max_input_tokens"] == 200000 for m in data)
        assert any("opus" in m["id"] for m in data)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_models_limit_applied():
    client = await _client()
    try:
        resp = await client.get("/v1/models?limit=2")
        body = await resp.json()
        assert len(body["data"]) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unknown_route_anthropic_error():
    client = await _client()
    try:
        resp = await client.get("/v1/nonsense")
        assert resp.status == 404
        body = await resp.json()
        assert body["type"] == "error"
        assert "type" in body["error"] and "message" in body["error"]
    finally:
        await client.close()
