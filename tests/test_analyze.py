"""
TruthLens AI — Endpoint Integration Tests

Tests the POST /analyze endpoint for:
    - Valid input → correct schema
    - Empty/bad input → 400
    - Rate limiting → 429
    - Response schema compliance
    - Caching behavior
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest_asyncio.fixture
async def client():
    """Create a test client with a fresh app instance, manually initializing state."""
    app = create_app()

    # Manually initialize app state (lifespan doesn't run with ASGITransport)
    from app.middleware.rate_limiter.in_memory import InMemoryRateLimiter
    from app.utils.cache import InMemoryCache

    app.state.rate_limiter = InMemoryRateLimiter(max_requests=100, window_seconds=60)
    app.state.cache = InMemoryCache(ttl_seconds=300, max_size=1000)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Health check should return 200 with status=healthy."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "cache" in data
    assert "rate_limiter" in data


@pytest.mark.asyncio
async def test_analyze_valid_input(client: AsyncClient):
    """Valid input should return 200 with complete schema."""
    resp = await client.post(
        "/analyze",
        json={"text": "COVID vaccines cause infertility according to some social media posts"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Verify all required fields exist
    required_fields = [
        "label", "confidence", "language", "claim", "sub_claims",
        "explanation", "reasoning", "fact_sources", "evidence",
        "trust_score", "bias_detected", "xai", "final_verdict",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    # Verify label is valid
    assert data["label"] in ["Real", "Fake", "Misleading"]

    # Verify score ranges
    assert 0 <= data["confidence"] <= 100
    assert 0 <= data["trust_score"] <= 100

    # Verify sub_claims is non-empty
    assert len(data["sub_claims"]) >= 1

    # Verify evidence structure
    assert "image_explanation" in data["evidence"]
    assert len(data["evidence"]["video_search_queries"]) == 3

    # Verify XAI structure
    assert "keywords_detected" in data["xai"]
    assert "lack_of_evidence" in data["xai"]
    assert 0 <= data["xai"]["bias_score"] <= 1

    # Verify fact sources
    assert len(data["fact_sources"]) >= 1
    for source in data["fact_sources"]:
        assert "title" in source
        assert "source" in source
        assert source["trust"] in ["high", "medium", "low"]


@pytest.mark.asyncio
async def test_analyze_empty_input(client: AsyncClient):
    """Empty text should return 422 (validation error)."""
    resp = await client.post("/analyze", json={"text": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_whitespace_input(client: AsyncClient):
    """Whitespace-only text should return 422."""
    resp = await client.post("/analyze", json={"text": "   "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_missing_text_field(client: AsyncClient):
    """Missing text field should return 422."""
    resp = await client.post("/analyze", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_non_json_body(client: AsyncClient):
    """Non-JSON body should return 422."""
    resp = await client.post(
        "/analyze",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_known_fake_claim(client: AsyncClient):
    """Known fake claim should be classified as Fake."""
    resp = await client.post(
        "/analyze",
        json={"text": "5G towers cause COVID-19 and the government is hiding this from us"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "Fake"
    assert data["trust_score"] < 50
    assert data["bias_detected"] is True


@pytest.mark.asyncio
async def test_analyze_credible_claim(client: AsyncClient):
    """Credible claim with source attribution should score higher trust."""
    resp = await client.post(
        "/analyze",
        json={
            "text": (
                "According to a peer-reviewed study published in the Journal of Medicine, "
                "regular exercise may reduce the risk of heart disease."
            )
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ["Real", "Misleading"]
    assert data["trust_score"] >= 40


@pytest.mark.asyncio
async def test_analyze_caching(client: AsyncClient):
    """Second request with same text should be served from cache."""
    text = "The earth is round according to NASA"

    # First request
    resp1 = await client.post("/analyze", json={"text": text})
    assert resp1.status_code == 200
    assert resp1.headers.get("x-truthlens-cached") == "false"

    # Second request (should be cached)
    resp2 = await client.post("/analyze", json={"text": text})
    assert resp2.status_code == 200
    assert resp2.headers.get("x-truthlens-cached") == "true"

    # Results should be identical
    assert resp1.json() == resp2.json()


@pytest.mark.asyncio
async def test_rate_limiting(client: AsyncClient):
    """Exceeding rate limit should return 429."""
    # Create a fresh app with very low rate limit
    from app.main import create_app as _create

    app = _create()
    # Override rate limiter with low limit and initialize cache
    from app.middleware.rate_limiter.in_memory import InMemoryRateLimiter
    from app.utils.cache import InMemoryCache

    app.state.rate_limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    app.state.cache = InMemoryCache(ttl_seconds=300, max_size=1000)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as limited_client:
        # Use unique texts to avoid cache hits
        for i in range(2):
            resp = await limited_client.post(
                "/analyze", json={"text": f"Test claim number {i} for rate limiting"}
            )
            assert resp.status_code == 200

        # Third request should be rate limited
        resp = await limited_client.post(
            "/analyze", json={"text": "This should be rate limited"}
        )
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers


@pytest.mark.asyncio
async def test_response_explanation_length(client: AsyncClient):
    """Explanation must be at least 50 characters (3-5 lines)."""
    resp = await client.post(
        "/analyze",
        json={"text": "Scientists discovered a new species of deep sea fish"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["explanation"]) >= 50


@pytest.mark.asyncio
async def test_response_reasoning_count(client: AsyncClient):
    """Reasoning must have at least 3 points."""
    resp = await client.post(
        "/analyze",
        json={"text": "The stock market crashed today due to global uncertainty"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["reasoning"]) >= 3
