"""
TruthLens AI — Service Unit Tests

Tests for individual services:
    - Classifier (scoring-based)
    - Bias detector
    - Trust scorer
    - Confidence calibrator
    - Claim extractor
    - Cache
"""

from __future__ import annotations

import pytest

from app.models import ClassificationLabel, SubClaimStatus
from app.services.classifier import ClaimScores


# ── Classifier Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classifier_fake_claim():
    """Known misinformation should be classified as False."""
    from app.services.classifier import classify_sub_claim

    status, scores = await classify_sub_claim(
        "5G towers cause COVID-19"
    )
    assert status == SubClaimStatus.FALSE
    assert scores.fake_score > scores.real_score


@pytest.mark.asyncio
async def test_classifier_credible_claim():
    """Credible claim with source attribution should be True."""
    from app.services.classifier import classify_sub_claim

    status, scores = await classify_sub_claim(
        "According to a peer-reviewed study published in Nature, the vaccine is effective"
    )
    assert status == SubClaimStatus.TRUE
    assert scores.real_score > scores.fake_score


@pytest.mark.asyncio
async def test_classifier_neutral_claim():
    """Ambiguous claim should be Misleading (no clear evidence)."""
    from app.services.classifier import classify_sub_claim

    status, _ = await classify_sub_claim("Something happened somewhere")
    assert status in [SubClaimStatus.FALSE, SubClaimStatus.MISLEADING]


@pytest.mark.asyncio
async def test_aggregate_label_all_supported():
    """All true → Real."""
    from app.services.classifier import aggregate_label

    result = await aggregate_label([SubClaimStatus.TRUE, SubClaimStatus.TRUE])
    assert result == ClassificationLabel.REAL


@pytest.mark.asyncio
async def test_aggregate_label_any_false():
    """Mix of true and false → Misleading."""
    from app.services.classifier import aggregate_label

    result = await aggregate_label([SubClaimStatus.TRUE, SubClaimStatus.FALSE])
    assert result == ClassificationLabel.MISLEADING


@pytest.mark.asyncio
async def test_aggregate_label_all_false():
    """All false → Fake."""
    from app.services.classifier import aggregate_label

    result = await aggregate_label([SubClaimStatus.FALSE, SubClaimStatus.FALSE])
    assert result == ClassificationLabel.FAKE


# ── Bias Detector Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bias_detector_sensational():
    """Text with sensational words should have high bias score."""
    from app.services.bias_detector import detect_bias

    report = await detect_bias(
        "SHOCKING!! You won't believe what they don't want you to know!!!"
    )
    assert report.bias_detected is True
    assert report.bias_score > 0.1
    assert len(report.keywords_detected) >= 1


@pytest.mark.asyncio
async def test_bias_detector_neutral():
    """Neutral text should have low bias score."""
    from app.services.bias_detector import detect_bias

    report = await detect_bias(
        "The study found a correlation between exercise and improved health outcomes.",
        has_source_attribution=True,
    )
    assert report.bias_score < 0.2


@pytest.mark.asyncio
async def test_bias_detector_missing_sources():
    """Text without sources should flag lack_of_evidence."""
    from app.services.bias_detector import detect_bias

    report = await detect_bias("Something bad happened.", has_source_attribution=False)
    assert report.lack_of_evidence is True


# ── Trust Scorer Tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trust_score_high():
    """All-supported claims with low bias should score high."""
    from app.services.trust_scorer import compute_trust_score

    results = [
        ("claim 1", SubClaimStatus.TRUE, ClaimScores(real_score=5, fake_score=0)),
        ("claim 2", SubClaimStatus.TRUE, ClaimScores(real_score=4, fake_score=0)),
    ]
    score = await compute_trust_score(results, bias_score=0.05, lack_of_evidence=False)
    assert score >= 60


@pytest.mark.asyncio
async def test_trust_score_low():
    """All-false claims with high bias should score low."""
    from app.services.trust_scorer import compute_trust_score

    results = [
        ("claim 1", SubClaimStatus.FALSE, ClaimScores(real_score=0, fake_score=10)),
        ("claim 2", SubClaimStatus.FALSE, ClaimScores(real_score=0, fake_score=8)),
    ]
    score = await compute_trust_score(results, bias_score=0.8, lack_of_evidence=True)
    assert score <= 35


# ── Confidence Calibrator Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confidence_high_agreement():
    """Strong agreement should yield higher confidence."""
    from app.services.confidence_calibrator import calibrate_confidence

    results = [
        ("claim", SubClaimStatus.FALSE, ClaimScores(real_score=0, fake_score=10)),
        ("claim", SubClaimStatus.FALSE, ClaimScores(real_score=0, fake_score=8)),
    ]
    conf = await calibrate_confidence(results, bias_score=0.2, lack_of_evidence=False, trust_score=20)
    assert conf >= 60


@pytest.mark.asyncio
async def test_confidence_reduced_by_bias():
    """High bias should reduce confidence."""
    from app.services.confidence_calibrator import calibrate_confidence

    results = [
        ("claim", SubClaimStatus.MISLEADING, ClaimScores(real_score=2, fake_score=2)),
    ]
    conf_low_bias = await calibrate_confidence(results, bias_score=0.1, lack_of_evidence=False, trust_score=50)
    conf_high_bias = await calibrate_confidence(results, bias_score=0.9, lack_of_evidence=True, trust_score=50)
    assert conf_low_bias > conf_high_bias


# ── Claim Extractor Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_claim_strips_filler():
    """Filler prefixes should be stripped."""
    from app.services.claim_extractor import extract_claim

    result = await extract_claim("BREAKING: Scientists discover new planet")
    assert "BREAKING" not in result
    assert "planet" in result.lower()


@pytest.mark.asyncio
async def test_split_sub_claims_compound():
    """Compound claims should be split."""
    from app.services.claim_extractor import split_sub_claims

    result = await split_sub_claims(
        "The vaccine is safe and also it is effective for all ages"
    )
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_split_sub_claims_single():
    """Single claim should return as-is."""
    from app.services.claim_extractor import split_sub_claims

    result = await split_sub_claims("The sky is blue")
    assert len(result) == 1


# ── Cache Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_set_get():
    """Cache should store and retrieve values."""
    from app.utils.cache import InMemoryCache

    cache = InMemoryCache(ttl_seconds=60, max_size=10)
    await cache.set("hello", {"result": "world"})
    result = await cache.get("hello")
    assert result == {"result": "world"}


@pytest.mark.asyncio
async def test_cache_miss():
    """Cache miss should return None."""
    from app.utils.cache import InMemoryCache

    cache = InMemoryCache(ttl_seconds=60, max_size=10)
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_cache_eviction():
    """Cache should evict oldest when max_size is reached."""
    from app.utils.cache import InMemoryCache

    cache = InMemoryCache(ttl_seconds=60, max_size=2)
    await cache.set("key1", "val1")
    await cache.set("key2", "val2")
    await cache.set("key3", "val3")  # Should evict key1

    assert await cache.get("key1") is None
    assert await cache.get("key2") == "val2"
    assert await cache.get("key3") == "val3"


@pytest.mark.asyncio
async def test_cache_stats():
    """Cache stats should track hits and misses."""
    from app.utils.cache import InMemoryCache

    cache = InMemoryCache(ttl_seconds=60, max_size=10)
    await cache.set("x", 42)
    await cache.get("x")  # hit
    await cache.get("y")  # miss

    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


# ── External Evidence Processor Tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_evidence_processor_confirming():
    """External evidence that confirms a claim should yield high confirmation score."""
    from app.services.evidence_processor import process_external_evidence

    report = await process_external_evidence(
        external_evidence="This claim has been confirmed and verified by multiple studies.",
        ai_analysis="The claim is accurate and supported by evidence.",
    )
    assert report.has_external_data is True
    assert report.confirmation_score > report.denial_score
    assert report.net_score > 0


@pytest.mark.asyncio
async def test_evidence_processor_denying():
    """External evidence that denies a claim should yield high denial score."""
    from app.services.evidence_processor import process_external_evidence

    report = await process_external_evidence(
        external_evidence="This claim has been debunked and rated false by fact-checkers.",
        ai_analysis="The claim is baseless misinformation with no evidence.",
    )
    assert report.has_external_data is True
    assert report.denial_score > report.confirmation_score
    assert report.net_score < 0


@pytest.mark.asyncio
async def test_evidence_processor_mixed():
    """Mixed evidence should yield mixed signals."""
    from app.services.evidence_processor import process_external_evidence

    report = await process_external_evidence(
        external_evidence="The claim is partially true but taken out of context and exaggerated.",
    )
    assert report.has_external_data is True
    assert report.mixed_score > 0


@pytest.mark.asyncio
async def test_evidence_processor_no_data():
    """No external data should return default empty report."""
    from app.services.evidence_processor import process_external_evidence

    report = await process_external_evidence()
    assert report.has_external_data is False
    assert report.confirmation_score == 0.0
    assert report.denial_score == 0.0
    assert report.evidence_strength == "none"


@pytest.mark.asyncio
async def test_evidence_processor_news_results():
    """News results should be extracted as fact sources."""
    from app.models import NewsResult
    from app.services.evidence_processor import process_external_evidence

    news = [
        NewsResult(
            title="Reuters confirms vaccine safety",
            source="reuters.com",
            snippet="Multiple studies have confirmed that the vaccine is safe and effective.",
            url="https://reuters.com/article/vaccine-safety",
        ),
        NewsResult(
            title="BBC report on vaccine trials",
            source="bbc.com",
            snippet="The vaccine has been verified through clinical trials.",
            url="https://bbc.com/news/vaccine-trials",
        ),
    ]
    report = await process_external_evidence(news_results=news)
    assert report.has_external_data is True
    assert report.source_count >= 2
    assert len(report.extracted_sources) >= 2
    assert report.confirmation_score > 0
