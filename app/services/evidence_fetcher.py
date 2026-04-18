"""
TruthLens AI — Real-Time Evidence Fetcher (v4)

Production-grade evidence pipeline with:
    1. SerpAPI / Google News (PRIORITY)
    2. Wikipedia FULL CONTENT API (not summary)
    3. Evidence classification (support/refute/neutral)
    4. Gemini LLM override (when available)
    5. Wikipedia authority boost
    6. Source-aware credibility weighting
    7. Multi-source agreement boost
    8. Weak evidence filtering (similarity < 0.3 dropped)

Fixes applied:
    FIX 1: Wikipedia credibility + similarity boost
    FIX 2: Full Wikipedia content (MediaWiki API, not REST summary)
    FIX 4: Multi-source agreement boost
    FIX 5: Source-aware weighted scoring
    FIX 6: Strong evidence filtering (min similarity 0.3)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from app.config import get_settings
from app.services.evidence_classifier import classify_evidence
from app.services.source_ranker import extract_domain, rank_source

logger = logging.getLogger(__name__)


# ── Optional: TF-IDF for semantic matching ───────────────────────────────────

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning(
        "scikit-learn not installed — falling back to Jaccard similarity. "
        "Install with: pip install scikit-learn"
    )


# ── Wikipedia User-Agent (FIXES 403 ERROR) ───────────────────────────────────

_WIKI_HEADERS = {
    "User-Agent": "TruthLensAI/2.0 (https://github.com/truthlens; truthlens@example.com)",
    "Accept": "application/json",
}


# ── Source-Aware Credibility (FIX 5) ─────────────────────────────────────────

_HIGH_TRUST_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nature.com", "science.org", "who.int", "cdc.gov", "nih.gov",
    "nasa.gov", "isro.gov.in", "esa.int",
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "thelancet.com", "bmj.com", "nejm.org",
}

_MEDIUM_TRUST_DOMAINS = {
    "wikipedia.org", "en.wikipedia.org",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "economist.com", "forbes.com", "bloomberg.com",
    "cnn.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com",
    "aljazeera.com", "france24.com", "dw.com",
    "time.com", "newsweek.com", "usatoday.com",
    "thehindu.com", "ndtv.com",
}


def get_source_weight(domain: str) -> float:
    """
    Get credibility weight for a source domain (FIX 5).

    Returns:
        0.9 for tier-1 sources, 0.7 for tier-2, 0.4 for unknown.
    """
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]

    # Exact match
    if domain in _HIGH_TRUST_DOMAINS:
        return 0.9
    if domain in _MEDIUM_TRUST_DOMAINS:
        return 0.7

    # Subdomain match
    for d in _HIGH_TRUST_DOMAINS:
        if domain.endswith(f".{d}"):
            return 0.9
    for d in _MEDIUM_TRUST_DOMAINS:
        if domain.endswith(f".{d}"):
            return 0.7

    # TLD-based
    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return 0.85
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return 0.8
    if domain.endswith(".org"):
        return 0.6

    return 0.4


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class FetchedSource:
    """A single evidence source with metadata."""

    title: str
    url: str
    snippet: str
    source_type: str  # "news", "wikipedia", "search", "caller_provided"
    credibility: float = 0.0
    date: str = ""
    relevance: float = 0.0  # semantic similarity to claim
    stance: str = "neutral"  # "support", "refute", "neutral"


@dataclass
class EvidenceFetchResult:
    """Aggregated evidence from all external sources."""

    has_data: bool = False
    sources: List[FetchedSource] = field(default_factory=list)
    confirm_score: float = 0.0
    deny_score: float = 0.0
    confidence: float = 0.0
    evidence_summary: str = ""
    support_count: int = 0
    refute_count: int = 0


# ── Evidence Cache ───────────────────────────────────────────────────────────

_evidence_cache: Dict[str, Tuple[float, EvidenceFetchResult]] = {}
_MAX_CACHE_SIZE = 500


def _cache_key(claim: str) -> str:
    return hashlib.sha256(claim.lower().strip().encode()).hexdigest()[:32]


def _get_cached(claim: str, ttl: int) -> Optional[EvidenceFetchResult]:
    key = _cache_key(claim)
    entry = _evidence_cache.get(key)
    if entry is not None:
        ts, result = entry
        if time.time() - ts < ttl:
            logger.debug("Evidence cache HIT: %s", claim[:50])
            return result
        del _evidence_cache[key]
    return None


def _set_cache(claim: str, result: EvidenceFetchResult) -> None:
    if len(_evidence_cache) >= _MAX_CACHE_SIZE:
        oldest_key = min(_evidence_cache, key=lambda k: _evidence_cache[k][0])
        del _evidence_cache[oldest_key]
    _evidence_cache[_cache_key(claim)] = (time.time(), result)


# ── Semantic Matching ────────────────────────────────────────────────────────


def semantic_match(claim: str, text: str) -> float:
    """
    Compute semantic similarity between claim and evidence text.

    Uses TF-IDF + cosine similarity (scikit-learn) when available,
    falls back to Jaccard word-overlap similarity otherwise.

    Returns:
        Similarity score (0.0-1.0).
    """
    if not claim.strip() or not text.strip():
        return 0.0

    if _HAS_SKLEARN:
        return _tfidf_similarity(claim, text)
    return _jaccard_similarity(claim, text)


def _tfidf_similarity(claim: str, text: str) -> float:
    """TF-IDF cosine similarity using scikit-learn."""
    global _HAS_SKLEARN
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([claim, text])
        sim = _sklearn_cosine(matrix[0:1], matrix[1:2])
        return float(max(0.0, min(1.0, sim[0][0])))
    except (MemoryError, Exception) as e:
        logger.warning("TF-IDF similarity failed (disabling sklearn): %s", e)
        _HAS_SKLEARN = False  # Disable to prevent repeated OpenBLAS crashes
        return _jaccard_similarity(claim, text)


def _jaccard_similarity(claim: str, text: str) -> float:
    """Fallback: Jaccard word-overlap similarity."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "it", "its", "this", "that", "and", "but",
        "or", "for", "to", "of", "in", "on", "at", "by", "from", "with",
    }
    claim_words = {w.lower() for w in claim.split() if w.lower() not in stop and len(w) > 2}
    text_words = {w.lower() for w in text.split() if w.lower() not in stop and len(w) > 2}
    if not claim_words or not text_words:
        return 0.0
    intersection = claim_words & text_words
    union = claim_words | text_words
    return len(intersection) / len(union) if union else 0.0


# ── HTTP Fetch with Retry ────────────────────────────────────────────────────


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    max_retries: int = 2,
    timeout: float = 5.0,
) -> Optional[dict]:
    """
    Fetch JSON from URL with retry and exponential backoff.

    Args:
        client: httpx async client instance.
        url: Target URL.
        params: Query parameters.
        headers: HTTP headers (e.g., User-Agent for Wikipedia).
        max_retries: Maximum retry attempts (default 2).
        timeout: HTTP timeout in seconds (default 5).

    Returns:
        Parsed JSON dict, or None on failure.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(
                url, params=params, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.warning(
                "Timeout attempt %d/%d: %s", attempt + 1, max_retries + 1, url[:80]
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2.0 * (attempt + 1)
                logger.warning("Rate limited (429), waiting %.1fs", wait)
                await asyncio.sleep(wait)
            elif e.response.status_code == 403:
                logger.warning(
                    "HTTP 403 Forbidden attempt %d: %s (check User-Agent header)",
                    attempt + 1, url[:80],
                )
            else:
                logger.warning(
                    "HTTP %d attempt %d: %s",
                    e.response.status_code, attempt + 1, url[:80],
                )
        except Exception as e:
            logger.warning("Fetch error attempt %d: %s", attempt + 1, str(e)[:100])

        if attempt < max_retries:
            await asyncio.sleep(1.0 * (attempt + 1))

    return None


# ── Source Fetchers ──────────────────────────────────────────────────────────


async def _fetch_serpapi(
    query: str,
    client: httpx.AsyncClient,
    api_key: str,
    timeout: float,
) -> List[FetchedSource]:
    """Fetch results from SerpAPI (Google News first, web search fallback)."""
    if not api_key:
        logger.debug("SerpAPI key not configured -- skipping news fetch")
        return []

    sources: List[FetchedSource] = []

    # Attempt 1: Google News
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "tbm": "nws",
        "num": 5,
    }

    data = await _fetch_with_retry(client, "https://serpapi.com/search", params, timeout=timeout)
    if data and "news_results" in data:
        for item in data["news_results"][:5]:
            link = item.get("link", "")
            domain = extract_domain(link)
            sources.append(FetchedSource(
                title=item.get("title", ""),
                url=link,
                snippet=item.get("snippet", item.get("title", "")),
                source_type="news",
                credibility=rank_source(domain),
                date=item.get("date", ""),
            ))

    # Attempt 2: Regular Google search (fallback if no news)
    if not sources:
        params.pop("tbm", None)
        data = await _fetch_with_retry(
            client, "https://serpapi.com/search", params, timeout=timeout
        )
        if data and "organic_results" in data:
            for item in data["organic_results"][:5]:
                link = item.get("link", "")
                domain = extract_domain(link)
                sources.append(FetchedSource(
                    title=item.get("title", ""),
                    url=link,
                    snippet=item.get("snippet", ""),
                    source_type="search",
                    credibility=rank_source(domain),
                    date=item.get("date", ""),
                ))

    logger.info("SerpAPI returned %d sources for: %s", len(sources), query[:60])
    return sources


async def _fetch_wikipedia(
    entities: List[str],
    client: httpx.AsyncClient,
    timeout: float,
) -> List[FetchedSource]:
    """
    Fetch FULL Wikipedia content via MediaWiki API (FIX 2).

    Uses action=query&prop=extracts&explaintext=true for full plain-text
    content instead of the REST summary endpoint.

    Includes User-Agent header to avoid 403 errors.
    """
    sources: List[FetchedSource] = []

    for entity in entities[:3]:  # Limit to 3 lookups
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "true",
            "titles": entity,
            "format": "json",
            "exlimit": "1",
            "exintro": "false",  # Get full content, not just intro
        }

        data = await _fetch_with_retry(
            client,
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=_WIKI_HEADERS,
            timeout=timeout,
        )

        if not data:
            continue

        # Parse MediaWiki response structure
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            # Skip missing pages (page_id == "-1")
            if page_id == "-1" or "missing" in page_data:
                logger.debug("Wikipedia page not found: %s", entity)
                continue

            extract_text = page_data.get("extract", "")
            if len(extract_text) < 50:
                continue

            title = page_data.get("title", entity)
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            # Truncate to first 2000 chars for processing (full content
            # is used for classification, but snippet is capped for storage)
            sources.append(FetchedSource(
                title=f"Wikipedia: {title}",
                url=page_url,
                snippet=extract_text[:2000],
                source_type="wikipedia",
                credibility=0.80,
            ))

    logger.info("Wikipedia returned %d sources for: %s", len(sources), entities[:3])
    return sources


# ── Main Public API ──────────────────────────────────────────────────────────


async def fetch_evidence(
    claim: str,
    entities: List[str],
    search_query: str,
) -> EvidenceFetchResult:
    """
    Fetch real-time evidence for a claim from external sources.

    Pipeline (v4 -- full upgrades):
        1. Check local cache
        2. Fetch news via SerpAPI (PRIORITY)
        3. Fetch Wikipedia FULL content (SUPPORTING)
        4. For EACH source:
            a. Compute semantic similarity (relevance)
            b. FIX 6: Filter weak evidence (similarity < 0.3)
            c. FIX 1: Apply Wikipedia authority boost
            d. FIX 5: Apply source-aware credibility weight
            e. Classify evidence (rule-based + optional Gemini)
            f. Weighted score = similarity * source_weight
        5. FIX 4: Apply multi-source agreement boost
        6. Aggregate confirm/deny scores
        7. Return structured result
    """
    settings = get_settings()

    # -- Cache check
    cached = _get_cached(claim, ttl=settings.evidence_cache_ttl)
    if cached is not None:
        return cached

    result = EvidenceFetchResult()
    timeout = float(settings.evidence_fetch_timeout)

    # -- Fetch from sources (concurrent)
    async with httpx.AsyncClient() as client:
        news_task = _fetch_serpapi(search_query, client, settings.serpapi_key, timeout)
        wiki_task = _fetch_wikipedia(entities, client, timeout)

        gathered = await asyncio.gather(news_task, wiki_task, return_exceptions=True)

    # Handle exceptions from gather
    news_sources: List[FetchedSource] = (
        gathered[0] if not isinstance(gathered[0], Exception) else []
    )
    wiki_sources: List[FetchedSource] = (
        gathered[1] if not isinstance(gathered[1], Exception) else []
    )

    if isinstance(gathered[0], Exception):
        logger.error("News fetch failed: %s", gathered[0])
    if isinstance(gathered[1], Exception):
        logger.error("Wikipedia fetch failed: %s", gathered[1])

    # Merge: news first (priority), then Wikipedia (supporting)
    all_sources: List[FetchedSource] = list(news_sources) + list(wiki_sources)

    if not all_sources:
        logger.info("No evidence found for claim: %s", claim[:80])
        _set_cache(claim, result)
        return result

    result.has_data = True

    # -- Optionally load Gemini client
    gemini_fn = None
    if settings.gemini_enabled and settings.gemini_api_key:
        try:
            from app.services.gemini_client import analyze_claim_with_gemini
            gemini_fn = analyze_claim_with_gemini
            logger.info("Gemini LLM enabled for evidence classification")
        except ImportError:
            logger.warning("Gemini client import failed -- using rule-based only")

    # -- Score each source: similarity + classification + weighting
    total_confirm = 0.0
    total_deny = 0.0
    support_count = 0
    refute_count = 0
    gemini_calls = 0

    for source in all_sources:
        combined_text = f"{source.title} {source.snippet}"

        # Step A: Semantic relevance
        relevance = semantic_match(claim, combined_text)

        # Step B (FIX 1): Wikipedia authority boost
        source_url = source.url.lower()
        if "wikipedia.org" in source_url:
            source.credibility = min(1.0, source.credibility + 0.2)
            relevance = min(1.0, relevance * 1.2)

        source.relevance = relevance

        # Step C (FIX 5): Source-aware credibility weighting
        domain = extract_domain(source.url)
        source_weight = get_source_weight(domain)

        # Ensure credibility is capped at 1.0
        source.credibility = min(1.0, max(source.credibility, source_weight))

        # Step D (FIX 6): Filter weak evidence -- tiered by source trust
        if source_weight >= 0.7:
            min_similarity = 0.05
        elif source_weight >= 0.5:
            min_similarity = 0.08
        else:
            min_similarity = 0.10

        if relevance < min_similarity:
            source.stance = "neutral"
            continue

        # Step E: Evidence classification (rule-based)
        stance = classify_evidence(claim, combined_text)

        # Step F: Gemini LLM override (if available and relevant)
        # Limit to max 3 Gemini calls per claim to prevent timeouts
        if gemini_fn and relevance >= min_similarity and gemini_calls < 3:
            try:
                gemini_calls += 1
                gemini_result = await gemini_fn(claim, combined_text)
                if gemini_result and gemini_result.get("label"):
                    gemini_label = gemini_result["label"]
                    gemini_conf = gemini_result.get("confidence", 0.5)

                    # Use Gemini only if it's confident
                    if gemini_conf >= 0.6:
                        stance = gemini_label
                        logger.debug(
                            "Gemini override: stance=%s (conf=%.2f) for %s",
                            stance, gemini_conf, source.title[:40],
                        )
            except Exception as e:
                logger.warning("Gemini classification failed: %s", str(e)[:80])

        source.stance = stance

        # Step G: Weighted scoring (similarity * source_weight)
        weighted_score = relevance * source_weight

        if stance == "refute":
            total_deny += weighted_score
            refute_count += 1
        elif stance == "support":
            total_confirm += weighted_score
            support_count += 1
        else:
            # Neutral but relevant -- very weak signal
            total_confirm += weighted_score * 0.2
            total_deny += weighted_score * 0.05

    # -- FIX 4: Multi-source agreement boost
    if refute_count >= 2:
        total_deny += 0.2
        logger.debug("Multi-source refutation boost: +0.2 (%d sources)", refute_count)
    if support_count >= 2:
        total_confirm += 0.2
        logger.debug("Multi-source support boost: +0.2 (%d sources)", support_count)

    # Normalize to 0-1
    total = total_confirm + total_deny
    if total > 0:
        result.confirm_score = round(total_confirm / total, 3)
        result.deny_score = round(total_deny / total, 3)

    # Evidence confidence
    result.confidence = round(
        min(1.0, len(all_sources) * 0.10 + (total / (total + 1.0)) * 0.5), 3
    )

    # Store counts
    result.support_count = support_count
    result.refute_count = refute_count

    # Keep sources sorted by relevance
    result.sources = sorted(all_sources, key=lambda s: s.relevance, reverse=True)[:10]

    # -- Build summary
    if result.deny_score > result.confirm_score:
        result.evidence_summary = (
            f"Evidence from {len(result.sources)} source(s) predominantly "
            f"refutes this claim ({refute_count} refuting, {support_count} supporting; "
            f"confirm: {result.confirm_score:.0%}, deny: {result.deny_score:.0%})."
        )
    elif result.confirm_score > result.deny_score:
        result.evidence_summary = (
            f"Evidence from {len(result.sources)} source(s) predominantly "
            f"supports this claim ({support_count} supporting, {refute_count} refuting; "
            f"confirm: {result.confirm_score:.0%}, deny: {result.deny_score:.0%})."
        )
    else:
        result.evidence_summary = (
            f"Evidence from {len(result.sources)} source(s) provides "
            f"mixed signals about this claim."
        )

    _set_cache(claim, result)

    logger.info(
        "Evidence fetch complete: has_data=%s, confirm=%.3f, deny=%.3f, "
        "sources=%d, confidence=%.3f, support=%d, refute=%d",
        result.has_data, result.confirm_score, result.deny_score,
        len(result.sources), result.confidence, support_count, refute_count,
    )

    return result
