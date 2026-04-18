"""
TruthLens AI — Source Credibility Ranking System

Assigns credibility scores to sources based on domain reputation.

Tiers:
    TIER 1 (0.95): Wire services, top journals, government agencies, fact-checkers
    TIER 2 (0.80): Major newspapers, Wikipedia, broadcasters
    TIER 3 (0.60): Regional outlets, tech publications, magazines
    TIER 4 (0.30): Unknown / unrecognized domains (default)
"""

from __future__ import annotations

import logging
from typing import Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── Credibility Tiers ────────────────────────────────────────────────────────

TIER_1: Dict[str, float] = {
    # Wire services
    "reuters.com": 0.95, "apnews.com": 0.95,
    # Global broadcasters
    "bbc.com": 0.95, "bbc.co.uk": 0.95,
    # Top newspapers
    "nytimes.com": 0.95, "washingtonpost.com": 0.95,
    # Scientific journals
    "nature.com": 0.95, "science.org": 0.95,
    "thelancet.com": 0.95, "bmj.com": 0.95, "nejm.org": 0.95,
    # Government & health agencies
    "who.int": 0.95, "cdc.gov": 0.95, "nih.gov": 0.95,
    "nasa.gov": 0.95, "isro.gov.in": 0.95, "esa.int": 0.95,
    "gov.uk": 0.95,
    # Fact-checkers
    "snopes.com": 0.95, "politifact.com": 0.95,
    "factcheck.org": 0.95, "fullfact.org": 0.95,
}

TIER_2: Dict[str, float] = {
    # Major newspapers
    "theguardian.com": 0.80, "economist.com": 0.80,
    # Business / financial
    "forbes.com": 0.80, "bloomberg.com": 0.80,
    # Broadcasters
    "cnn.com": 0.80, "cbsnews.com": 0.80, "nbcnews.com": 0.80,
    "abcnews.go.com": 0.80, "aljazeera.com": 0.80,
    "france24.com": 0.80, "dw.com": 0.80,
    # Magazines
    "time.com": 0.80, "newsweek.com": 0.80,
    # Wikipedia
    "wikipedia.org": 0.80, "en.wikipedia.org": 0.80,
    # Indian outlets
    "thehindu.com": 0.80, "ndtv.com": 0.80,
    # US outlets
    "usatoday.com": 0.80, "latimes.com": 0.80,
}

TIER_3: Dict[str, float] = {
    # Tabloids / opinion-heavy
    "foxnews.com": 0.60, "nypost.com": 0.60, "dailymail.co.uk": 0.60,
    # Digital-first
    "huffpost.com": 0.60, "vice.com": 0.60, "buzzfeednews.com": 0.60,
    # Tech publications
    "wired.com": 0.60, "techcrunch.com": 0.60, "theverge.com": 0.60,
    "arstechnica.com": 0.60,
    # Indian outlets
    "hindustantimes.com": 0.60, "indiatoday.in": 0.60,
    "timesofindia.indiatimes.com": 0.60,
    # Platforms
    "medium.com": 0.60, "substack.com": 0.60,
}

# Default credibility for unknown domains
_DEFAULT_CREDIBILITY = 0.30


# ── Public API ───────────────────────────────────────────────────────────────


def extract_domain(url: str) -> str:
    """
    Extract clean domain from a URL.

    Strips protocol, www prefix, port, and path.
    """
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc or parsed.path.split("/")[0]
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        if ":" in domain:
            domain = domain.split(":")[0]
        return domain
    except Exception:
        return url.lower().strip()


def rank_source(domain: str) -> float:
    """
    Rank source credibility by domain.

    Checks exact match, subdomain match, and TLD-based scoring.

    Returns:
        Credibility float (0.0–1.0).
    """
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]

    # Exact match in tiers
    for tier in (TIER_1, TIER_2, TIER_3):
        if domain in tier:
            return tier[domain]

    # Subdomain match (e.g., news.bbc.co.uk → bbc.co.uk)
    for tier in (TIER_1, TIER_2, TIER_3):
        for known, score in tier.items():
            if domain.endswith(f".{known}"):
                return score

    # TLD-based scoring
    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return 0.90
    if domain.endswith(".edu") or domain.endswith(".ac.in") or domain.endswith(".ac.uk"):
        return 0.85
    if domain.endswith(".org"):
        return 0.65

    # Unknown domain
    return _DEFAULT_CREDIBILITY


def compute_aggregate_credibility(credibilities: List[float]) -> float:
    """
    Compute weighted average credibility with multi-source agreement boost.

    Multiple high-credibility sources agreeing boosts the aggregate score.

    Args:
        credibilities: List of individual source credibility scores.

    Returns:
        Aggregate credibility (0.0–1.0).
    """
    if not credibilities:
        return 0.0

    avg = sum(credibilities) / len(credibilities)

    # Agreement boost: multiple high-credibility sources
    high_count = sum(1 for c in credibilities if c >= 0.80)
    if high_count >= 3:
        avg = min(1.0, avg + 0.05)
    elif high_count >= 2:
        avg = min(1.0, avg + 0.03)

    return round(avg, 3)
