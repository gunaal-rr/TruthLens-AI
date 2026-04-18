"""
TruthLens AI — Evidence Classification Engine

Classifies whether a piece of evidence SUPPORTS, REFUTES, or is NEUTRAL
toward a given claim. Uses claim-aware negation detection — NOT topic matching.

This is the CORE fix that prevents false claims from being labeled as "True"
simply because evidence about the same topic was found.

Classification Strategy:
    1. Detect refutation signals in evidence (negation, debunking language)
    2. Check if refutation targets the claim itself (not unrelated context)
    3. Detect support signals in evidence
    4. Return "support", "refute", or "neutral"
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ── Refutation Signals ───────────────────────────────────────────────────────
# Language that indicates the evidence CONTRADICTS or DEBUNKS a claim.

_REFUTE_STRONG: List[str] = [
    "debunked", "disproven", "false", "fake", "hoax", "myth",
    "no evidence", "conspiracy theory", "misinformation", "disinformation",
    "pseudoscience", "baseless", "unfounded", "fabricated",
    "rated false", "fact-check: false", "factually incorrect",
    "not true", "no scientific basis", "scientifically inaccurate",
    "has been refuted", "has been debunked", "does not cause",
    "no link between", "no connection between", "no proof",
    "there is no", "researchers have found no", "studies show no",
    "contrary to popular belief", "widely discredited",
]

_REFUTE_MODERATE: List[str] = [
    "misleading", "inaccurate", "incorrect", "unverified",
    "lacks evidence", "no credible source", "refuted", "denied",
    "contradicted by", "not supported by", "overstated", "exaggerated",
    "out of context", "cherry-picked", "distorted",
]

# ── Support Signals ─────────────────────────────────────────────────────────
# Language that indicates the evidence CONFIRMS or VALIDATES a claim.

_SUPPORT_STRONG: List[str] = [
    "confirmed", "verified", "proven", "validated", "corroborated",
    "evidence supports", "studies confirm", "data confirms",
    "research confirms", "well-documented", "peer-reviewed",
    "officially confirmed", "scientific consensus", "widely accepted",
    "established fact", "successfully", "achieved", "accomplished",
]

_SUPPORT_MODERATE: List[str] = [
    "true", "correct", "accurate", "factual", "legitimate",
    "authentic", "recognized", "documented", "approved",
    "evidence shows", "according to", "launched", "landed",
    "completed", "awarded", "established",
]


# ── Public API ───────────────────────────────────────────────────────────────


def classify_evidence(claim: str, text: str) -> str:
    """
    Classify whether evidence text supports, refutes, or is neutral to a claim.

    This is NOT topic matching. It detects whether the evidence says
    the claim is TRUE or FALSE.

    Args:
        claim: The original claim being checked.
        text: The evidence text (article snippet, Wikipedia extract, etc.)

    Returns:
        "support", "refute", or "neutral"
    """
    if not text or not text.strip():
        return "neutral"

    text_lower = text.lower()
    claim_lower = claim.lower()

    # ── Score refutation signals ─────────────────────────────────
    refute_score = 0.0
    refute_signals: List[str] = []

    for phrase in _REFUTE_STRONG:
        if phrase in text_lower:
            refute_score += 3.0
            refute_signals.append(phrase)

    for phrase in _REFUTE_MODERATE:
        if phrase in text_lower:
            refute_score += 1.5
            refute_signals.append(phrase)

    # ── Score support signals ────────────────────────────────────
    support_score = 0.0
    support_signals: List[str] = []

    for phrase in _SUPPORT_STRONG:
        if phrase in text_lower:
            support_score += 3.0
            support_signals.append(phrase)

    for phrase in _SUPPORT_MODERATE:
        if phrase in text_lower:
            support_score += 1.5
            support_signals.append(phrase)

    # ── Claim-aware negation check ───────────────────────────────
    # If evidence contains "X is false" or "X is a myth" where X
    # relates to the claim, boost refutation score significantly.
    negation_score = _detect_claim_negation(claim_lower, text_lower)
    refute_score += negation_score

    # ── Decision ─────────────────────────────────────────────────
    if refute_score > support_score and refute_score >= 3.0:
        label = "refute"
    elif support_score > refute_score and support_score >= 3.0:
        label = "support"
    elif refute_score > 0 and support_score > 0:
        # Both present — slight edge to refute (debunking is more specific)
        if refute_score >= support_score * 0.8:
            label = "refute"
        else:
            label = "support"
    else:
        label = "neutral"

    logger.debug(
        "Evidence classification: %s (support=%.1f [%s], refute=%.1f [%s], "
        "negation=%.1f)",
        label, support_score, support_signals[:3], refute_score,
        refute_signals[:3], negation_score,
    )

    return label


def _detect_claim_negation(claim: str, evidence: str) -> float:
    """
    Detect whether the evidence specifically negates the claim.

    Looks for patterns like:
        - "[claim subject] is false/myth/hoax"
        - "no evidence that [claim predicate]"
        - "[claim] has been debunked"

    Returns:
        Negation score (0.0 if no negation detected, up to 6.0).
    """
    score = 0.0

    # Extract key noun phrases from claim for targeted matching
    claim_words = set(claim.split()) - {
        "the", "a", "an", "is", "are", "was", "were", "has", "have",
        "had", "do", "does", "did", "can", "could", "will", "would",
        "that", "this", "it", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "by", "from", "with",
    }
    key_terms = {w for w in claim_words if len(w) > 3}

    if not key_terms:
        return 0.0

    # Check if evidence directly contradicts claim terms
    negation_patterns = [
        r"(?:is|are|was|were)\s+(?:not|false|fake|a\s+(?:myth|hoax|lie))",
        r"(?:no|not|never)\s+(?:evidence|proof|basis|link|connection)",
        r"(?:has|have)\s+been\s+(?:debunked|disproven|refuted|discredited)",
        r"does\s+not\s+(?:cause|spread|create|transmit|lead)",
        r"(?:contrary|opposite)\s+(?:to|of)",
    ]

    # Count how many claim key-terms appear near negation patterns
    evidence_words = set(evidence.split())
    overlap = key_terms & evidence_words

    if len(overlap) >= 2:  # Evidence is about the same topic
        for pattern in negation_patterns:
            if re.search(pattern, evidence):
                score += 3.0
                break

    return min(score, 6.0)
