"""
TruthLens AI — Evidence-Based Trust Score Computation

Computes claim reliability score based on:
    - Evidence confidence (60% weight)
    - Source credibility (30% weight)
    - Rule-based score (10% weight)
    - High-credibility source bonus (+15 if >= 2 high-cred sources)

Formula:
    trust_score = int(
        (evidence_confidence * 0.6 +
         source_credibility * 0.3 +
         rule_based_score * 0.1) * 100
    ) + high_credibility_bonus
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def compute_trust_score(
    evidence_confidence: float,
    source_credibility: float,
    rule_based_score: float,
    source_credibilities: Optional[List[float]] = None,
) -> int:
    """
    Compute evidence-based trust score.

    Args:
        evidence_confidence: Evidence quality/confidence (0–1).
        source_credibility: Average source credibility (0–1).
        rule_based_score: Normalized rule-based classifier score (0–1).
        source_credibilities: Optional list of individual source credibility
            scores, used for high-credibility bonus calculation.

    Returns:
        Trust score integer (0–100).
    """
    trust = (
        evidence_confidence * 0.6 +
        source_credibility * 0.3 +
        rule_based_score * 0.1
    )

    trust_score = int(round(trust * 100))

    # ── High-credibility source bonus ────────────────────────────
    if source_credibilities:
        high_credibility_sources = sum(
            1 for c in source_credibilities if c >= 0.80
        )
        if high_credibility_sources >= 2:
            trust_score += 15
            logger.debug(
                "High-credibility bonus applied: +15 (%d sources >= 0.80)",
                high_credibility_sources,
            )

    # Clamp to [0, 100]
    trust_score = max(0, min(100, trust_score))

    logger.debug(
        "Trust score: %d (evidence=%.3f, credibility=%.3f, rules=%.3f)",
        trust_score, evidence_confidence, source_credibility, rule_based_score,
    )

    return trust_score
