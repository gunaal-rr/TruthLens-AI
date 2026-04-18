"""
TruthLens AI — Evidence-Based Confidence Calibration

Computes system confidence based on external evidence quality:
    - Number of external sources
    - Agreement ratio between sources
    - Average source credibility

Formula:
    confidence = min(95, int(
        (num_sources * 8) +
        (agreement_ratio * 40) +
        (avg_credibility * 30)
    ))

Cap: 95 (never 100 — always reflect uncertainty)
Floor: 10
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def calibrate_confidence(
    num_sources: int,
    confirm_score: float,
    deny_score: float,
    avg_credibility: float,
) -> int:
    """
    Compute evidence-based confidence score.

    Args:
        num_sources: Number of valid external sources found.
        confirm_score: Normalized confirmation score (0–1).
        deny_score: Normalized denial score (0–1).
        avg_credibility: Average source credibility (0–1).

    Returns:
        Confidence integer (10–95).
    """
    # Agreement ratio: how strongly sources agree on a direction
    total = confirm_score + deny_score
    if total > 0:
        agreement_ratio = max(confirm_score, deny_score) / total
    else:
        agreement_ratio = 0.0

    # Evidence-based formula — capped at 95 to reflect inherent uncertainty
    confidence = min(95, int(
        (num_sources * 8) +
        (agreement_ratio * 40) +
        (avg_credibility * 30)
    ))

    # Floor: minimum 10% confidence
    confidence = max(10, confidence)

    logger.debug(
        "Confidence calibrated: %d (sources=%d, agreement=%.3f, credibility=%.3f)",
        confidence, num_sources, agreement_ratio, avg_credibility,
    )

    return confidence
