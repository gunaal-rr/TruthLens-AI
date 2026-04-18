"""
TruthLens AI — Decision Engine (v2)

Centralized classification decision logic based on evidence scores.
Uses ratio-based analysis with multi-source refutation override.

Decision Flow:
    0. Strong evidence override (confirm > 0.8, deny < 0.2 -> Real)
    1. Strong refutation override (deny > confirm AND refute_count >= 1 -> Fake)  [FIX 3]
    2. No evidence -> Insufficient Evidence
    3. Ratio-based: > 0.7 -> Real, < 0.3 -> Fake, else Misleading  [FIX 8]
    4. Edge cases: rule-based label as tiebreaker

Labels: Real / Fake / Misleading / Insufficient Evidence
NEVER outputs True / False.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models import ClassificationLabel

logger = logging.getLogger(__name__)


def decide(
    confirm_score: float,
    deny_score: float,
    has_evidence: bool,
    rule_based_label: Optional[ClassificationLabel] = None,
    source_credibility: float = 0.0,
    support_count: int = 0,
    refute_count: int = 0,
) -> ClassificationLabel:
    """
    Make final classification decision based on evidence.

    This is the ONLY decision logic in the system.
    Evidence-first; rule-based label used only as tiebreaker.

    Args:
        confirm_score: Normalized confirmation score (0-1).
        deny_score: Normalized denial score (0-1).
        has_evidence: Whether any external evidence was found.
        rule_based_label: Rule-based pre-check result (LOW weight).
        source_credibility: Average source credibility (0-1).
        support_count: Number of sources that support the claim.
        refute_count: Number of sources that refute the claim.

    Returns:
        Final ClassificationLabel (Real/Fake/Misleading/Insufficient Evidence).
    """
    # -- Strong evidence override: overwhelming confirmation
    if has_evidence and confirm_score > 0.8 and deny_score < 0.2:
        logger.info(
            "Decision: REAL (strong evidence override -- confirm=%.3f, deny=%.3f)",
            confirm_score, deny_score,
        )
        return ClassificationLabel.TRUE

    # -- Strong evidence override: overwhelming denial
    if has_evidence and deny_score > 0.8 and confirm_score < 0.2:
        logger.info(
            "Decision: FAKE (strong evidence override -- confirm=%.3f, deny=%.3f)",
            confirm_score, deny_score,
        )
        return ClassificationLabel.FALSE

    # -- FIX 3: Strong refutation override
    # If deny_score beats confirm_score AND at least 1 source refutes,
    # this MUST override weak positive signals.
    if has_evidence and deny_score > confirm_score and refute_count >= 1:
        logger.info(
            "Decision: FAKE (refutation override -- deny=%.3f > confirm=%.3f, "
            "refute_count=%d)",
            deny_score, confirm_score, refute_count,
        )
        return ClassificationLabel.FALSE

    # -- No evidence -> Insufficient Evidence
    if not has_evidence:
        logger.info("Decision: INSUFFICIENT_EVIDENCE (no external evidence found)")
        return ClassificationLabel.INSUFFICIENT_EVIDENCE

    # -- FIX 8: Strict ratio-based decision
    total = confirm_score + deny_score

    if total == 0:
        logger.info("Decision: INSUFFICIENT_EVIDENCE (zero evidence scores)")
        return ClassificationLabel.INSUFFICIENT_EVIDENCE

    ratio = confirm_score / total

    logger.info(
        "Decision engine: confirm=%.3f, deny=%.3f, ratio=%.3f, "
        "credibility=%.3f, support=%d, refute=%d, rule_based=%s",
        confirm_score, deny_score, ratio,
        source_credibility, support_count, refute_count,
        rule_based_label.value if rule_based_label else "None",
    )

    if ratio > 0.7:
        label = ClassificationLabel.TRUE
    elif ratio < 0.3:
        label = ClassificationLabel.FALSE
    elif 0.4 <= ratio <= 0.6:
        label = ClassificationLabel.MISLEADING
    else:
        # Edge zones (0.3-0.4 and 0.6-0.7): use rule-based as tiebreaker
        if ratio > 0.5:
            if rule_based_label == ClassificationLabel.TRUE:
                label = ClassificationLabel.TRUE
            else:
                label = ClassificationLabel.MISLEADING
        else:
            if rule_based_label == ClassificationLabel.FALSE:
                label = ClassificationLabel.FALSE
            else:
                label = ClassificationLabel.MISLEADING

    logger.info("Decision: %s (ratio=%.3f)", label.value, ratio)
    return label
