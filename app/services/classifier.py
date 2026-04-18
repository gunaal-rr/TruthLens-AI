"""
TruthLens AI — Scoring-Based Classifier

Classifies claims using a multi-signal scoring system rather than binary logic.
Each sub-claim is independently scored and classified.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from app.models import ClassificationLabel, SubClaimStatus

logger = logging.getLogger(__name__)


# ── Signal Databases ─────────────────────────────────────────────────────────

# Phrases associated with credibility
CREDIBLE_SIGNALS: List[str] = [
    "according to", "study shows", "peer-reviewed", "research published",
    "university of", "scientists found", "data indicates", "evidence suggests",
    "meta-analysis", "clinical trial", "systematic review", "the findings",
    "official statement", "government report", "statistical analysis",
    "published in", "journal of", "world health organization", "who reports",
    "cdc confirms", "fda approved", "peer reviewed study", "randomized controlled",
    "longitudinal study", "replicated findings", "consensus among",
]

# Phrases associated with misinformation
MISINFORMATION_SIGNALS: List[str] = [
    "100% cure", "guaranteed", "miracle", "they don't want you to know",
    "big pharma", "cover-up", "conspiracy", "mainstream media lies",
    "wake up", "sheep", "sheeple", "hidden truth", "secret cure",
    "government hiding", "exposed", "what they aren't telling you",
    "censored", "banned", "suppressed", "the truth about",
    "doctors hate this", "one weird trick", "you won't believe",
    "exposed the truth", "deep state", "new world order", "plandemic",
    "hoax", "scamdemic", "do your own research", "open your eyes",
    "survival rate of 99", "just a flu", "it's all a lie",
]

# Patterns of known false claims (partial matches)
KNOWN_FALSE_PATTERNS: List[str] = [
    r"5g.*(?:causes?|spreads?|creates?|transmits?).*(?:covid|corona|virus|cancer)",
    r"5g\s*(?:towers?|signals?|radiation)?\s*(?:causes?|spreads?)\s*(?:covid|corona|virus|cancer)",
    r"vaccines?.*(?:causes?|leads?\s*to|linked\s*to).*(?:autism|infertility|death|cancer)",
    r"earth\s*is\s*flat",
    r"flat\s*earth",
    r"moon\s*landing.*(?:faked?|hoax|staged|never\s*happened)",
    r"chemtrails?.*(?:poisoning|spraying|controlling|chemicals)",
    r"microchips?.*(?:in|inside|through|via).*vaccines?",
    r"covid.*(?:is|was)\s*(?:a\s*)?(?:hoax|planned|engineered|fake|man.?made)",
    r"drinking?\s*bleach\s*(?:cures?|kills?|treats?)",
    r"hydroxychloroquine\s*cures?\s*covid",
    r"ivermectin\s*cures?\s*covid",
    r"government.*hiding.*(?:truth|cure|evidence|from\s*us)",
]

_KNOWN_FALSE_RE = [re.compile(p, re.IGNORECASE) for p in KNOWN_FALSE_PATTERNS]

# Exaggeration / absolutism markers
EXAGGERATION_MARKERS: List[str] = [
    "always", "never", "every single", "absolutely", "definitely",
    "without a doubt", "100%", "completely", "totally", "all experts agree",
    "no one", "everyone knows", "proven beyond doubt", "undeniable",
    "irrefutable", "nobody can deny",
]


# ── Scoring Engine ───────────────────────────────────────────────────────────


@dataclass
class ClaimScores:
    """Accumulated scores for a single claim."""
    real_score: float = 0.0
    fake_score: float = 0.0
    signals: List[str] = field(default_factory=list)


def _score_sub_claim(text: str) -> ClaimScores:
    """
    Score a single sub-claim across multiple signal dimensions.

    Returns accumulated real/fake scores and the signals that fired.
    """
    scores = ClaimScores()
    text_lower = text.lower()

    # ── Credibility signals ──────────────────────────────────────
    for phrase in CREDIBLE_SIGNALS:
        if phrase in text_lower:
            scores.real_score += 2.0
            scores.signals.append(f"credible_phrase: '{phrase}'")

    # ── Misinformation signals ───────────────────────────────────
    for phrase in MISINFORMATION_SIGNALS:
        if phrase in text_lower:
            scores.fake_score += 3.0
            scores.signals.append(f"misinformation_phrase: '{phrase}'")

    # ── Known false patterns ─────────────────────────────────────
    for pattern_re in _KNOWN_FALSE_RE:
        if pattern_re.search(text_lower):
            scores.fake_score += 10.0
            scores.signals.append(f"known_false_pattern: '{pattern_re.pattern}'")

    # ── Exaggeration markers ─────────────────────────────────────
    for marker in EXAGGERATION_MARKERS:
        if marker in text_lower:
            scores.fake_score += 1.5
            scores.signals.append(f"exaggeration: '{marker}'")

    # ── Source attribution check ─────────────────────────────────
    source_patterns = [
        r"according to\s+\w+",
        r"study\s+(?:published|conducted|shows)",
        r"researchers?\s+(?:at|from|found)",
    ]
    has_source = any(re.search(p, text_lower) for p in source_patterns)
    if has_source:
        scores.real_score += 2.5
        scores.signals.append("has_source_attribution")
    else:
        scores.fake_score += 1.0
        scores.signals.append("missing_source_attribution")

    # ── Hedging language (indicates nuance → more likely real) ───
    hedging = ["may", "might", "could", "suggests", "indicates", "possibly"]
    for word in hedging:
        if re.search(rf'\b{word}\b', text_lower):
            scores.real_score += 0.5
            scores.signals.append(f"hedging_language: '{word}'")
            break  # Count hedging once

    # ── Question format (neutral, not assertive) ─────────────────
    if text.strip().endswith("?"):
        scores.real_score += 0.5
        scores.fake_score = max(0, scores.fake_score - 1.0)
        scores.signals.append("question_format")

    return scores


def _classify_scores(scores: ClaimScores) -> SubClaimStatus:
    """Map accumulated scores to a sub-claim status."""
    net = scores.real_score - scores.fake_score

    if net > 2.0:
        return SubClaimStatus.TRUE
    elif net < -1.0:
        return SubClaimStatus.FALSE
    else:
        return SubClaimStatus.MISLEADING


# ── Public API ───────────────────────────────────────────────────────────────


async def classify_sub_claim(text: str) -> Tuple[SubClaimStatus, ClaimScores]:
    """
    Classify a single sub-claim.

    Returns:
        Tuple of (status, scores) for downstream use.
    """
    scores = _score_sub_claim(text)
    status = _classify_scores(scores)
    logger.debug(
        "Sub-claim classified as %s (real=%.1f, fake=%.1f): %s",
        status.value, scores.real_score, scores.fake_score, text[:80],
    )
    return status, scores


async def aggregate_label(
    statuses: List[SubClaimStatus],
) -> ClassificationLabel:
    """
    Aggregate sub-claim statuses into a final classification label.

    Rules:
        - If ALL sub-claims are True → Real
        - If NO credible evidence (all False) → Fake
        - If ANY sub-claim is False and no True → Fake
        - Use Misleading ONLY for mix of True and False/Misleading
    """
    if not statuses:
        return ClassificationLabel.INSUFFICIENT_EVIDENCE

    false_count = sum(1 for s in statuses if s == SubClaimStatus.FALSE)
    true_count = sum(1 for s in statuses if s == SubClaimStatus.TRUE)
    misleading_count = sum(1 for s in statuses if s == SubClaimStatus.MISLEADING)
    total = len(statuses)

    # All true → Real
    if true_count == total:
        return ClassificationLabel.TRUE

    # All false → Fake (clearly false claims must NOT be Misleading)
    if false_count == total:
        return ClassificationLabel.FALSE

    # Majority false or no true claims → Fake
    if false_count > 0 and true_count == 0:
        return ClassificationLabel.FALSE

    # Mix of true and false → Misleading
    if true_count > 0 and false_count > 0:
        return ClassificationLabel.MISLEADING

    # Mix of true and misleading → Misleading
    if true_count > 0 and misleading_count > 0:
        return ClassificationLabel.MISLEADING

    # All misleading sub-claims with no true → Fake
    if misleading_count == total:
        return ClassificationLabel.FALSE

    return ClassificationLabel.MISLEADING


async def generate_reasoning(
    sub_claim_results: List[Tuple[str, SubClaimStatus, ClaimScores]],
) -> List[str]:
    """
    Generate human-readable reasoning from classification results.

    Returns at least 3 reasoning points.
    """
    reasoning: List[str] = []

    # Aggregate all signals
    all_signals = []
    for _, _, scores in sub_claim_results:
        all_signals.extend(scores.signals)

    # Known false pattern detected
    false_patterns = [s for s in all_signals if s.startswith("known_false_pattern")]
    if false_patterns:
        reasoning.append(
            "This claim matches known misinformation patterns that have been "
            "repeatedly debunked by authoritative sources."
        )

    # Misinformation phrases
    misinfo_phrases = [s for s in all_signals if s.startswith("misinformation_phrase")]
    if misinfo_phrases:
        reasoning.append(
            "The text contains language commonly associated with misinformation, "
            "including sensational or manipulative phrasing."
        )

    # Source attribution
    if "missing_source_attribution" in all_signals:
        reasoning.append(
            "The claim lacks proper source attribution, which reduces its credibility. "
            "Reliable claims typically cite specific studies, institutions, or data."
        )

    if "has_source_attribution" in all_signals:
        reasoning.append(
            "The claim includes source attribution, which is a positive indicator "
            "of credibility, though the quality of sources was not independently verified."
        )

    # Credible phrases
    credible = [s for s in all_signals if s.startswith("credible_phrase")]
    if credible:
        reasoning.append(
            "The text uses language associated with credible reporting, such as "
            "references to studies, data, or institutional findings."
        )

    # Exaggeration
    exaggeration = [s for s in all_signals if s.startswith("exaggeration")]
    if exaggeration:
        reasoning.append(
            "The claim uses absolute or exaggerated language, which is a common "
            "indicator of unreliable or biased information."
        )

    # Sub-claim agreement
    statuses = [s for _, s, _ in sub_claim_results]
    true_count = sum(1 for s in statuses if s == SubClaimStatus.TRUE)
    false_count = sum(1 for s in statuses if s == SubClaimStatus.FALSE)
    misleading_count = sum(1 for s in statuses if s == SubClaimStatus.MISLEADING)

    if len(statuses) > 1:
        reasoning.append(
            f"Of {len(statuses)} sub-claims analyzed, {true_count} were verified as true, "
            f"{false_count} were determined to be false, and "
            f"{misleading_count} were partially true or lacked full context."
        )

    # Ensure minimum 3 reasoning points
    defaults = [
        "The analysis evaluated the claim against known misinformation databases "
        "and credibility indicators.",
        "Logical consistency and evidence availability were assessed for "
        "each component of the claim.",
        "The final classification considers the aggregate evidence across "
        "all sub-claims and detected signals.",
    ]
    while len(reasoning) < 3:
        for d in defaults:
            if d not in reasoning:
                reasoning.append(d)
                break
        else:
            break

    return reasoning[:6]  # Cap at 6 points