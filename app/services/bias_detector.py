"""
TruthLens AI — Bias & Misinformation Detection Service

Detects sensational language, emotional manipulation, overgeneralization,
and missing sources. Produces a bias_detected flag and bias_score (0–1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ── Bias Signal Lexicons ─────────────────────────────────────────────────────

SENSATIONAL_WORDS: List[str] = [
    "shocking", "bombshell", "explosive", "unbelievable", "jaw-dropping",
    "mind-blowing", "devastating", "terrifying", "horrifying", "outrageous",
    "scandalous", "incredible", "insane", "crazy", "wild", "epic",
    "nightmare", "catastrophic", "unprecedented", "alarming", "breaking",
    "urgent", "exclusive", "developing", "massive", "huge",
]

EMOTIONAL_MANIPULATION_PHRASES: List[str] = [
    "you need to see this", "share before it's deleted", "share before they remove",
    "this will make you cry", "you won't believe what happened",
    "pray for", "like and share", "spread the word", "forward this to everyone",
    "the media won't show you this", "what they don't want you to know",
    "open your eyes", "wake up", "think about your children",
    "protect your family", "they are lying to you", "don't be fooled",
    "the truth will shock you", "if you care about",
    # Conspiracy / cover-up language
    "government is hiding", "hiding this from us", "hiding the truth",
    "cover-up", "cover up", "being silenced", "being suppressed",
    "they don't want you to know", "exposed", "exposed the truth",
    "big pharma", "deep state", "new world order",
]

OVERGENERALIZATION_PATTERNS: List[str] = [
    r"\ball\s+(?:doctors?|scientists?|experts?|studies|research)\b",
    r"\bnobody\s+(?:talks?|mentions?|knows?)\b",
    r"\beveryone\s+(?:knows?|agrees?|says?)\b",
    r"\bnever\s+(?:before|in\s+history)\b",
    r"\balways\s+(?:happens?|works?|fails?)\b",
    r"\bno\s+(?:doctor|scientist|expert)\s+will\s+tell\b",
    r"\b(?:all|every)\s+single\b",
    r"\bwithout\s+exception\b",
]

CLICKBAIT_PATTERNS: List[str] = [
    r"(?:you\s+)?won'?t\s+believe",
    r"what\s+happens?\s+next",
    r"number\s+\d+\s+will\s+(?:shock|surprise)",
    r"(?:doctors?|scientists?)\s+(?:hate|don'?t\s+want)",
    r"one\s+(?:simple|weird|strange)\s+trick",
    r"\d+\s+(?:reasons?|ways?|things?)\s+(?:you|that)",
]

_OVERGENERALIZATION_RE = [re.compile(p, re.IGNORECASE) for p in OVERGENERALIZATION_PATTERNS]
_CLICKBAIT_RE = [re.compile(p, re.IGNORECASE) for p in CLICKBAIT_PATTERNS]


# ── Result Model ─────────────────────────────────────────────────────────────


@dataclass
class BiasReport:
    """Complete bias analysis report."""
    bias_detected: bool = False
    bias_score: float = 0.0  # 0–1
    keywords_detected: List[str] = field(default_factory=list)
    lack_of_evidence: bool = False
    signals: List[str] = field(default_factory=list)


# ── Detection Engine ─────────────────────────────────────────────────────────


async def detect_bias(text: str, has_source_attribution: bool = False) -> BiasReport:
    """
    Analyze text for bias, sensationalism, and misinformation indicators.

    Args:
        text: The claim or news text.
        has_source_attribution: Whether the text references sources.

    Returns:
        BiasReport with scores, flags, and detected keywords.
    """
    report = BiasReport()
    text_lower = text.lower()
    raw_score = 0.0
    max_possible = 0.0

    # ── Sensational words ────────────────────────────────────────
    max_possible += 30.0
    for word in SENSATIONAL_WORDS:
        if word in text_lower:
            raw_score += 3.0
            report.keywords_detected.append(word)
            report.signals.append(f"sensational: {word}")

    # ── Emotional manipulation ───────────────────────────────────
    max_possible += 25.0
    for phrase in EMOTIONAL_MANIPULATION_PHRASES:
        if phrase in text_lower:
            raw_score += 5.0
            report.keywords_detected.append(phrase)
            report.signals.append(f"emotional_manipulation: {phrase}")

    # ── Overgeneralization ───────────────────────────────────────
    max_possible += 20.0
    for pattern_re in _OVERGENERALIZATION_RE:
        match = pattern_re.search(text_lower)
        if match:
            matched_text = match.group()
            raw_score += 4.0
            report.keywords_detected.append(matched_text.strip())
            report.signals.append(f"overgeneralization: {matched_text.strip()}")

    # ── Clickbait patterns ───────────────────────────────────────
    max_possible += 15.0
    for pattern_re in _CLICKBAIT_RE:
        match = pattern_re.search(text_lower)
        if match:
            matched_text = match.group()
            raw_score += 5.0
            report.keywords_detected.append(matched_text.strip())
            report.signals.append(f"clickbait: {matched_text.strip()}")

    # ── Excessive punctuation (!!!, ???, ALL CAPS) ───────────────
    max_possible += 10.0
    exclamation_count = text.count("!")
    question_count = text.count("?")
    if exclamation_count >= 3:
        raw_score += min(exclamation_count, 5)
        report.signals.append(f"excessive_exclamation: {exclamation_count}")
        report.keywords_detected.append("!!!")

    if question_count >= 3:
        raw_score += min(question_count - 2, 3)
        report.signals.append(f"excessive_questions: {question_count}")

    # All-caps word ratio
    words = text.split()
    if words:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
        if caps_ratio > 0.3:
            raw_score += 5.0
            report.signals.append(f"excessive_caps: ratio={caps_ratio:.2f}")
            report.keywords_detected.append("ALL CAPS")

    # ── Missing source attribution ───────────────────────────────
    if not has_source_attribution:
        report.lack_of_evidence = True
        raw_score += 3.0
        report.signals.append("missing_sources")

    # ── Normalize score to 0–1 ───────────────────────────────────
    max_possible = max(max_possible, 1.0)
    report.bias_score = round(min(raw_score / max_possible, 1.0), 3)
    report.bias_detected = report.bias_score > 0.10

    # Ensure at least one meaningful keyword is always present
    if not report.keywords_detected:
        # Extract meaningful keywords from the text itself
        import re as _re
        _stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "it", "its", "this", "that", "and", "but", "or",
            "for", "so", "to", "of", "in", "on", "at", "by", "from", "with",
            "about", "not", "no", "they", "their", "them", "we", "you", "he", "she",
        }
        words = _re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        meaningful = [w for w in words if w not in _stopwords]
        # Take up to 3 unique meaningful keywords
        seen_kw = set()
        for w in meaningful:
            if w not in seen_kw:
                seen_kw.add(w)
                report.keywords_detected.append(w)
            if len(report.keywords_detected) >= 3:
                break
        # Ultimate fallback
        if not report.keywords_detected:
            report.keywords_detected = ["unverified claim"]

    logger.debug(
        "Bias analysis: detected=%s, score=%.3f, keywords=%d",
        report.bias_detected, report.bias_score, len(report.keywords_detected),
    )

    return report
