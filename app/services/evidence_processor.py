"""
TruthLens AI — External Evidence Processor

Processes and scores external evidence from APIs (Gemini, Wikipedia,
Google News, etc.) to supplement rule-based analysis.

External evidence is treated as PRIMARY signal for classification decisions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models import NewsResult, FactSource, TrustLevel

logger = logging.getLogger(__name__)


# ── Sentiment / Verdict Keywords ─────────────────────────────────────────────

# Phrases in external evidence that CONFIRM a claim
CONFIRMATION_SIGNALS: List[str] = [
    "confirmed", "verified", "true", "accurate", "correct", "factual",
    "proven", "well-documented", "supported by evidence", "widely accepted",
    "scientific consensus", "peer-reviewed", "officially stated",
    "evidence supports", "data confirms", "studies show", "research confirms",
    "credible", "authentic", "legitimate", "validated", "corroborated",
]

# Phrases in external evidence that DENY a claim
DENIAL_SIGNALS: List[str] = [
    "debunked", "false", "fake", "hoax", "disproven", "no evidence",
    "misleading", "inaccurate", "fabricated", "conspiracy", "myth",
    "misinformation", "disinformation", "baseless", "unfounded",
    "no scientific basis", "not supported", "contradicted by",
    "fact-check: false", "rated false", "unverified", "refuted",
    "lacks evidence", "no credible source", "pseudoscience",
]

# Phrases indicating MIXED / partially true
MIXED_SIGNALS: List[str] = [
    "partially true", "misleading", "out of context", "lacks context",
    "exaggerated", "distorted", "half-truth", "oversimplified",
    "nuanced", "incomplete", "cherry-picked", "taken out of context",
    "mostly false", "mostly true", "needs context",
]


# ── Evidence Report ──────────────────────────────────────────────────────────


@dataclass
class ExternalEvidenceReport:
    """Aggregated analysis of all external evidence."""
    has_external_data: bool = False
    confirmation_score: float = 0.0   # 0–1: how much evidence confirms claim
    denial_score: float = 0.0         # 0–1: how much evidence denies claim
    mixed_score: float = 0.0          # 0–1: how much evidence is mixed
    source_count: int = 0             # Number of external sources
    evidence_summary: str = ""        # Brief summary of what evidence says
    signals: List[str] = field(default_factory=list)
    extracted_sources: List[FactSource] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)

    @property
    def net_score(self) -> float:
        """Positive = confirms, Negative = denies."""
        return self.confirmation_score - self.denial_score

    @property
    def evidence_strength(self) -> str:
        """How strong is the external evidence overall."""
        total = self.confirmation_score + self.denial_score + self.mixed_score
        if total < 0.1:
            return "none"
        elif total < 0.3:
            return "weak"
        elif total < 0.6:
            return "moderate"
        else:
            return "strong"


# ── Scoring Functions ────────────────────────────────────────────────────────


def _score_text(text: str) -> Tuple[float, float, float, List[str]]:
    """
    Score a text block for confirmation, denial, and mixed signals.

    Returns:
        (confirmation, denial, mixed, signals_fired)
    """
    text_lower = text.lower()
    confirmation = 0.0
    denial = 0.0
    mixed = 0.0
    signals: List[str] = []

    for phrase in CONFIRMATION_SIGNALS:
        if phrase in text_lower:
            confirmation += 1.0
            signals.append(f"confirms: '{phrase}'")

    for phrase in DENIAL_SIGNALS:
        if phrase in text_lower:
            denial += 1.5  # Denial signals weighted slightly higher
            signals.append(f"denies: '{phrase}'")

    for phrase in MIXED_SIGNALS:
        if phrase in text_lower:
            mixed += 1.0
            signals.append(f"mixed: '{phrase}'")

    return confirmation, denial, mixed, signals


def _extract_trust_level(source: str) -> TrustLevel:
    """Determine trust level from source domain."""
    high_trust = [
        "reuters", "apnews", "bbc", "nytimes", "washingtonpost",
        "nature.com", "science.org", "who.int", "cdc.gov", "nih.gov",
        "nasa.gov", "gov.uk", ".edu", "lancet", "bmj.com",
        "wikipedia.org", "britannica", "snopes", "politifact",
        "factcheck.org", "fullfact.org",
    ]
    medium_trust = [
        "theguardian", "forbes", "bloomberg", "economist",
        "time.com", "newsweek", "usatoday", "cbsnews", "nbcnews",
        "abcnews", "aljazeera", "dw.com", "france24",
    ]

    source_lower = source.lower()
    for domain in high_trust:
        if domain in source_lower:
            return TrustLevel.HIGH
    for domain in medium_trust:
        if domain in source_lower:
            return TrustLevel.MEDIUM
    return TrustLevel.LOW


# ── Public API ───────────────────────────────────────────────────────────────


async def process_external_evidence(
    external_evidence: Optional[str] = None,
    news_results: Optional[List[NewsResult]] = None,
    wiki_summary: Optional[str] = None,
    ai_analysis: Optional[str] = None,
) -> ExternalEvidenceReport:
    """
    Process all external evidence sources and produce a unified report.

    Priority:
        1. news_results (multiple source corroboration)
        2. external_evidence (API-provided structured data)
        3. wiki_summary (encyclopedic reference)
        4. ai_analysis (LLM reasoning)

    Returns:
        ExternalEvidenceReport with scores, signals, and extracted sources.
    """
    report = ExternalEvidenceReport()

    total_confirm = 0.0
    total_deny = 0.0
    total_mixed = 0.0
    max_possible = 0.0

    # ── Process News Results (highest weight: 3x) ────────────────
    if news_results and len(news_results) > 0:
        report.has_external_data = True
        report.source_count += len(news_results)

        for nr in news_results:
            combined = f"{nr.title} {nr.snippet}"
            c, d, m, sigs = _score_text(combined)

            total_confirm += c * 3.0
            total_deny += d * 3.0
            total_mixed += m * 3.0
            max_possible += 6.0  # realistic ceiling per news result
            report.signals.extend(sigs)

            # Extract as fact source
            if nr.title and nr.source:
                trust = _extract_trust_level(nr.source)
                report.extracted_sources.append(
                    FactSource(
                        title=nr.title[:100],
                        source=nr.source or nr.url or "unknown",
                        trust=trust,
                    )
                )

            if nr.snippet:
                report.key_findings.append(nr.snippet[:150])

    # ── Process External Evidence (weight: 2x) ───────────────────
    if external_evidence and external_evidence.strip():
        report.has_external_data = True
        report.source_count += 1

        c, d, m, sigs = _score_text(external_evidence)
        total_confirm += c * 2.0
        total_deny += d * 2.0
        total_mixed += m * 2.0
        max_possible += 6.0
        report.signals.extend(sigs)

        # Extract key sentence as finding
        sentences = re.split(r'[.!?]+', external_evidence)
        for s in sentences[:2]:
            s = s.strip()
            if len(s) > 20:
                report.key_findings.append(s[:150])

    # ── Process Wikipedia Summary (weight: 2x) ───────────────────
    if wiki_summary and wiki_summary.strip():
        report.has_external_data = True
        report.source_count += 1

        c, d, m, sigs = _score_text(wiki_summary)
        total_confirm += c * 2.0
        total_deny += d * 2.0
        total_mixed += m * 2.0
        max_possible += 6.0
        report.signals.extend(sigs)

        report.extracted_sources.append(
            FactSource(
                title="Wikipedia Reference",
                source="en.wikipedia.org",
                trust=TrustLevel.MEDIUM,
            )
        )

        sentences = re.split(r'[.!?]+', wiki_summary)
        for s in sentences[:1]:
            s = s.strip()
            if len(s) > 20:
                report.key_findings.append(s[:150])

    # ── Process AI Analysis (weight: 1.5x) ────────────────────────
    if ai_analysis and ai_analysis.strip():
        report.has_external_data = True
        report.source_count += 1

        c, d, m, sigs = _score_text(ai_analysis)
        total_confirm += c * 1.5
        total_deny += d * 1.5
        total_mixed += m * 1.5
        max_possible += 5.0
        report.signals.extend(sigs)

        report.extracted_sources.append(
            FactSource(
                title="AI Model Analysis",
                source="gemini/llm analysis",
                trust=TrustLevel.MEDIUM,
            )
        )

    # ── Normalize Scores ─────────────────────────────────────────
    max_possible = max(max_possible, 1.0)
    report.confirmation_score = round(min(total_confirm / max_possible, 1.0), 3)
    report.denial_score = round(min(total_deny / max_possible, 1.0), 3)
    report.mixed_score = round(min(total_mixed / max_possible, 1.0), 3)

    # ── Build Summary ────────────────────────────────────────────
    if report.has_external_data:
        if report.denial_score > report.confirmation_score:
            report.evidence_summary = (
                f"External evidence from {report.source_count} source(s) "
                f"predominantly contradicts this claim."
            )
        elif report.confirmation_score > report.denial_score:
            report.evidence_summary = (
                f"External evidence from {report.source_count} source(s) "
                f"predominantly supports this claim."
            )
        else:
            report.evidence_summary = (
                f"External evidence from {report.source_count} source(s) "
                f"provides mixed or inconclusive signals."
            )
    else:
        report.evidence_summary = "No external evidence was provided for verification."

    # Limit extracted sources
    report.extracted_sources = report.extracted_sources[:5]
    report.key_findings = report.key_findings[:4]

    logger.info(
        "External evidence processed: has_data=%s, confirm=%.3f, deny=%.3f, "
        "mixed=%.3f, sources=%d, strength=%s",
        report.has_external_data, report.confirmation_score, report.denial_score,
        report.mixed_score, report.source_count, report.evidence_strength,
    )

    return report
