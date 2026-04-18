"""
TruthLens AI — Context-Aware Evidence Generator

Generates meaningful evidence artifacts by extracting key entities from the claim
rather than producing generic queries.

Updated for new ClassificationLabel values (TRUE/FALSE/MISLEADING/INSUFFICIENT_EVIDENCE).
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from app.models import ClassificationLabel

logger = logging.getLogger(__name__)

# ── Entity Extraction Patterns ───────────────────────────────────────────────

# Organizations / institutions
_ORG_PATTERN = re.compile(
    r'\b(?:WHO|CDC|FDA|NIH|NASA|UN|UNESCO|UNICEF|EU|NATO|'
    r'World Health Organization|Centers for Disease Control|'
    r'Food and Drug Administration|National Institutes of Health|'
    r'European Union|United Nations)\\b',
    re.IGNORECASE,
)

# Medical / scientific terms
_MEDICAL_PATTERN = re.compile(
    r'\b(?:vaccine|covid|coronavirus|pandemic|virus|cancer|'
    r'treatment|cure|therapy|drug|medication|clinical|trial|'
    r'study|research|symptoms|diagnosis|dna|rna|mrna|protein|'
    r'antibody|immunity|infection|disease|health|medical)\b',
    re.IGNORECASE,
)

# Political / social entities
_POLITICAL_PATTERN = re.compile(
    r'\b(?:president|government|congress|senate|parliament|'
    r'election|vote|democracy|republican|democrat|political|'
    r'policy|law|regulation|supreme court|legislation)\b',
    re.IGNORECASE,
)

# Technology terms
_TECH_PATTERN = re.compile(
    r'\b(?:5g|ai|artificial intelligence|blockchain|crypto|'
    r'bitcoin|technology|internet|social media|algorithm|'
    r'data|privacy|surveillance|cybersecurity|quantum)\b',
    re.IGNORECASE,
)

# Named entities (capitalized multi-word phrases — heuristic NER)
_NAMED_ENTITY_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
)


def _extract_entities(text: str) -> List[str]:
    """Extract key entities from text for query generation."""
    entities: List[str] = []

    for pattern in [_ORG_PATTERN, _MEDICAL_PATTERN, _POLITICAL_PATTERN, _TECH_PATTERN]:
        matches = pattern.findall(text)
        entities.extend(m.strip() for m in matches)

    # Named entities (heuristic)
    named = _NAMED_ENTITY_PATTERN.findall(text)
    entities.extend(named[:3])

    # Deduplicate, preserve order
    seen = set()
    unique: List[str] = []
    for e in entities:
        lower = e.lower()
        if lower not in seen and len(e) > 1:
            seen.add(lower)
            unique.append(e)

    return unique[:8]  # Limit to 8 entities


def _extract_core_topic(text: str) -> str:
    """Extract the most important 3-5 word topic from the text."""
    # Remove common stopwords for topic extraction
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "it", "its", "this", "that",
        "and", "but", "or", "nor", "for", "so", "yet", "to", "of", "in", "on",
        "at", "by", "from", "with", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "not", "no", "if",
        "then", "than", "very", "just", "also", "they", "their", "them",
        "we", "our", "you", "your", "he", "she", "his", "her", "i", "me",
        "my", "who", "what", "which", "when", "where", "why", "how",
    }

    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    meaningful = [w for w in words if w.lower() not in stopwords]

    return " ".join(meaningful[:5]) if meaningful else text[:50]


# ── Public API ───────────────────────────────────────────────────────────────


async def generate_evidence(
    claim: str,
    label: ClassificationLabel,
    sub_claims: List[str],
) -> Tuple[str, List[str]]:
    """
    Generate context-aware evidence artifacts.

    Args:
        claim: The core claim text.
        label: Classification label.
        sub_claims: List of sub-claim texts.

    Returns:
        Tuple of:
            - image_explanation: What visual proof would validate the claim
            - video_search_queries: 3 YouTube search queries for verification
    """
    entities = _extract_entities(claim)
    topic = _extract_core_topic(claim)
    entity_str = ", ".join(entities[:4]) if entities else topic

    # ── Image Explanation (claim-specific, not generic) ────────────
    claim_short = claim[:100].rstrip(".")
    primary_entity = entities[0] if entities else "relevant authority"

    if label == ClassificationLabel.FALSE:
        image_explanation = (
            f"A screenshot or scan of an official statement from {primary_entity} "
            f"explicitly addressing and debunking the claim that '{claim_short}'. "
            f"The image should show the original false claim alongside the verified "
            f"correction, with data points or citations disproving each specific assertion. "
            f"Key entities to verify: {entity_str}."
        )
    elif label == ClassificationLabel.MISLEADING:
        image_explanation = (
            f"A side-by-side visual comparison showing what the claim '{claim_short}' "
            f"states versus what {primary_entity} has actually reported. The image "
            f"should highlight the specific context or data points that the original "
            f"claim omits or misrepresents, with source citations for each correction."
        )
    elif label == ClassificationLabel.INSUFFICIENT_EVIDENCE:
        image_explanation = (
            f"A visual showing that the claim '{claim_short}' could not be verified "
            f"or denied due to insufficient available evidence. The image should "
            f"illustrate the need for additional reliable sources and data from "
            f"{primary_entity} or similar authoritative bodies."
        )
    else:  # TRUE
        image_explanation = (
            f"Official documentation, data chart, or published report from "
            f"{primary_entity} that directly supports the claim '{claim_short}'. "
            f"The image should show verifiable data, dates, and methodology that "
            f"corroborate the specific assertions made in the claim."
        )

    # ── Video Search Queries (context-aware) ─────────────────────
    queries: List[str] = []

    if label == ClassificationLabel.FALSE:
        query_templates = [
            f"{entity_str} fact check debunked",
            f"{topic} myth vs reality explained",
            f"misinformation about {topic} official response",
        ]
    elif label == ClassificationLabel.MISLEADING:
        query_templates = [
            f"{entity_str} full context explained",
            f"{topic} what the evidence really shows",
            f"{topic} misleading claims analysis",
        ]
    elif label == ClassificationLabel.INSUFFICIENT_EVIDENCE:
        query_templates = [
            f"{entity_str} fact check evidence",
            f"{topic} verification sources",
            f"{topic} claims and evidence review",
        ]
    else:  # TRUE
        query_templates = [
            f"{entity_str} official report verification",
            f"{topic} evidence and sources",
            f"{topic} latest research findings",
        ]

    queries = [q.strip() for q in query_templates]

    # Ensure exactly 3 queries
    while len(queries) < 3:
        queries.append(f"{topic} verified information")
    queries = queries[:3]

    logger.debug(
        "Generated evidence: entities=%s, queries=%s",
        entities[:3], [q[:40] for q in queries],
    )

    return image_explanation, queries
