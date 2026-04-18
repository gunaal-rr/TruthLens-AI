"""
TruthLens AI — Claim Extraction Service

Extracts the core claim from raw text and splits compound claims
into individual sub-claims using NLP heuristics.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Sentence-ending punctuation pattern
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# Conjunctions / separators that indicate multiple assertions
_CONJUNCTION_PATTERN = re.compile(
    r'\b(?:and also|and\s+that|furthermore|moreover|additionally|also|'
    r'in addition|as well as|not only|but also)\b',
    re.IGNORECASE,
)

# Filler prefixes to strip when extracting the core claim
_FILLER_PREFIXES = [
    r'^(?:breaking|urgent|alert|exclusive|just in|report|sources say|'
    r'according to reports|it has been reported that|people are saying that|'
    r'some claim that|experts say)\s*[:\-–—]?\s*',
]
_FILLER_RE = [re.compile(p, re.IGNORECASE) for p in _FILLER_PREFIXES]

# Minimum sub-claim length (characters) to keep
_MIN_SUBCLAIM_LEN = 10


async def extract_claim(text: str) -> str:
    """
    Extract the core claim from raw input text.

    Strips filler prefixes, extracts the primary assertive statement,
    and normalizes whitespace.

    Args:
        text: Raw input text.

    Returns:
        The extracted core claim string.
    """
    claim = text.strip()

    # Remove common filler prefixes
    for pattern in _FILLER_RE:
        claim = pattern.sub("", claim).strip()

    # If multiline, take the first substantial line as the core claim
    lines = [ln.strip() for ln in claim.splitlines() if ln.strip()]
    if lines:
        # Use the longest line as the primary claim
        claim = max(lines, key=len)

    # Normalize whitespace
    claim = re.sub(r'\s+', ' ', claim).strip()

    if not claim:
        claim = text.strip()

    logger.debug("Extracted claim: %s", claim[:100])
    return claim


async def split_sub_claims(claim: str) -> List[str]:
    """
    Split a compound claim into individual sub-claims.

    Strategy:
        1. Split on sentence boundaries.
        2. For each sentence, further split on conjunctions if present.
        3. Filter out fragments that are too short to be meaningful.

    Args:
        claim: The core claim text.

    Returns:
        List of individual sub-claim strings (at least 1).
    """
    sub_claims: List[str] = []

    # Step 1: Split into sentences
    sentences = _SENTENCE_SPLIT.split(claim)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Step 2: Check for conjunction-based compounds
        parts = _CONJUNCTION_PATTERN.split(sentence)
        if len(parts) > 1:
            for part in parts:
                part = part.strip().strip(".,;:!?")
                if len(part) >= _MIN_SUBCLAIM_LEN:
                    sub_claims.append(part)
        else:
            if len(sentence) >= _MIN_SUBCLAIM_LEN:
                sub_claims.append(sentence.strip(".,;:!?").strip())

    # Ensure at least one sub-claim
    if not sub_claims:
        sub_claims = [claim]

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for sc in sub_claims:
        normalized = sc.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(sc)

    logger.debug("Split into %d sub-claims", len(unique))
    return unique
