"""
TruthLens AI — Entity Extraction Service

Extracts key entities (organizations, mission names, proper nouns, dates)
from claim text using pattern-based noun phrase extraction.

Used to build targeted search queries for evidence fetching.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


# ── Extraction Patterns ──────────────────────────────────────────────────────

# Known organizations / agencies
_ORG_RE = re.compile(
    r'\b(?:NASA|ISRO|WHO|CDC|FDA|NIH|UN|UNESCO|UNICEF|EU|NATO|'
    r'BBC|Reuters|AP|CNN|FBI|CIA|NSA|IMF|WTO|IAEA|ESA|JAXA|'
    r'World Health Organization|United Nations|European Union|'
    r'Centers for Disease Control|Indian Space Research Organisation|'
    r'National Institutes of Health|Food and Drug Administration|'
    r'European Space Agency|National Aeronautics and Space Administration)\b',
    re.IGNORECASE,
)

# Mission / project names (e.g., Chandrayaan-3, Apollo-11, Sputnik-1)
_MISSION_RE = re.compile(r'\b([A-Z][a-zA-Z]+[-‑]\d+[a-zA-Z]?)\b')

# Multi-word proper nouns (e.g., "South Pole", "New Delhi")
_MULTI_PROPER_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')

# Single proper nouns (capitalized words, 3+ chars)
_SINGLE_PROPER_RE = re.compile(r'\b([A-Z][a-z]{2,})\b')

# Full date patterns (e.g., "August 23, 2023")
_DATE_FULL_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2},?\s+\d{4}\b'
)

# Common English words that look like proper nouns but aren't entities
_STOP_ENTITIES = {
    "the", "this", "that", "these", "those", "what", "which", "where",
    "when", "how", "why", "who", "whom", "whose", "has", "have", "had",
    "was", "were", "been", "being", "are", "not", "but", "and", "for",
    "with", "from", "into", "about", "also", "very", "just", "made",
    "history", "successfully", "region", "according", "however",
    "moreover", "furthermore", "therefore", "meanwhile", "although",
    "because", "since", "after", "before", "during", "while",
    "said", "says", "told", "reported", "claimed", "stated",
}


# ── Public API ───────────────────────────────────────────────────────────────


def extract_entities(text: str) -> List[str]:
    """
    Extract key entities from text for evidence lookup.

    Priority order:
        0. Known claim concepts (mapped to Wikipedia article names)
        1. Known organizations / agencies
        2. Mission / project names (hyphenated + digit)
        3. Dates
        4. Multi-word proper nouns
        5. Single proper nouns
        6. Key claim phrases (fallback)

    Returns:
        Deduplicated list of entity strings (max 10).
    """
    entities: List[str] = []
    text_lower = text.lower()

    # Priority 0: Known claim concepts -> Wikipedia article names
    # Maps common misinformation topics and scientific claims to their
    # Wikipedia article for targeted evidence lookup
    _concept_map = {
        "flat earth": "Flat Earth",
        "earth is flat": "Flat Earth",
        "5g": "5G conspiracy theories",
        "5g covid": "5G conspiracy theories",
        "5g tower": "5G conspiracy theories",
        "vaccine cause": "Vaccine misinformation",
        "vaccines cause": "Vaccine misinformation",
        "anti-vax": "Anti-vaccine activism",
        "climate change hoax": "Climate change denial",
        "climate change fake": "Climate change denial",
        "global warming hoax": "Climate change denial",
        "moon landing fake": "Moon landing conspiracy theories",
        "moon landing hoax": "Moon landing conspiracy theories",
        "never landed on the moon": "Moon landing conspiracy theories",
        "chemtrail": "Chemtrail conspiracy theory",
        "reptilian": "Reptilian conspiracy theory",
        "illuminati": "Illuminati",
        "hollow earth": "Hollow Earth",
        "water memory": "Water memory",
        "homeopathy": "Homeopathy",
        "autism vaccine": "Vaccines and autism",
        "chandrayaan": "Chandrayaan-3",
    }
    for pattern, wiki_name in _concept_map.items():
        if pattern in text_lower:
            entities.append(wiki_name)

    # Priority 1: Known organizations
    for match in _ORG_RE.finditer(text):
        entities.append(match.group().strip())

    # Priority 2: Mission / project names
    for match in _MISSION_RE.finditer(text):
        entities.append(match.group().strip())

    # Priority 3: Dates
    for match in _DATE_FULL_RE.finditer(text):
        entities.append(match.group().strip())

    # Priority 4: Multi-word proper nouns
    for match in _MULTI_PROPER_RE.finditer(text):
        name = match.group().strip()
        first_word = name.split()[0].lower()
        if first_word not in _STOP_ENTITIES:
            entities.append(name)

    # Priority 5: Single proper nouns
    for match in _SINGLE_PROPER_RE.finditer(text):
        word = match.group().strip()
        if word.lower() not in _STOP_ENTITIES and len(word) >= 3:
            entities.append(word)

    # Priority 6 (fallback): Extract key noun-like words from claim
    # This ensures we always have something to search Wikipedia for
    if not entities:
        stop = _STOP_ENTITIES | {
            "is", "are", "was", "were", "the", "a", "an", "it", "its",
            "and", "or", "but", "of", "to", "in", "on", "at", "by",
        }
        for w in text.split():
            clean = w.strip(".,!?\"'()")
            if clean.lower() not in stop and len(clean) > 3:
                entities.append(clean.capitalize())
                if len(entities) >= 3:
                    break

    # Deduplicate preserving priority order
    seen: set = set()
    unique: List[str] = []
    for e in entities:
        lower = e.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(e)

    logger.debug("Extracted %d entities: %s", len(unique), unique[:5])
    return unique[:10]


def entities_to_search_query(entities: List[str], claim: str) -> str:
    """
    Build a fact-check-optimized search query from entities and claim text.

    Strategy:
        1. Extract meaningful keywords from the claim itself (PRIORITY)
        2. Supplement with entities
        3. Append "fact check" to guide results toward verification articles
    """
    # Step 1: Extract meaningful keywords from claim
    stop = _STOP_ENTITIES | {
        "is", "are", "was", "were", "the", "a", "an", "it", "its",
        "and", "or", "but", "of", "to", "in", "on", "at", "by",
        "truth", "really", "actually", "hiding",
    }
    claim_words = [
        w for w in claim.split()
        if w.lower().strip(".,!?\"'()") not in stop and len(w) > 2
    ]

    # Step 2: Combine claim keywords + entities (deduped)
    seen = set()
    query_parts: List[str] = []
    for w in claim_words[:6]:
        wl = w.lower().strip(".,!?\"'()")
        if wl not in seen:
            seen.add(wl)
            query_parts.append(w.strip(".,!?\"'()"))
    for e in entities[:3]:
        el = e.lower()
        if el not in seen:
            seen.add(el)
            query_parts.append(e)

    if not query_parts:
        query_parts = claim.split()[:6]

    # Step 3: Append "fact check" to guide toward verification articles
    query = " ".join(query_parts[:8])
    return f"{query} fact check"

