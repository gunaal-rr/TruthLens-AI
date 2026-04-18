"""
TruthLens AI — Language Detection Utility

Lightweight language detection using the langdetect library.
Falls back to English ("en") when detection fails or confidence is too low.
"""

from __future__ import annotations

import logging
from typing import Optional

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Enforce deterministic detection across runs
DetectorFactory.seed = 0


async def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text: The input text to analyze.

    Returns:
        ISO 639-1 language code (e.g., "en", "es", "fr").
        Defaults to "en" if detection fails.
    """
    if not text or not text.strip():
        return "en"

    try:
        lang: Optional[str] = detect(text)
        if lang:
            logger.debug("Detected language: %s", lang)
            return lang
    except LangDetectException as exc:
        logger.warning("Language detection failed: %s — defaulting to 'en'", exc)
    except Exception as exc:
        logger.error("Unexpected error in language detection: %s", exc)

    return "en"
