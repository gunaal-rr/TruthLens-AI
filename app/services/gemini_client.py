"""
TruthLens AI — Gemini LLM Integration Client (v2)

Provides AI-powered claim analysis using Google's Gemini API.
Used as a HIGH-CONFIDENCE signal for evidence classification.

v2 Upgrades (FIX 7):
    - Enhanced reasoning prompt (scientific consensus, misinformation patterns)
    - Retry logic (max 2 retries with backoff)
    - Timeout handling (10s per request)
    - Graceful fallback to rule-based classifier on failure
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Gemini REST API endpoint
_GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# FIX 7: Enhanced system prompt with deeper reasoning
_SYSTEM_PROMPT = """You are an expert fact-checking AI with deep knowledge of science, history, and current events.

Given a CLAIM and EVIDENCE, analyze whether the evidence SUPPORTS, REFUTES, or is NEUTRAL to the claim.

ANALYSIS REQUIREMENTS:
1. Look for FACTUAL CONTRADICTIONS between claim and evidence
2. Consider SCIENTIFIC CONSENSUS - if the claim contradicts established science, evidence debunking it should be classified as "refute"
3. Detect MISINFORMATION PATTERNS - conspiracy theories, pseudoscience, debunked claims
4. Distinguish between "evidence about the same topic" (neutral) vs "evidence that confirms/denies the claim" (support/refute)
5. Pay attention to NEGATION words: "not", "no evidence", "debunked", "false", "myth", "disproven"

CLASSIFICATION RULES:
- "support" = evidence directly confirms the claim is factually correct
- "refute" = evidence contradicts, debunks, disproves, or shows the claim is false
- "neutral" = evidence is related but does not clearly confirm or deny

You MUST return ONLY valid JSON in this exact format:
{"label": "support|refute|neutral", "confidence": 0.0-1.0, "reasoning": "brief explanation of your analysis"}

Do NOT return anything outside the JSON object."""


async def analyze_claim_with_gemini(
    claim: str,
    evidence: str,
    max_retries: int = 2,
) -> Optional[dict]:
    """
    Use Gemini to classify whether evidence supports or refutes a claim.

    Includes retry logic with exponential backoff (FIX 7).

    Args:
        claim: The claim being fact-checked.
        evidence: The evidence text to analyze against the claim.
        max_retries: Maximum retry attempts (default 2).

    Returns:
        Dict with keys: label ("support"|"refute"|"neutral"),
        confidence (float), reasoning (str).
        Returns None on any failure (graceful fallback to rule-based).
    """
    settings = get_settings()

    if not settings.gemini_api_key:
        logger.debug("Gemini API key not configured -- skipping LLM analysis")
        return None

    # Truncate evidence to avoid token limits
    evidence_truncated = evidence[:4000] if len(evidence) > 4000 else evidence

    user_prompt = (
        f"CLAIM: {claim}\n\n"
        f"EVIDENCE: {evidence_truncated}\n\n"
        f"Analyze the claim against this evidence. Determine if the evidence "
        f"SUPPORTS, REFUTES, or is NEUTRAL to the claim. Consider factual "
        f"contradiction, scientific consensus, and misinformation patterns. "
        f"Return JSON only."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{_SYSTEM_PROMPT}\n\n{user_prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 300,
            "topP": 0.8,
        },
    }

    params = {"key": settings.gemini_api_key}

    # -- Retry loop with backoff (FIX 7)
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _GEMINI_API_URL,
                    json=payload,
                    params=params,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini returned no candidates (attempt %d)", attempt + 1)
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return None

            text = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            if not text:
                logger.warning("Gemini returned empty text (attempt %d)", attempt + 1)
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return None

            # Parse JSON from response
            result = _parse_gemini_json(text)

            if result and "label" in result:
                label = result["label"].lower().strip()
                if label not in ("support", "refute", "neutral"):
                    logger.warning("Gemini returned invalid label: %s", label)
                    return None

                confidence = float(result.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                parsed = {
                    "label": label,
                    "confidence": confidence,
                    "reasoning": result.get("reasoning", ""),
                }

                logger.info(
                    "Gemini analysis: label=%s, confidence=%.2f, reasoning=%s",
                    parsed["label"], parsed["confidence"],
                    parsed["reasoning"][:100],
                )
                return parsed

            logger.warning("Gemini response missing 'label' key: %s", text[:200])
            return None

        except httpx.TimeoutException:
            logger.warning("Gemini API timeout (attempt %d/%d)", attempt + 1, max_retries + 1)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning(
                "Gemini API HTTP %d (attempt %d/%d): %s",
                status, attempt + 1, max_retries + 1, str(e)[:100],
            )
            # Fail fast on client errors (bad request, auth, not found)
            if status in (400, 401, 403, 404):
                logger.debug("Fatal client error HTTP %d. Skipping retries.", status)
                break
        except Exception as e:
            logger.warning(
                "Gemini analysis failed (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, str(e)[:150],
            )

        # Backoff before retry
        if attempt < max_retries:
            wait = 1.5 * (attempt + 1)
            logger.debug("Retrying Gemini in %.1fs...", wait)
            await asyncio.sleep(wait)

    logger.warning("All Gemini retries exhausted -- falling back to rule-based")
    return None


def _parse_gemini_json(text: str) -> Optional[dict]:
    """
    Parse JSON from Gemini response, handling markdown code blocks.

    Gemini sometimes wraps JSON in ```json ... ``` blocks.
    """
    text = text.strip()

    # Remove markdown code block wrappers
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse Gemini JSON: %s", text[:200])
    return None
