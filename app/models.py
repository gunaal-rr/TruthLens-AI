"""
TruthLens AI — Pydantic Request/Response Models

Strict schema definitions ensuring no empty fields and fully JSON-serializable output.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class ClassificationLabel(str, Enum):
    """Possible classification labels for a claim."""
    TRUE = "Real"
    FALSE = "Fake"
    MISLEADING = "Misleading"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"


class TrustLevel(str, Enum):
    """Trust level for a fact source."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SubClaimStatus(str, Enum):
    """Status of an individual sub-claim evaluation."""
    TRUE = "Real"
    FALSE = "Fake"
    MISLEADING = "Misleading"


# ── Request ──────────────────────────────────────────────────────────────────


class NewsResult(BaseModel):
    """A single news result from external search."""
    title: str = Field(default="", description="Headline of the news article")
    source: str = Field(default="", description="Publisher or domain")
    snippet: str = Field(default="", description="Summary or excerpt")
    url: str = Field(default="", description="Full URL to the article")
    date: str = Field(default="", description="Publication date if available")


class AnalyzeRequest(BaseModel):
    """Input payload for the /analyze endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="News text, claim, or question to analyze",
    )

    # ── Optional External Evidence ────────────────────────────────
    external_evidence: Optional[str] = Field(
        default=None,
        description="Structured evidence from external APIs (Gemini, search, etc.)",
    )
    news_results: Optional[List[NewsResult]] = Field(
        default=None,
        description="News search results relevant to the claim",
    )
    wiki_summary: Optional[str] = Field(
        default=None,
        description="Wikipedia summary relevant to the claim",
    )
    ai_analysis: Optional[str] = Field(
        default=None,
        description="AI model analysis from Gemini or other LLMs",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be blank or whitespace-only")
        return v.strip()



# ── Response Sub-Models ──────────────────────────────────────────────────────


class EvaluatedSubClaim(BaseModel):
    """A sub-claim with its evaluation status."""
    claim: str = Field(..., min_length=1, description="The sub-claim text")
    status: SubClaimStatus = Field(..., description="Evaluation result")


class FactSource(BaseModel):
    """A reference source used for fact-checking."""
    title: str = Field(..., min_length=1, description="Source title")
    source: str = Field(..., min_length=1, description="Source identifier or URL")
    trust: TrustLevel = Field(..., description="Trust level of the source")


class ExternalSource(BaseModel):
    """An external source used for real-time evidence verification."""
    title: str = Field(..., min_length=1, description="Source title or headline")
    url: str = Field(default="", description="Source URL")
    credibility: float = Field(..., ge=0.0, le=1.0, description="Credibility score (0-1)")


class Evidence(BaseModel):
    """Evidence artifacts for claim verification."""
    image_explanation: str = Field(
        ..., min_length=1, description="What visual proof would validate the claim"
    )
    video_search_queries: List[str] = Field(
        ..., min_length=3, max_length=3, description="3 YouTube search queries"
    )


class XAI(BaseModel):
    """Explainable AI metadata."""
    keywords_detected: List[str] = Field(
        ..., min_length=1, description="Keywords triggering bias/misinformation flags"
    )
    lack_of_evidence: bool = Field(
        ..., description="Whether evidence is lacking for the claim"
    )
    bias_score: float = Field(
        ..., ge=0.0, le=1.0, description="Bias intensity score (0–1)"
    )


# ── Main Response ────────────────────────────────────────────────────────────


class AnalyzeResponse(BaseModel):
    """Full structured response from the analysis pipeline."""

    label: ClassificationLabel = Field(..., description="Classification label")
    confidence: int = Field(
        ..., ge=0, le=100, description="System certainty (0–100)"
    )
    language: str = Field(..., min_length=1, description="Detected language code")

    claim: str = Field(..., min_length=1, description="Extracted core claim")

    sub_claims: List[EvaluatedSubClaim] = Field(
        ..., min_length=1, description="Evaluated sub-claims"
    )

    explanation: str = Field(
        ..., min_length=50, description="Detailed explanation (3–5 lines minimum)"
    )

    reasoning: List[str] = Field(
        ..., min_length=1, description="Step-by-step reasoning points"
    )

    fact_sources: List[FactSource] = Field(
        ..., min_length=1, description="Referenced fact sources"
    )

    evidence: Evidence = Field(..., description="Evidence for verification")

    trust_score: int = Field(
        ..., ge=0, le=100, description="Claim reliability score (0–100)"
    )

    bias_detected: bool = Field(..., description="Whether bias was detected")

    xai: XAI = Field(..., description="Explainable AI metadata")

    external_sources: List[ExternalSource] = Field(
        default_factory=list,
        description="External sources from real-time evidence verification",
    )

    final_verdict: str = Field(
        ..., min_length=1, description="Concise justification"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "label": "Fake",
                "confidence": 72,
                "language": "en",
                "claim": "COVID vaccines cause infertility",
                "sub_claims": [
                    {"claim": "COVID vaccines affect fertility", "status": "Fake"}
                ],
                "explanation": (
                    "The claim that COVID vaccines cause infertility is not supported by "
                    "scientific evidence. Multiple large-scale studies have shown no link "
                    "between COVID-19 vaccination and fertility issues in either men or women. "
                    "Major health organizations including the WHO and CDC have explicitly "
                    "debunked this claim. The misinformation likely originated from a "
                    "misinterpretation of early vaccine trial data."
                ),
                "reasoning": [
                    "No peer-reviewed studies support this claim",
                    "WHO and CDC have debunked this",
                    "Known misinformation pattern detected",
                ],
                "fact_sources": [
                    {
                        "title": "WHO COVID-19 Vaccine Safety",
                        "source": "who.int",
                        "trust": "high",
                    }
                ],
                "evidence": {
                    "image_explanation": (
                        "A chart from peer-reviewed fertility studies showing no "
                        "statistical difference between vaccinated and unvaccinated groups."
                    ),
                    "video_search_queries": [
                        "WHO COVID vaccine infertility myth debunked",
                        "scientific studies vaccine fertility evidence",
                        "CDC statement vaccine reproductive health",
                    ],
                },
                "trust_score": 15,
                "bias_detected": True,
                "xai": {
                    "keywords_detected": ["cause", "infertility"],
                    "lack_of_evidence": True,
                    "bias_score": 0.7,
                },
                "final_verdict": (
                    "This claim is classified as Fake due to lack of scientific "
                    "evidence and alignment with known misinformation patterns."
                ),
            }
        }
    }


# ── Error Response ───────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standardized error response."""
    detail: str = Field(..., description="Error description")
    status_code: int = Field(..., description="HTTP status code")
