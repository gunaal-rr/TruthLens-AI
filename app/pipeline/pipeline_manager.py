"""
TruthLens AI — Pipeline Orchestrator (v2)

Production-grade fact-checking pipeline with real-time evidence fetching.

Pipeline Steps:
    1.  Input processing & validation
    2.  Language detection
    3.  Claim extraction
    4.  Sub-claim splitting
    5.  Entity extraction (NEW)
    6.  Rule-based pre-check (LOW weight — 10%)
    7.  External evidence fetch (HIGH priority — 60%) (NEW)
    8.  Source credibility scoring (NEW)
    9.  Decision engine — final label (NEW)
    10. Trust score (evidence-based formula) (UPDATED)
    11. Confidence calibration (source-quality formula) (UPDATED)
    12. Bias detection
    13. Evidence artifact generation (images/video)
    14. Explanation & reasoning generation
    15. Response assembly

Key architectural change: Evidence is fetched SERVER-SIDE, not passed
by the caller. The pipeline actively queries Wikipedia and SerpAPI.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.models import (
    AnalyzeResponse,
    ClassificationLabel,
    Evidence,
    EvaluatedSubClaim,
    ExternalSource,
    FactSource,
    NewsResult,
    SubClaimStatus,
    TrustLevel,
    XAI,
)
from app.services.bias_detector import BiasReport, detect_bias
from app.services.claim_extractor import extract_claim, split_sub_claims
from app.services.classifier import (
    ClaimScores,
    aggregate_label,
    classify_sub_claim,
    generate_reasoning,
)
from app.services.confidence_calibrator import calibrate_confidence
from app.services.decision_engine import decide
from app.services.entity_extractor import extract_entities, entities_to_search_query
from app.services.evidence_fetcher import (
    EvidenceFetchResult,
    FetchedSource,
    fetch_evidence,
)
from app.services.evidence_generator import generate_evidence
from app.services.evidence_processor import (
    ExternalEvidenceReport,
    process_external_evidence,
)
from app.services.source_ranker import compute_aggregate_credibility
from app.services.trust_scorer import compute_trust_score
from app.utils.language_detector import detect_language

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Orchestrates the complete TruthLens AI analysis pipeline.

    v2 Architecture:
        - Entity extraction for targeted evidence lookup
        - Real-time evidence fetching (Wikipedia + SerpAPI)
        - Semantic matching (TF-IDF) instead of keyword matching
        - Evidence-first decision engine (rule-based = 10% weight)
        - Evidence-based trust score and confidence formulas
    """

    async def run(
        self,
        text: str,
        external_evidence: Optional[str] = None,
        news_results: Optional[List[NewsResult]] = None,
        wiki_summary: Optional[str] = None,
        ai_analysis: Optional[str] = None,
    ) -> AnalyzeResponse:
        """
        Execute the full analysis pipeline.

        Args:
            text: Raw input text (claim, news article, or question).
            external_evidence: Optional caller-provided evidence (backward compat).
            news_results: Optional caller-provided news results (backward compat).
            wiki_summary: Optional caller-provided Wikipedia summary (backward compat).
            ai_analysis: Optional caller-provided AI analysis (backward compat).

        Returns:
            Fully validated AnalyzeResponse.

        Raises:
            ValueError: If input text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty")

        text = text.strip()

        # ── Step 1: Language Detection ───────────────────────────
        language = await detect_language(text)
        logger.info("Step 1/15: language=%s", language)

        # ── Step 2: Claim Extraction ─────────────────────────────
        claim = await extract_claim(text)
        logger.info("Step 2/15: claim extracted (%d chars)", len(claim))

        # ── Step 3: Sub-Claim Splitting ──────────────────────────
        raw_sub_claims = await split_sub_claims(claim)
        logger.info("Step 3/15: %d sub-claims", len(raw_sub_claims))

        # ── Step 4: Entity Extraction (NEW) ──────────────────────
        entities = extract_entities(claim)
        search_query = entities_to_search_query(entities, claim)
        logger.info(
            "Step 4/15: entities=%s, query='%s'",
            entities[:5], search_query[:60],
        )

        # ── Step 5: Rule-Based Pre-Check (LOW weight — 10%) ─────
        sub_claim_results: List[Tuple[str, SubClaimStatus, ClaimScores]] = []
        for sc_text in raw_sub_claims:
            status, scores = await classify_sub_claim(sc_text)
            sub_claim_results.append((sc_text, status, scores))

        statuses = [s for _, s, _ in sub_claim_results]
        rule_based_label = await aggregate_label(statuses)
        logger.info("Step 5/15: rule_based_label=%s", rule_based_label.value)

        # ── Step 6: External Evidence Fetch (HIGH priority) (NEW) ─
        evidence_result: EvidenceFetchResult = await fetch_evidence(
            claim=claim,
            entities=entities,
            search_query=search_query,
        )
        logger.info(
            "Step 6/15: evidence has_data=%s, confirm=%.3f, deny=%.3f, sources=%d",
            evidence_result.has_data,
            evidence_result.confirm_score,
            evidence_result.deny_score,
            len(evidence_result.sources),
        )

        # Merge caller-provided evidence if any (backward compatibility)
        if any([external_evidence, news_results, wiki_summary, ai_analysis]):
            caller_evidence: ExternalEvidenceReport = await process_external_evidence(
                external_evidence=external_evidence,
                news_results=news_results,
                wiki_summary=wiki_summary,
                ai_analysis=ai_analysis,
            )
            if caller_evidence.has_external_data:
                evidence_result = self._merge_evidence(evidence_result, caller_evidence)
                logger.info("Step 6.5/15: merged caller-provided evidence")

        # ── Step 7: Source Credibility ────────────────────────────
        source_credibility = compute_aggregate_credibility(
            [s.credibility for s in evidence_result.sources]
        )
        logger.info("Step 7/15: source_credibility=%.3f", source_credibility)

        # ── Step 8: Decision Engine ───────────────────────────────
        label = decide(
            confirm_score=evidence_result.confirm_score,
            deny_score=evidence_result.deny_score,
            has_evidence=evidence_result.has_data,
            rule_based_label=rule_based_label,
            source_credibility=source_credibility,
            support_count=getattr(evidence_result, 'support_count', 0),
            refute_count=getattr(evidence_result, 'refute_count', 0),
        )
        logger.info("Step 8/15: final_label=%s", label.value)

        # ── Step 8.5: Align Sub-Claim Labels with Final Label ────
        #    Sub-claims must NOT contradict the final label.
        sub_claim_results = self._align_sub_claims(label, sub_claim_results)
        logger.info("Step 8.5/15: sub-claims aligned to final_label=%s", label.value)

        # ── Step 9: Trust Score (evidence-based formula) ─────────
        rule_based_score = self._compute_rule_based_score(sub_claim_results)
        individual_credibilities = [
            s.credibility for s in evidence_result.sources
        ]
        trust_score = compute_trust_score(
            evidence_confidence=evidence_result.confidence,
            source_credibility=source_credibility,
            rule_based_score=rule_based_score,
            source_credibilities=individual_credibilities,
        )
        logger.info("Step 9/15: trust_score=%d", trust_score)

        # ── Step 10: Confidence Calibration (source-quality) ─────
        confidence = calibrate_confidence(
            num_sources=len(evidence_result.sources),
            confirm_score=evidence_result.confirm_score,
            deny_score=evidence_result.deny_score,
            avg_credibility=source_credibility,
        )
        logger.info("Step 10/15: confidence=%d", confidence)

        # ── Step 11: Bias Detection ──────────────────────────────
        all_signals = []
        for _, _, scores in sub_claim_results:
            all_signals.extend(scores.signals)
        has_source = "has_source_attribution" in all_signals

        bias_report: BiasReport = await detect_bias(
            text, has_source_attribution=has_source
        )
        logger.info(
            "Step 11/15: bias=%s, score=%.3f",
            bias_report.bias_detected, bias_report.bias_score,
        )

        # ── Step 12: Evidence Artifacts (images/video) ───────────
        image_explanation, video_queries = await generate_evidence(
            claim=claim,
            label=label,
            sub_claims=raw_sub_claims,
        )
        logger.info("Step 12/15: evidence artifacts generated")

        # ── Step 13: Reasoning ───────────────────────────────────
        reasoning = await generate_reasoning(sub_claim_results)

        # Prepend evidence findings to reasoning
        if evidence_result.has_data and evidence_result.evidence_summary:
            reasoning.insert(
                0, f"External evidence: {evidence_result.evidence_summary}"
            )
            reasoning = reasoning[:6]

        logger.info("Step 13/15: %d reasoning points", len(reasoning))

        # ── Step 14: Explanation (label-aligned) ─────────────────
        explanation = self._build_explanation(
            claim=claim,
            label=label,
            sub_claim_results=sub_claim_results,
            bias_report=bias_report,
            trust_score=trust_score,
            evidence_result=evidence_result,
        )

        # ── Guardrail: Ensure explanation matches final label ────
        explanation = self._enforce_explanation_alignment(
            explanation, label, claim
        )
        logger.info("Step 14/15: explanation generated (aligned to %s)", label.value)

        # ── Step 15: Response Assembly ───────────────────────────
        fact_sources = self._generate_fact_sources(
            claim, label, all_signals, evidence_result
        )
        final_verdict = self._generate_verdict(label, trust_score, confidence, claim)

        evaluated_sub_claims = [
            EvaluatedSubClaim(claim=sc_text, status=self._map_status_to_label(status, label))
            for sc_text, status, _ in sub_claim_results
        ]

        external_sources = [
            ExternalSource(
                title=s.title if s.title else "External Source",
                url=s.url,
                credibility=round(s.credibility, 2),
            )
            for s in evidence_result.sources[:5]
        ]

        response = AnalyzeResponse(
            label=label,
            confidence=confidence,
            language=language,
            claim=claim,
            sub_claims=evaluated_sub_claims,
            explanation=explanation,
            reasoning=reasoning,
            fact_sources=fact_sources,
            evidence=Evidence(
                image_explanation=image_explanation,
                video_search_queries=video_queries,
            ),
            trust_score=trust_score,
            bias_detected=bias_report.bias_detected,
            xai=XAI(
                keywords_detected=bias_report.keywords_detected,
                lack_of_evidence=bias_report.lack_of_evidence,
                bias_score=round(bias_report.bias_score, 3),
            ),
            external_sources=external_sources,
            final_verdict=final_verdict,
        )

        logger.info(
            "Pipeline complete: label=%s, confidence=%d, trust=%d, sources=%d",
            label.value, confidence, trust_score, len(external_sources),
        )

        return response

    # ── Private Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_rule_based_score(
        sub_claim_results: List[Tuple[str, SubClaimStatus, ClaimScores]],
    ) -> float:
        """
        Compute normalized rule-based score (0–1).

        Used as the 10%-weight input to the trust score formula.
        """
        if not sub_claim_results:
            return 0.5

        total_real = sum(sc.real_score for _, _, sc in sub_claim_results)
        total_fake = sum(sc.fake_score for _, _, sc in sub_claim_results)
        total = total_real + total_fake

        if total == 0:
            return 0.5

        return total_real / total

    @staticmethod
    def _align_sub_claims(
        final_label: ClassificationLabel,
        sub_claim_results: List[Tuple[str, SubClaimStatus, ClaimScores]],
    ) -> List[Tuple[str, SubClaimStatus, ClaimScores]]:
        """
        Align sub-claim statuses with the final decision label.

        Prevents logical contradictions where sub-claims say "Misleading"
        but the overall label says "True".

        Rules:
            - TRUE overall  → sub-claims become TRUE (unless strongly false)
            - FALSE overall → sub-claims become FALSE (unless strongly true)
            - MISLEADING    → keep original (mixed is expected)
            - INSUFFICIENT  → keep original
        """
        if final_label == ClassificationLabel.MISLEADING:
            return sub_claim_results
        if final_label == ClassificationLabel.INSUFFICIENT_EVIDENCE:
            return sub_claim_results

        aligned: List[Tuple[str, SubClaimStatus, ClaimScores]] = []
        for sc_text, status, scores in sub_claim_results:
            if final_label == ClassificationLabel.TRUE:
                # Only override if both confirm and deny are NOT both significant
                if not (scores.real_score > 2.0 and scores.fake_score > 2.0):
                    status = SubClaimStatus.TRUE
            elif final_label == ClassificationLabel.FALSE:
                if not (scores.real_score > 2.0 and scores.fake_score > 2.0):
                    status = SubClaimStatus.FALSE
            aligned.append((sc_text, status, scores))

        return aligned

    @staticmethod
    def _enforce_explanation_alignment(
        explanation: str,
        label: ClassificationLabel,
        claim: str,
    ) -> str:
        """
        Guardrail: Ensure explanation text does not contradict the final label.

        Catches cases where rule-based sub-claim language leaks
        contradictory wording into the explanation.
        """
        claim_preview = claim[:80] + ("..." if len(claim) > 80 else "")

        # Map label → allowed explanation type
        contradictions = {
            ClassificationLabel.TRUE: ["classified as Fake", "classified as Misleading"],
            ClassificationLabel.FALSE: ["classified as Real", "classified as Misleading"],
            ClassificationLabel.MISLEADING: ["classified as Real", "classified as Fake"],
        }

        bad_phrases = contradictions.get(label, [])
        has_contradiction = any(phrase in explanation for phrase in bad_phrases)

        if has_contradiction:
            logger.warning(
                "Explanation contradicts label %s — regenerating opening",
                label.value,
            )
            # Replace the contradictory opening with label-correct one
            label_openings = {
                ClassificationLabel.TRUE: (
                    f'The claim "{claim_preview}" has been classified as Real based on '
                    f"corroborating evidence from external sources."
                ),
                ClassificationLabel.FALSE: (
                    f'The claim "{claim_preview}" has been classified as Fake based on '
                    f"contradicting evidence from credible sources."
                ),
                ClassificationLabel.MISLEADING: (
                    f'The claim "{claim_preview}" has been classified as Misleading because '
                    f"evidence provides mixed signals about its accuracy."
                ),
                ClassificationLabel.INSUFFICIENT_EVIDENCE: (
                    f'The claim "{claim_preview}" has been classified as Insufficient Evidence '
                    f"because no reliable external sources could verify or deny it."
                ),
            }

            # Find the first sentence boundary after the opening and replace
            sentences = explanation.split(". ")
            if sentences:
                sentences[0] = label_openings.get(label, sentences[0])
            explanation = ". ".join(sentences)

        return explanation

    @staticmethod
    def _map_status_to_label(
        status: SubClaimStatus,
        final_label: ClassificationLabel,
    ) -> SubClaimStatus:
        """
        Final mapping of sub-claim status for response serialization.

        Ensures the serialized sub-claims are consistent with the overall label.
        """
        if final_label == ClassificationLabel.TRUE:
            if status == SubClaimStatus.MISLEADING:
                return SubClaimStatus.TRUE
        elif final_label == ClassificationLabel.FALSE:
            if status == SubClaimStatus.MISLEADING:
                return SubClaimStatus.FALSE
        return status

    @staticmethod
    def _merge_evidence(
        fetched: EvidenceFetchResult,
        caller: ExternalEvidenceReport,
    ) -> EvidenceFetchResult:
        """
        Merge caller-provided evidence into fetched evidence.

        Fetched evidence is PRIMARY (70% weight), caller is SECONDARY (30%).
        """
        # Add caller's extracted sources
        for src in caller.extracted_sources[:3]:
            fetched.sources.append(FetchedSource(
                title=src.title,
                url=src.source,
                snippet=caller.evidence_summary[:200] if caller.evidence_summary else "",
                source_type="caller_provided",
                credibility=0.70,
            ))

        # Blend scores
        if caller.has_external_data:
            fetched.has_data = True
            w_fetch = 0.7
            w_call = 0.3
            fetched.confirm_score = (
                fetched.confirm_score * w_fetch + caller.confirmation_score * w_call
            )
            fetched.deny_score = (
                fetched.deny_score * w_fetch + caller.denial_score * w_call
            )
            # Re-normalize
            total = fetched.confirm_score + fetched.deny_score
            if total > 0:
                fetched.confirm_score = round(fetched.confirm_score / total, 3)
                fetched.deny_score = round(fetched.deny_score / total, 3)

        return fetched

    def _build_explanation(
        self,
        claim: str,
        label: ClassificationLabel,
        sub_claim_results: List[Tuple[str, SubClaimStatus, ClaimScores]],
        bias_report: BiasReport,
        trust_score: int,
        evidence_result: Optional[EvidenceFetchResult] = None,
    ) -> str:
        """Build a detailed, minimum 3–5 line explanation."""
        parts: List[str] = []

        total = len(sub_claim_results)
        true_count = sum(1 for _, s, _ in sub_claim_results if s == SubClaimStatus.TRUE)
        false_count = sum(1 for _, s, _ in sub_claim_results if s == SubClaimStatus.FALSE)
        misleading_count = total - true_count - false_count
        claim_preview = claim[:80] + ("..." if len(claim) > 80 else "")

        # Opening based on label
        label_explanations = {
            ClassificationLabel.TRUE: (
                f'The claim "{claim_preview}" has been classified as Real based on '
                f"corroborating evidence from external sources."
            ),
            ClassificationLabel.FALSE: (
                f'The claim "{claim_preview}" has been classified as Fake based on '
                f"contradicting evidence from credible sources."
            ),
            ClassificationLabel.MISLEADING: (
                f'The claim "{claim_preview}" has been classified as Misleading because '
                f"evidence provides mixed signals about its accuracy."
            ),
            ClassificationLabel.INSUFFICIENT_EVIDENCE: (
                f'The claim "{claim_preview}" has been classified as Insufficient Evidence '
                f"because no reliable external sources could verify or deny it."
            ),
        }
        parts.append(
            label_explanations.get(label, label_explanations[ClassificationLabel.MISLEADING])
        )

        # External evidence summary
        if evidence_result and evidence_result.has_data:
            parts.append(evidence_result.evidence_summary)
        else:
            parts.append(
                "No external evidence was retrieved for this claim. The classification "
                "relies primarily on textual analysis."
            )

        # Sub-claim breakdown
        parts.append(
            f"The analysis evaluated {total} sub-claim(s): {true_count} showed alignment "
            f"with available evidence, {false_count} were contradicted, and "
            f"{misleading_count} could not be conclusively verified."
        )

        # Bias assessment
        if bias_report.bias_detected:
            kw = ", ".join(bias_report.keywords_detected[:5])
            parts.append(
                f"Bias indicators detected (score: {bias_report.bias_score:.2f}), "
                f"including: {kw}."
            )
        else:
            parts.append("No significant bias indicators were detected in the text.")

        # Trust score context
        if trust_score >= 80:
            parts.append(
                f"Trust score of {trust_score}/100 indicates high reliability, "
                f"backed by strong evidence from credible sources."
            )
        elif trust_score >= 50:
            parts.append(
                f"Trust score of {trust_score}/100 indicates partial reliability. "
                f"Some aspects are supported but others need verification."
            )
        else:
            parts.append(
                f"Trust score of {trust_score}/100 indicates low reliability. "
                f"Insufficient evidence or contradicting signals found."
            )

        # Evidence note
        if bias_report.lack_of_evidence:
            parts.append(
                "The text does not cite specific sources or verifiable evidence, "
                "which significantly reduces confidence in the claim's accuracy."
            )

        return " ".join(parts)

    def _generate_fact_sources(
        self,
        claim: str,
        label: ClassificationLabel,
        signals: List[str],
        evidence_result: Optional[EvidenceFetchResult] = None,
    ) -> List[FactSource]:
        """
        Generate fact sources, prioritizing REAL external sources.

        Priority:
            1. Sources from real-time evidence fetch (real data)
            2. Domain-specific authoritative sources (heuristic fallback)
            3. Generic fallback
        """
        sources: List[FactSource] = []

        # ── Priority 1: Real external sources ────────────────────
        if evidence_result and evidence_result.sources:
            for s in evidence_result.sources[:4]:
                if s.credibility >= 0.80:
                    trust = TrustLevel.HIGH
                elif s.credibility >= 0.60:
                    trust = TrustLevel.MEDIUM
                else:
                    trust = TrustLevel.LOW

                title = s.title[:100] if s.title else "External Source"
                source_id = s.url or s.source_type
                if not any(fs.source == source_id for fs in sources):
                    sources.append(FactSource(
                        title=title,
                        source=source_id,
                        trust=trust,
                    ))

        # ── Priority 2: Domain-specific fallback ─────────────────
        if len(sources) < 2:
            claim_lower = claim.lower()
            domain_sources: Dict[str, List[FactSource]] = {
                "medical": [
                    FactSource(title="World Health Organization (WHO)", source="who.int", trust=TrustLevel.HIGH),
                    FactSource(title="Centers for Disease Control and Prevention", source="cdc.gov", trust=TrustLevel.HIGH),
                ],
                "space": [
                    FactSource(title="NASA", source="nasa.gov", trust=TrustLevel.HIGH),
                    FactSource(title="Indian Space Research Organisation", source="isro.gov.in", trust=TrustLevel.HIGH),
                ],
                "tech": [
                    FactSource(title="IEEE Standards Association", source="ieee.org", trust=TrustLevel.HIGH),
                ],
                "climate": [
                    FactSource(title="Intergovernmental Panel on Climate Change", source="ipcc.ch", trust=TrustLevel.HIGH),
                ],
            }

            kw_map = {
                "medical": ["vaccine", "covid", "health", "disease", "virus"],
                "space": ["nasa", "isro", "space", "moon", "mars", "chandrayaan"],
                "tech": ["5g", "ai", "blockchain", "crypto"],
                "climate": ["climate", "global warming", "carbon"],
            }

            for domain, keywords in kw_map.items():
                if any(kw in claim_lower for kw in keywords):
                    for fs in domain_sources.get(domain, []):
                        if len(sources) < 3 and not any(s.source == fs.source for s in sources):
                            sources.append(fs)

        # ── Fallback ─────────────────────────────────────────────
        if not sources:
            sources.append(FactSource(
                title="No authoritative source identified",
                source="internal analysis",
                trust=TrustLevel.LOW,
            ))

        return sources[:5]

    def _generate_verdict(
        self,
        label: ClassificationLabel,
        trust_score: int,
        confidence: int,
        claim: str,
    ) -> str:
        """Generate a concise final verdict."""
        claim_preview = claim[:60] + ("..." if len(claim) > 60 else "")

        verdicts = {
            ClassificationLabel.TRUE: (
                f'The claim "{claim_preview}" is classified as REAL with a trust score '
                f"of {trust_score}/100 and confidence of {confidence}%. Evidence from "
                f"credible sources corroborates this claim."
            ),
            ClassificationLabel.FALSE: (
                f'The claim "{claim_preview}" is classified as FAKE with a trust score '
                f"of {trust_score}/100 and confidence of {confidence}%. Credible sources "
                f"contradict this claim."
            ),
            ClassificationLabel.MISLEADING: (
                f'The claim "{claim_preview}" is classified as MISLEADING with a trust '
                f"score of {trust_score}/100 and confidence of {confidence}%. Evidence "
                f"provides mixed signals."
            ),
            ClassificationLabel.INSUFFICIENT_EVIDENCE: (
                f'The claim "{claim_preview}" has INSUFFICIENT EVIDENCE. Trust score: '
                f"{trust_score}/100, confidence: {confidence}%. No reliable sources "
                f"could verify or deny this claim."
            ),
        }

        return verdicts.get(label, verdicts[ClassificationLabel.MISLEADING])
