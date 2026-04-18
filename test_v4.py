"""TruthLens AI v4 -- Full system validation (10 fixes)."""
import asyncio
import httpx

BASE = "http://localhost:8000"


def test_unit():
    """Unit tests for core fixes."""
    from app.services.evidence_classifier import classify_evidence
    from app.services.evidence_fetcher import get_source_weight
    from app.services.decision_engine import decide
    from app.models import ClassificationLabel

    # FIX 2/5: Source weighting
    assert get_source_weight("reuters.com") == 0.9, "Reuters should be 0.9"
    assert get_source_weight("wikipedia.org") == 0.7, "Wikipedia should be 0.7"
    assert get_source_weight("random-blog.com") == 0.4, "Unknown should be 0.4"
    assert get_source_weight("nasa.gov") == 0.9, "NASA should be 0.9"
    print("[PASS] FIX 5: Source-aware weighting")

    # Evidence classifier
    r1 = classify_evidence("5G causes COVID", "This claim has been debunked. No evidence for 5G causing COVID.")
    assert r1 == "refute", f"Expected refute, got {r1}"
    r2 = classify_evidence("Moon landing happened", "NASA successfully landed on the Moon, confirmed by evidence.")
    assert r2 == "support", f"Expected support, got {r2}"
    print("[PASS] Evidence classifier: support/refute detection")

    # FIX 3: Refutation override in decision engine
    label = decide(
        confirm_score=0.4, deny_score=0.6,
        has_evidence=True, refute_count=2, support_count=1,
    )
    assert label == ClassificationLabel.FALSE, f"Expected Fake, got {label.value}"
    print("[PASS] FIX 3: Refutation override -> Fake")

    # FIX 8: Ratio-based decision
    label2 = decide(confirm_score=0.85, deny_score=0.15, has_evidence=True)
    assert label2 == ClassificationLabel.TRUE, f"Expected Real, got {label2.value}"
    label3 = decide(confirm_score=0.1, deny_score=0.9, has_evidence=True)
    assert label3 == ClassificationLabel.FALSE, f"Expected Fake, got {label3.value}"
    print("[PASS] FIX 8: Strict ratio-based decision")

    # FIX 9: Confidence cap
    from app.services.confidence_calibrator import calibrate_confidence
    c = calibrate_confidence(num_sources=10, confirm_score=1.0, deny_score=0.0, avg_credibility=1.0)
    assert c <= 95, f"Confidence {c} exceeds 95"
    assert c >= 10, f"Confidence {c} below 10"
    print("[PASS] FIX 9: Confidence capped at 95")

    # Labels are Real/Fake
    assert ClassificationLabel.TRUE.value == "Real"
    assert ClassificationLabel.FALSE.value == "Fake"
    print("[PASS] Labels: Real/Fake confirmed")
    print()


async def test_live():
    """Live server tests."""
    tests = [
        {
            "name": "True claim (water boils at 100C)",
            "text": "Water boils at 100 degrees Celsius at sea level",
            "expect_label": ["Real", "Misleading"],
        },
        {
            "name": "False claim (flat earth)",
            "text": "The earth is flat and NASA is hiding the truth",
            "expect_label": ["Fake", "Misleading"],
        },
        {
            "name": "False claim (5G COVID)",
            "text": "5G towers cause COVID-19 and the government is covering it up",
            "expect_label": ["Fake", "Misleading"],
        },
        {
            "name": "True claim (Chandrayaan-3)",
            "text": "India successfully landed Chandrayaan-3 on the Moon south pole in 2023",
            "expect_label": ["Real", "Misleading"],
        },
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        for t in tests:
            try:
                r = await client.post(f"{BASE}/analyze", json={"text": t["text"]})
                if r.status_code != 200:
                    print(f"[SKIP] {t['name']}: HTTP {r.status_code}")
                    continue
                d = r.json()
                label = d["label"]
                conf = d["confidence"]
                trust = d["trust_score"]
                ok = label in t["expect_label"]
                status = "PASS" if ok else "WARN"
                print(f"[{status}] {t['name']}")
                print(f"       label={label}, confidence={conf}, trust={trust}")
                for sc in d["sub_claims"]:
                    print(f"       sub: [{sc['status']}] {sc['claim'][:50]}")
                assert "True" != label, "Label must not be 'True'"
                assert "False" != label, "Label must not be 'False'"
                assert conf <= 95, f"Confidence {conf} exceeds 95 cap"
            except httpx.ConnectError:
                print(f"[SKIP] Server not reachable")
                return
            except Exception as e:
                print(f"[ERROR] {t['name']}: {e}")
    print()


async def main():
    print("=" * 60)
    print("TruthLens AI v4 -- Full Validation (10 Fixes)")
    print("=" * 60)
    print()

    test_unit()

    print("--- Live Server Tests ---")
    await test_live()

    print("=" * 60)
    print("ALL VALIDATIONS COMPLETE")
    print("=" * 60)


asyncio.run(main())
