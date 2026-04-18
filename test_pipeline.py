"""Quick test script for the TruthLens AI pipeline."""
import httpx
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

tests = [
    {
        "name": "Chandrayaan-3 (should be True)",
        "text": "India made history when the Chandrayaan-3 mission successfully landed on the Moon south pole region on August 23, 2023.",
        "expected": "True",
    },
    {
        "name": "5G COVID (should be False)",
        "text": "5G towers cause COVID-19 infections and the virus spreads through wireless signals.",
        "expected": "False",
    },
    {
        "name": "No Evidence (should be Insufficient Evidence)",
        "text": "John ate pizza yesterday and then went to the park.",
        "expected": "Insufficient Evidence",
    },
]

for test in tests:
    print(f"\n{'='*60}")
    print(f"TEST: {test['name']}")
    print(f"{'='*60}")

    try:
        r = httpx.post(
            f"{BASE_URL}/analyze",
            json={"text": test["text"]},
            timeout=60,
        )
        d = r.json()

        label = d.get("label", "???")
        trust = d.get("trust_score", 0)
        confidence = d.get("confidence", 0)
        sources = d.get("external_sources", [])
        verdict = d.get("final_verdict", "")

        status = "PASS" if label == test["expected"] else "FAIL"
        print(f"  Label:      {label} (expected: {test['expected']}) [{status}]")
        print(f"  Trust:      {trust}/100")
        print(f"  Confidence: {confidence}%")
        print(f"  Sources:    {len(sources)}")
        for s in sources[:3]:
            print(f"    - {s['title'][:60]} (cred: {s['credibility']})")
        print(f"  Verdict:    {verdict[:150]}...")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("Tests complete.")
