"""Quick re-test: false claims after cache clear."""
import asyncio
import httpx

BASE = "http://localhost:8000"

async def main():
    # Clear internal cache by importing and resetting
    from app.services.evidence_fetcher import _evidence_cache
    _evidence_cache.clear()
    print("Cache cleared.")

    tests = [
        ("The earth is flat and NASA is hiding the truth", ["Fake", "Misleading"]),
        ("5G towers cause COVID-19 and the government is covering it up", ["Fake", "Misleading"]),
        ("Water boils at 100 degrees Celsius at sea level", ["Real", "Misleading"]),
        ("India successfully landed Chandrayaan-3 on the Moon in 2023", ["Real", "Misleading"]),
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        for text, expected in tests:
            r = await client.post(f"{BASE}/analyze", json={"text": text})
            if r.status_code != 200:
                print(f"[ERROR] HTTP {r.status_code}: {text[:40]}")
                continue
            d = r.json()
            label = d["label"]
            ok = label in expected
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] '{text[:45]}...'")
            print(f"       label={label}, confidence={d['confidence']}, trust={d['trust_score']}")
            srcs = d.get("external_sources", [])
            for s in srcs[:3]:
                print(f"       src: [{s.get('credibility')}] {s.get('title', '?')[:55]}")

asyncio.run(main())
