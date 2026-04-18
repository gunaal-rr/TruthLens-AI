"""Quick test: Wikipedia full content API + evidence scoring."""
import asyncio
import httpx


async def test_wikipedia():
    headers = {
        "User-Agent": "TruthLensAI/2.0 (truthlens@example.com)",
        "Accept": "application/json",
    }
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "true",
        "titles": "Flat Earth",
        "format": "json",
        "exlimit": "1",
        "exintro": "false",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params=params, headers=headers, timeout=10,
        )
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            ext = p.get("extract", "")
            title = p.get("title", "N/A")
            print(f"Page ID: {pid}")
            print(f"Title: {title}")
            print(f"Extract length: {len(ext)} chars")
            print(f"First 300 chars: {ext[:300]}")
            print()

    # Test evidence fetch pipeline
    from app.services.evidence_fetcher import fetch_evidence
    print("--- Testing fetch_evidence for 'The earth is flat' ---")
    result = await fetch_evidence(
        claim="The earth is flat",
        entities=["Flat Earth", "Earth", "NASA"],
        search_query="flat earth fact check",
    )
    print(f"has_data: {result.has_data}")
    print(f"confirm: {result.confirm_score}, deny: {result.deny_score}")
    print(f"support_count: {result.support_count}, refute_count: {result.refute_count}")
    print(f"sources: {len(result.sources)}")
    for s in result.sources[:5]:
        print(f"  [{s.stance}] cred={s.credibility:.2f} rel={s.relevance:.3f} {s.title[:60]}")
    print(f"summary: {result.evidence_summary[:150]}")


asyncio.run(test_wikipedia())
