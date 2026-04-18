"""Detailed trace to see Gemini behavior for 5G"""
import asyncio
from app.services.evidence_fetcher import fetch_evidence

async def main():
    print('Testing 5G...')
    res = await fetch_evidence('5G towers cause COVID-19', ['5G conspiracy theories', 'COVID-19'], '5G towers cause COVID-19 fact check')
    print('CONFIRM:', res.confirm_score, 'DENY:', res.deny_score)
    for s in res.sources[:4]:
        print(f'[{s.stance}] rel={s.relevance:.3f} cred={s.credibility:.3f} {s.title[:60]}')
        print(f'          snippet: {s.snippet[:80]}')
    
    print('Testing Chandrayaan...')
    res = await fetch_evidence('India successfully landed Chandrayaan-3 on the Moon in 2023', ['Chandrayaan-3', 'India', 'Moon'], 'Chandrayaan-3 Moon landing 2023 fact check')
    print('CONFIRM:', res.confirm_score, 'DENY:', res.deny_score)
    for s in res.sources[:4]:
        print(f'[{s.stance}] rel={s.relevance:.3f} cred={s.credibility:.3f} {s.title[:60]}')
        print(f'          snippet: {s.snippet[:80]}')

asyncio.run(main())
