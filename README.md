# TruthLens AI

**Production-grade fake news detection and fact-checking REST API.**

Built with FastAPI, featuring multi-signal classification, bias detection, explainable AI, and structured JSON output.

---

## Features

- **Multi-signal classifier** — Scoring-based system using credibility indicators, misinformation patterns, and known false claim databases
- **Bias & misinformation detection** — Detects sensationalism, emotional manipulation, overgeneralization, and clickbait
- **Trust score & confidence** — Independently computed: trust measures claim reliability, confidence measures system certainty
- **Context-aware evidence** — Generates entity-specific verification queries and image explanations
- **Explainable AI (XAI)** — Transparent reasoning with keyword detection and bias scoring
- **In-memory caching** — TTL-aware LRU cache (Redis-ready interface)
- **Rate limiting** — Sliding window per-IP rate limiter (swappable backend)
- **API key security** — Optional Bearer token authentication
- **Structured logging** — JSON-formatted request/response logs with latency tracking

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "5G towers cause COVID-19 and the government is hiding it"}'
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

### `POST /analyze`

Analyze a claim or news text for misinformation.

**Request:**
```json
{
  "text": "Your claim or news text here"
}
```

**Response:** Structured JSON with label, confidence, trust score, sub-claims, explanation, reasoning, evidence, bias detection, and XAI metadata. See `app/models.py` for the complete schema.

### `GET /health`

Health check returning cache and rate limiter statistics.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_MAX_REQUESTS` | `10` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window (seconds) |
| `RATE_LIMITER_BACKEND` | `memory` | `memory` or `redis` |
| `API_KEY_ENABLED` | `false` | Enable API key auth |
| `API_KEY` | `changeme…` | API key value |
| `CACHE_TTL_SECONDS` | `300` | Cache TTL (0 = no expiry) |
| `CACHE_MAX_SIZE` | `1000` | Max cache entries |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Architecture

```
Request → Security → Logging → Rate Limiter → Cache → Pipeline → Response
                                                         │
                                    ┌────────────────────┼────────────────────┐
                                    │                    │                    │
                              Language          Claim Extraction        Classification
                              Detection         Sub-claim Split         (Scoring-based)
                                    │                    │                    │
                              Bias Detection      Trust Scoring      Evidence Generation
                                    │                    │                    │
                                    └────────────────────┼────────────────────┘
                                                         │
                                               Confidence Calibration
                                                         │
                                                 Response Assembly
```

---

## Project Structure

```
app/
├── main.py                     # FastAPI entrypoint
├── config.py                   # Settings (pydantic-settings)
├── models.py                   # Request/response schemas
├── pipeline/
│   └── pipeline_manager.py     # Orchestrator
├── services/
│   ├── claim_extractor.py      # Claim & sub-claim extraction
│   ├── classifier.py           # Scoring-based classification
│   ├── bias_detector.py        # Bias & misinformation detection
│   ├── trust_scorer.py         # Trust score computation
│   ├── evidence_generator.py   # Context-aware evidence
│   └── confidence_calibrator.py# Confidence calibration
├── middleware/
│   ├── rate_limiter/
│   │   ├── base.py             # Abstract interface
│   │   ├── in_memory.py        # In-memory implementation
│   │   └── redis_limiter.py    # Redis placeholder
│   ├── logging_middleware.py   # Structured JSON logging
│   └── security.py             # API key middleware
└── utils/
    ├── language_detector.py    # Language detection
    └── cache.py                # In-memory cache with TTL
```

---

## License

MIT
