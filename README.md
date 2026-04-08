# Milestone 5 — LLM Inference Optimization with Batching and Caching

**Course:** IDS 568 MLOps | Module 6  
**NetID:** [your_netid]

---

## Overview

This project implements a production-ready LLM inference API that demonstrates two
fundamental serving optimizations:

- **Dynamic request batching** — groups concurrent requests into GPU-efficient batches,
  amortizing the cost of weight loading across multiple requests.
- **Privacy-safe response caching** — eliminates redundant computation for repeated
  prompts using hashed keys and TTL-based expiration.

---

## Repository Structure

```
ids568-milestone5-[netid]/
├── src/
│   ├── server.py        # FastAPI inference server (main entrypoint)
│   ├── batching.py      # Dynamic batching logic (asyncio)
│   ├── caching.py       # In-process and Redis cache implementation
│   └── config.py        # Pydantic-settings configuration
├── benchmarks/
│   ├── run_benchmarks.py   # Orchestrates all benchmark experiments
│   ├── load_generator.py   # Synthetic load generation
│   └── results/            # Raw benchmark JSON output
├── analysis/
│   ├── performance_report.pdf
│   ├── governance_memo.pdf
│   └── visualizations/     # PNG charts
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU for real model inference
- (Optional) Redis for production caching

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure (optional — all have sensible defaults)

Settings are read from environment variables prefixed `LLM_`:

```bash
export LLM_MODEL_NAME="sshleifer/tiny-gpt2"   # Swap for any HuggingFace model
export LLM_MAX_BATCH_SIZE=8
export LLM_BATCH_TIMEOUT_MS=50
export LLM_CACHE_BACKEND=memory               # or "redis"
export LLM_CACHE_TTL_SECONDS=3600
export LLM_CACHE_MAX_ENTRIES=1000
```

Alternatively, create a `.env` file at the project root:

```
LLM_MODEL_NAME=sshleifer/tiny-gpt2
LLM_MAX_BATCH_SIZE=8
LLM_BATCH_TIMEOUT_MS=50
```

### 4. Start the server

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 5. Test a request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 100}'
```

---

## API Reference

### `POST /generate`

Generate text for a prompt.

**Request body:**

| Field          | Type    | Default | Description                                  |
|----------------|---------|---------|----------------------------------------------|
| `prompt`       | string  | —       | Input text (required)                        |
| `max_tokens`   | int     | 100     | Max output tokens (1–512)                    |
| `temperature`  | float   | 0.0     | Sampling temp; 0 = greedy & cache-eligible   |
| `bypass_cache` | bool    | false   | Force fresh inference, skip cache            |

**Response:**

```json
{
  "generated_text": "Machine learning is...",
  "cached": false,
  "latency_ms": 312.4,
  "model": "sshleifer/tiny-gpt2"
}
```

### `GET /health`

Returns server status, model name, and device.

### `GET /metrics`

Returns live batching and caching statistics:

```json
{
  "batching": {
    "total_requests": 142,
    "total_batches": 28,
    "avg_batch_size": 5.07,
    "pending_requests": 0
  },
  "caching": {
    "hits": 56,
    "misses": 86,
    "hit_rate": 0.394,
    "size": 86
  }
}
```

### `POST /cache/clear`

Flush the entire cache (admin use).

---

## Running Benchmarks

### Full suite (runs all 5 experiments + generates charts)

```bash
python benchmarks/run_benchmarks.py --target http://localhost:8000
```

### Quick mode (shorter durations, for testing)

```bash
python benchmarks/run_benchmarks.py --target http://localhost:8000 --quick
```

### Individual load test

```bash
python benchmarks/load_generator.py \
  --target http://localhost:8000 \
  --rate 10 \
  --duration 30 \
  --repeat-ratio 0.4 \
  --output benchmarks/results/custom_run.json
```

**Load levels tested:**

| Level  | Rate (req/s) | Duration |
|--------|-------------|----------|
| Low    | 2           | 20 s     |
| Medium | 5           | 20 s     |
| High   | 10          | 20 s     |

---

## Benchmark Experiments

| # | Experiment                   | What It Measures                                   |
|---|------------------------------|----------------------------------------------------|
| 1 | Single-request baseline      | Unoptimised latency per request (no batching)      |
| 2 | Cold vs. warm cache          | Cache speedup for identical prompts                |
| 3 | Cache hit-rate over time     | Rolling hit rate as traffic accumulates            |
| 4 | Throughput by load level     | Req/s and P95 latency at low / medium / high load  |
| 5 | Batching impact — concurrency| Per-request latency at varying concurrency levels  |

Results are saved as JSON in `benchmarks/results/` and charts in `analysis/visualizations/`.

---

## Configuration Reference

| Environment Variable       | Default               | Description                        |
|----------------------------|-----------------------|------------------------------------|
| `LLM_MODEL_NAME`           | `sshleifer/tiny-gpt2` | HuggingFace model ID               |
| `LLM_MAX_BATCH_SIZE`       | `8`                   | Max requests per batch             |
| `LLM_BATCH_TIMEOUT_MS`     | `50.0`                | Batch flush timeout (ms)           |
| `LLM_CACHE_BACKEND`        | `memory`              | `memory` or `redis`                |
| `LLM_REDIS_URL`            | `redis://localhost`   | Redis connection string            |
| `LLM_CACHE_TTL_SECONDS`    | `3600`                | Cache entry TTL (seconds)          |
| `LLM_CACHE_MAX_ENTRIES`    | `1000`                | Max in-process cache entries (LRU) |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `30.0`             | Per-request timeout                |
| `LLM_MAX_OUTPUT_TOKENS`    | `200`                 | Hard cap on generated tokens       |

---

## Using Redis Cache (Production)

```bash
# Start Redis (Docker)
docker run -d -p 6379:6379 redis:alpine

# Configure server
export LLM_CACHE_BACKEND=redis
export LLM_REDIS_URL=redis://localhost:6379

uvicorn src.server:app --port 8000
```

---

## Privacy & Governance Notes

- **No plaintext prompts are stored** — all cache keys are SHA-256 hashes.
- **No user identifiers** are ever included in cache keys or stored data.
- **TTL expiration** ensures cached data ages out automatically.
- Cache bypass is available per-request via `bypass_cache: true`.
- Temperature > 0 requests are automatically excluded from caching (non-deterministic).

See `analysis/governance_memo.pdf` for the full governance memo.

---

## Reproducibility Checklist

1. Clone the repository
2. `pip install -r requirements.txt`
3. `uvicorn src.server:app --port 8000`
4. `python benchmarks/run_benchmarks.py --target http://localhost:8000`

All benchmark scripts are deterministic (seeded where applicable) and produce
consistent results across runs on the same hardware.