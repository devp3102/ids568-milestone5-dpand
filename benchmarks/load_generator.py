import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import aiohttp

# Prompt pool — mix of unique and repeated to simulate real traffic

_REPEATED_PROMPTS = [
    "What is machine learning?",
    "Explain the transformer architecture.",
    "What is the difference between supervised and unsupervised learning?",
    "How does gradient descent work?",
    "What is overfitting and how do you prevent it?",
    "Describe the attention mechanism in transformers.",
    "What is a neural network?",
    "How does backpropagation work?",
    "What is regularization in deep learning?",
    "Explain the concept of embeddings.",
]

_UNIQUE_TEMPLATES = [
    "Describe the history of {topic} in detail.",
    "What are the key challenges in {topic}?",
    "List 5 important facts about {topic}.",
    "Summarize the latest developments in {topic}.",
    "Explain {topic} to a 10-year-old.",
]

_UNIQUE_TOPICS = [
    "quantum computing", "reinforcement learning", "computer vision",
    "natural language processing", "federated learning", "edge inference",
    "model compression", "knowledge distillation", "transfer learning",
    "generative adversarial networks", "diffusion models", "autonomous vehicles",
    "drug discovery AI", "climate modeling", "financial forecasting",
]


def generate_prompt(repeat_ratio: float = 0.4) -> str:
    """Return a prompt from the repeated pool or a freshly generated unique one."""
    if random.random() < repeat_ratio:
        return random.choice(_REPEATED_PROMPTS)
    topic = random.choice(_UNIQUE_TOPICS)
    template = random.choice(_UNIQUE_TEMPLATES)
    return template.format(topic=topic)


# Result data class

@dataclass
class RequestResult:
    prompt: str
    latency_ms: float
    status_code: int
    cached: bool
    error: Optional[str] = None


@dataclass
class LoadTestResults:
    target: str
    rate: float
    duration_s: float
    concurrency: int
    repeat_ratio: float
    results: List[RequestResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    #Derived statistics 

    def summary(self) -> dict:
        successful = [r for r in self.results if r.error is None and r.status_code == 200]
        errors = [r for r in self.results if r.error is not None or r.status_code != 200]
        latencies = [r.latency_ms for r in successful]
        cached = [r for r in successful if r.cached]

        latencies_sorted = sorted(latencies) if latencies else [0]

        def pct(lst, p):
            if not lst:
                return 0
            idx = int(len(lst) * p / 100)
            return round(lst[min(idx, len(lst) - 1)], 2)

        return {
            "total_requests": len(self.results),
            "successful": len(successful),
            "errors": len(errors),
            "error_rate": round(len(errors) / max(len(self.results), 1), 4),
            "cache_hits": len(cached),
            "cache_hit_rate": round(len(cached) / max(len(successful), 1), 4),
            "throughput_rps": round(len(self.results) / max(self.duration_s, 1), 2),
            "latency_p50_ms": pct(latencies_sorted, 50),
            "latency_p95_ms": pct(latencies_sorted, 95),
            "latency_p99_ms": pct(latencies_sorted, 99),
            "latency_min_ms": round(min(latencies_sorted), 2),
            "latency_max_ms": round(max(latencies_sorted), 2),
            "latency_mean_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        }


# Load generator

async def _single_request(
    session: aiohttp.ClientSession,
    target: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> RequestResult:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{target}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            body = await resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status == 200:
                return RequestResult(
                    prompt=prompt,
                    latency_ms=round(latency_ms, 2),
                    status_code=200,
                    cached=body.get("cached", False),
                )
            return RequestResult(
                prompt=prompt,
                latency_ms=round(latency_ms, 2),
                status_code=resp.status,
                cached=False,
                error=str(body),
            )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            prompt=prompt,
            latency_ms=round(latency_ms, 2),
            status_code=0,
            cached=False,
            error=str(exc),
        )


async def run_load_test(
    target: str,
    rate: float,           # requests per second
    duration: float,       # seconds
    concurrency: int = 20,
    repeat_ratio: float = 0.4,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> LoadTestResults:
    """
    Send requests at the given `rate` (req/s) for `duration` seconds.

    A semaphore limits maximum in-flight requests to `concurrency`.
    """
    load_result = LoadTestResults(
        target=target,
        rate=rate,
        duration_s=duration,
        concurrency=concurrency,
        repeat_ratio=repeat_ratio,
    )

    semaphore = asyncio.Semaphore(concurrency)
    interval = 1.0 / rate
    deadline = time.perf_counter() + duration

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        async def _bounded_request(prompt: str):
            async with semaphore:
                r = await _single_request(session, target, prompt, max_tokens, temperature)
                load_result.results.append(r)

        while time.perf_counter() < deadline:
            prompt = generate_prompt(repeat_ratio)
            tasks.append(asyncio.create_task(_bounded_request(prompt)))
            # Rate limiting: sleep between request dispatches
            await asyncio.sleep(interval)

        # Wait for all in-flight requests to finish
        await asyncio.gather(*tasks, return_exceptions=True)

    return load_result


# CLI

def main():
    parser = argparse.ArgumentParser(description="Synthetic load generator for Milestone 5")
    parser.add_argument("--target", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--rate", type=float, default=10.0, help="Requests per second")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--repeat-ratio", type=float, default=0.4,
                        help="Fraction of requests from repeated prompt pool (0-1)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Output tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output", default=None, help="Save JSON results to this file")
    args = parser.parse_args()

    print(f"Load test: {args.rate} req/s for {args.duration}s → {args.target}")
    results = asyncio.run(
        run_load_test(
            target=args.target,
            rate=args.rate,
            duration=args.duration,
            concurrency=args.concurrency,
            repeat_ratio=args.repeat_ratio,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )

    summary = results.summary()
    print("\n=== Load Test Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "target": args.target,
                "rate": args.rate,
                "duration_s": args.duration,
                "concurrency": args.concurrency,
                "repeat_ratio": args.repeat_ratio,
            },
            "summary": summary,
            "raw_results": [asdict(r) for r in results.results],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()