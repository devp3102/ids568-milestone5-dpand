import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.load_generator import run_load_test, _single_request, generate_prompt

RESULTS_DIR = Path(__file__).parent / "results"
VIZ_DIR = Path(__file__).parent.parent / "analysis" / "visualizations"


# Helpers

def save_json(name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved: {path}")
    return path


async def _clear_cache(session: aiohttp.ClientSession, target: str):
    try:
        async with session.post(f"{target}/cache/clear") as resp:
            await resp.json()
        print("  Cache cleared")
    except Exception as e:
        print(f"  Warning: could not clear cache: {e}")


# Experiment 1: Single-request baseline (no batching)

async def bench_single_requests(session: aiohttp.ClientSession, target: str, n: int = 10):
    """Send N requests one at a time (sequential) to establish a latency baseline."""
    print(f"\n[1] Single-request baseline (n={n})")
    await _clear_cache(session, target)

    latencies = []
    prompts = [generate_prompt(repeat_ratio=0.0) for _ in range(n)]  # all unique

    for i, prompt in enumerate(prompts):
        result = await _single_request(session, target, prompt, max_tokens=100)
        latencies.append(result.latency_ms)
        print(f"    {i+1}/{n}  {result.latency_ms:.1f} ms  cached={result.cached}")

    latencies_sorted = sorted(latencies)
    summary = {
        "experiment": "single_request_baseline",
        "n": n,
        "latency_mean_ms": round(sum(latencies) / n, 2),
        "latency_p50_ms": latencies_sorted[n // 2],
        "latency_p95_ms": latencies_sorted[int(n * 0.95)],
        "latency_p99_ms": latencies_sorted[int(n * 0.99)],
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "raw_latencies_ms": latencies,
    }
    save_json("single_request_baseline", summary)
    return summary


# Experiment 2: Cold vs. warm cache comparison

async def bench_cache(session: aiohttp.ClientSession, target: str, n: int = 20):
    """Compare cold-cache and warm-cache latency for identical prompts."""
    print(f"\n[2] Cold vs. warm cache (n={n})")

    # Same pool of prompts repeated twice
    prompts = [generate_prompt(repeat_ratio=1.0) for _ in range(n)]  # from repeat pool

    # --- Cold run ---
    await _clear_cache(session, target)
    cold_latencies = []
    for p in prompts:
        r = await _single_request(session, target, p, temperature=0.0)
        cold_latencies.append(r.latency_ms)
    cold_mean = round(sum(cold_latencies) / n, 2)
    print(f"  Cold mean latency: {cold_mean} ms")

    # --- Warm run (same prompts, cache populated) ---
    warm_latencies = []
    for p in prompts:
        r = await _single_request(session, target, p, temperature=0.0)
        warm_latencies.append(r.latency_ms)
    warm_mean = round(sum(warm_latencies) / n, 2)
    print(f"  Warm mean latency: {warm_mean} ms")

    speedup = round(cold_mean / warm_mean, 2) if warm_mean else None

    summary = {
        "experiment": "cold_vs_warm_cache",
        "n": n,
        "cold_mean_ms": cold_mean,
        "warm_mean_ms": warm_mean,
        "speedup": speedup,
        "cold_latencies_ms": cold_latencies,
        "warm_latencies_ms": warm_latencies,
    }
    save_json("cold_vs_warm_cache", summary)
    return summary


# Experiment 3: Cache hit-rate over time

async def bench_cache_hitrate(
    session: aiohttp.ClientSession, target: str, n: int = 50, repeat_ratio: float = 0.5
):
    """Track rolling cache hit rate as traffic accumulates."""
    print(f"\n[3] Cache hit-rate tracking (n={n}, repeat_ratio={repeat_ratio})")
    await _clear_cache(session, target)

    rolling = []
    hits = 0

    for i in range(n):
        prompt = generate_prompt(repeat_ratio)
        r = await _single_request(session, target, prompt, temperature=0.0)
        if r.cached:
            hits += 1
        rolling.append(round(hits / (i + 1), 4))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n}  cumulative hit rate: {rolling[-1]:.1%}")

    summary = {
        "experiment": "cache_hitrate_over_time",
        "n": n,
        "repeat_ratio": repeat_ratio,
        "final_hit_rate": rolling[-1] if rolling else 0,
        "rolling_hit_rate": rolling,
    }
    save_json("cache_hitrate", summary)
    return summary

# Experiment 4: Throughput under multiple load levels

async def bench_throughput(target: str, levels: list = None):
    """Measure throughput (req/s) at low, medium, and high load."""
    if levels is None:
        levels = [
            {"rate": 2, "duration": 20, "label": "low"},
            {"rate": 5, "duration": 20, "label": "medium"},
            {"rate": 10, "duration": 20, "label": "high"},
        ]

    print("\n[4] Throughput under multiple load levels")
    all_results = []

    for cfg in levels:
        print(f"  Load: {cfg['label']} ({cfg['rate']} req/s for {cfg['duration']}s)")
        result = await run_load_test(
            target=target,
            rate=cfg["rate"],
            duration=cfg["duration"],
            concurrency=cfg["rate"] * 3,   # Concurrency 3× rate
            repeat_ratio=0.4,
            max_tokens=100,
        )
        summary = result.summary()
        summary["load_level"] = cfg["label"]
        summary["target_rate"] = cfg["rate"]
        all_results.append(summary)

        for k in ["throughput_rps", "latency_p50_ms", "latency_p95_ms", "cache_hit_rate", "error_rate"]:
            print(f"    {k}: {summary.get(k)}")

    save_json("throughput_by_load_level", {"experiments": all_results})
    return all_results


# Experiment 5: Batch size impact on latency

async def bench_batching_impact(target: str):
    """
    Compare per-request latency at different concurrency levels to show
    batching's effect (lower effective latency per request at higher load).
    """
    print("\n[5] Batching impact — concurrency sweep")
    concurrency_levels = [1, 4, 8, 16]
    results = []

    for c in concurrency_levels:
        print(f"  Concurrency={c}")
        result = await run_load_test(
            target=target,
            rate=c,
            duration=15,
            concurrency=c,
            repeat_ratio=0.0,     # All unique → no cache help
            max_tokens=100,
        )
        s = result.summary()
        s["concurrency"] = c
        results.append(s)
        print(f"    throughput={s['throughput_rps']} rps  p50={s['latency_p50_ms']} ms")

    save_json("batching_impact", {"experiments": results})
    return results


# Visualisation

def generate_visualizations():
    """Generate charts from saved benchmark results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skipping charts")
        return

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    #Chart 1: Cold vs Warm cache bar chart 
    try:
        data = json.loads((RESULTS_DIR / "cold_vs_warm_cache.json").read_text())
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            ["Cold cache", "Warm cache"],
            [data["cold_mean_ms"], data["warm_mean_ms"]],
            color=["#e74c3c", "#2ecc71"],
        )
        ax.bar_label(bars, fmt="%.1f ms", padding=3)
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title(f"Cache Impact: Cold vs Warm\n(speedup: {data['speedup']}×)")
        ax.set_ylim(0, max(data["cold_mean_ms"], data["warm_mean_ms"]) * 1.25)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "cold_vs_warm_cache.png", dpi=150)
        plt.close()
        print(f"  Chart: cold_vs_warm_cache.png")
    except FileNotFoundError:
        pass

    #Chart 2: Cache hit-rate over time 
    try:
        data = json.loads((RESULTS_DIR / "cache_hitrate.json").read_text())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(data["rolling_hit_rate"]) + 1), data["rolling_hit_rate"],
                marker="o", markersize=3, linewidth=1.5, color="#3498db")
        ax.set_xlabel("Request Number")
        ax.set_ylabel("Cumulative Hit Rate")
        ax.set_title("Cache Hit Rate Over Time")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "cache_hitrate_over_time.png", dpi=150)
        plt.close()
        print(f"  Chart: cache_hitrate_over_time.png")
    except FileNotFoundError:
        pass

    #Chart 3: Throughput vs load level 
    try:
        data = json.loads((RESULTS_DIR / "throughput_by_load_level.json").read_text())
        exps = data["experiments"]
        labels = [e["load_level"] for e in exps]
        rps = [e["throughput_rps"] for e in exps]
        p95 = [e["latency_p95_ms"] for e in exps]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(labels, rps, color="#9b59b6")
        ax1.set_ylabel("Throughput (req/s)")
        ax1.set_title("Throughput by Load Level")

        ax2.bar(labels, p95, color="#e67e22")
        ax2.set_ylabel("P95 Latency (ms)")
        ax2.set_title("P95 Latency by Load Level")

        plt.tight_layout()
        plt.savefig(VIZ_DIR / "throughput_by_load_level.png", dpi=150)
        plt.close()
        print(f"  Chart: throughput_by_load_level.png")
    except FileNotFoundError:
        pass

    #Chart 4: Batching impact 
    try:
        data = json.loads((RESULTS_DIR / "batching_impact.json").read_text())
        exps = data["experiments"]
        concurrencies = [e["concurrency"] for e in exps]
        p50 = [e["latency_p50_ms"] for e in exps]
        throughputs = [e["throughput_rps"] for e in exps]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(concurrencies, p50, marker="o", color="#1abc9c")
        ax1.set_xlabel("Concurrency")
        ax1.set_ylabel("P50 Latency (ms)")
        ax1.set_title("Latency vs Concurrency (Batching Effect)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(concurrencies, throughputs, marker="o", color="#e74c3c")
        ax2.set_xlabel("Concurrency")
        ax2.set_ylabel("Throughput (req/s)")
        ax2.set_title("Throughput vs Concurrency")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(VIZ_DIR / "batching_impact.png", dpi=150)
        plt.close()
        print(f"  Chart: batching_impact.png")
    except FileNotFoundError:
        pass

    #Chart 5: Single request latency distribution 
    try:
        data = json.loads((RESULTS_DIR / "single_request_baseline.json").read_text())
        lats = data["raw_latencies_ms"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(lats, bins=min(20, len(lats)), color="#2980b9", edgecolor="white", alpha=0.8)
        ax.axvline(data["latency_p50_ms"], color="green", linestyle="--", label="P50")
        ax.axvline(data["latency_p95_ms"], color="orange", linestyle="--", label="P95")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Single-Request Latency Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "latency_distribution.png", dpi=150)
        plt.close()
        print(f"  Chart: latency_distribution.png")
    except FileNotFoundError:
        pass


# Main orchestration

async def run_all(target: str, quick: bool = False):
    print("=" * 60)
    print("Milestone 5 Benchmark Suite")
    print(f"Target: {target}")
    print("=" * 60)

    n_single = 5 if quick else 20
    n_cache = 10 if quick else 30
    n_hitrate = 20 if quick else 50
    load_levels = (
        [{"rate": 1, "duration": 10, "label": "low"},
         {"rate": 2, "duration": 10, "label": "medium"},
         {"rate": 3, "duration": 10, "label": "high"}]
        if quick else
        [{"rate": 2, "duration": 20, "label": "low"},
         {"rate": 5, "duration": 20, "label": "medium"},
         {"rate": 10, "duration": 20, "label": "high"}]
    )

    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        await bench_single_requests(session, target, n=n_single)
        await bench_cache(session, target, n=n_cache)
        await bench_cache_hitrate(session, target, n=n_hitrate)

    await bench_throughput(target, levels=load_levels)
    await bench_batching_impact(target)

    print("\n[6] Generating visualizations")
    generate_visualizations()

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print(f"Results: {RESULTS_DIR}")
    print(f"Charts:  {VIZ_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark orchestration for Milestone 5 LLM Inference Server"
    )
    parser.add_argument(
        "--target", default="http://localhost:8000", help="Server base URL"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a shorter version of all benchmarks (for quick testing)"
    )
    args = parser.parse_args()
    asyncio.run(run_all(args.target, quick=args.quick))


if __name__ == "__main__":
    main()