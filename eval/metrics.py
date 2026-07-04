# USD per 1M tokens, from https://api-docs.deepseek.com/quick_start/pricing/
PRICING = {
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


def run_cost_usd(model: str, totals: dict) -> float | None:
    """Compute the cost of one run from tracer totals; None for unknown models."""
    prices = PRICING.get(model)
    if prices is None:
        return None
    return (
        totals["cache_hit_tokens"] * prices["cache_hit"]
        + totals["cache_miss_tokens"] * prices["cache_miss"]
        + totals["completion_tokens"] * prices["output"]
    ) / 1_000_000


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-item eval records into summary metrics."""
    n = len(records)
    finished = [r for r in records if not r.get("failed")]
    costs = [r["cost_usd"] for r in finished if r.get("cost_usd") is not None]
    sub_total = sum(r["sub_answers_total"] for r in finished)
    return {
        "items": n,
        "accuracy": sum(r["correct"] for r in finished) / n if n else 0.0,
        "sub_answer_accuracy": (
            sum(r["sub_answers_matched"] for r in finished) / sub_total if sub_total else 0.0
        ),
        "run_failure_rate": (n - len(finished)) / n if n else 0.0,
        "exec_error_rate": (
            sum(r["exec_errors"] for r in finished) / max(sum(r["steps"] for r in finished), 1)
        ),
        "avg_steps": sum(r["steps"] for r in finished) / len(finished) if finished else 0.0,
        "avg_latency_s": sum(r["latency_s"] for r in finished) / len(finished) if finished else 0.0,
        "cache_hit_rate": _cache_hit_rate(finished),
        "total_cost_usd": sum(costs),
    }


def _cache_hit_rate(records: list[dict]) -> float:
    hits = sum(r["cache_hit_tokens"] for r in records)
    misses = sum(r["cache_miss_tokens"] for r in records)
    return hits / (hits + misses) if hits + misses else 0.0


def summary_markdown(model: str, config: str, summary: dict) -> str:
    return (
        f"| {model} | {config} | {summary['items']} "
        f"| {summary['accuracy']:.1%} | {summary['sub_answer_accuracy']:.1%} "
        f"| {summary['avg_steps']:.1f} | {summary['cache_hit_rate']:.1%} "
        f"| {summary['avg_latency_s']:.1f}s | ${summary['total_cost_usd']:.4f} |"
    )


SUMMARY_HEADER = (
    "| Model | Config | Items | Accuracy | Sub-answer acc. | Avg steps "
    "| Cache hit | Avg latency | Total cost |\n"
    "|---|---|---|---|---|---|---|---|---|"
)
