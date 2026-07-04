import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from data_analyst_agent.agent.loop import analyze
from data_analyst_agent.tracing import Tracer

from .checker import score_item
from .dabench import load_items
from .metrics import SUMMARY_HEADER, aggregate, run_cost_usd, summary_markdown

RESULTS_DIR = Path(__file__).parent / "results"


def evaluate_item(item, model: str, max_steps: int) -> dict:
    tracer = Tracer()
    record = {
        "id": item.id,
        "level": item.level,
        "file_name": item.file_name,
        "sub_answers_total": len(item.labels),
        "sub_answers_matched": 0,
        "correct": False,
        "failed": False,
    }

    start = time.perf_counter()
    try:
        answer = analyze(item.task_text(), item.csv_path, model=model, max_steps=max_steps, tracer=tracer)
        record.update(score_item(item.labels, answer))
        record["answer"] = answer
    except Exception as exc:
        record["failed"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"

    record["latency_s"] = time.perf_counter() - start
    record["steps"] = sum(1 for e in tracer.events if e["type"] == "usage")
    record["exec_errors"] = sum(
        1 for e in tracer.events if e["type"] == "tool_call" and not e["ok"]
    )
    record.update(tracer.totals)
    record["cost_usd"] = run_cost_usd(model, tracer.totals)
    return record


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the agent on a DABench subset.")
    parser.add_argument("--n", type=int, default=25, help="Number of questions to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model id to evaluate.")
    parser.add_argument("--max-steps", type=int, default=8, help="Agent step limit per question.")
    parser.add_argument("--config", default="agent", help="Config label recorded in the summary.")
    args = parser.parse_args()

    items = load_items(n=args.n, seed=args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.model}_{args.config}_n{args.n}_seed{args.seed}.jsonl"

    records = []
    with out_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            record = evaluate_item(item, args.model, args.max_steps)
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            status = "FAIL" if record["failed"] else ("ok" if record["correct"] else "wrong")
            print(
                f"[{i}/{len(items)}] id={record['id']} level={record['level']} "
                f"{status} steps={record['steps']} cost=${record['cost_usd'] or 0:.4f}"
            )

    summary = aggregate(records)
    print(json.dumps(summary, indent=2))

    summary_path = RESULTS_DIR / "summary.md"
    row = summary_markdown(args.model, args.config, summary)
    if summary_path.exists():
        summary_path.write_text(summary_path.read_text(encoding="utf-8") + row + "\n", encoding="utf-8")
    else:
        summary_path.write_text(SUMMARY_HEADER + "\n" + row + "\n", encoding="utf-8")
    print(f"Results: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
