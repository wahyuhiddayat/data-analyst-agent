import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from data_analyst_agent.agent.loop import RUN_PYTHON_TOOL, analyze
from data_analyst_agent.tracing import Tracer

from .checker import score_item
from .dabench import load_items
from .metrics import run_cost_usd

TRAJ_DIR = Path(__file__).parent / "trajectories"

# The held-out evaluation set (same subset the teacher baselines were measured on).
# Trajectories are collected from every OTHER question to avoid train/test leakage.
TEST_N = 25
TEST_SEED = 0


def serialize_messages(messages: list) -> list[dict]:
    """
    Convert the loop's conversation into plain JSON turns for SFT.

    Dict turns (system/user/tool) pass through; assistant turns are SDK objects,
    reduced to role/content/tool_calls. The teacher's private reasoning_content
    is intentionally dropped -- the student learns actions and answers, not the
    teacher's chain of thought.
    """
    serialized = []
    for m in messages:
        if isinstance(m, dict):
            serialized.append(m)
            continue
        data = m.model_dump(exclude_none=True)
        turn = {"role": data.get("role", "assistant"), "content": data.get("content") or ""}
        if data.get("tool_calls"):
            turn["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in data["tool_calls"]
            ]
        serialized.append(turn)
    return serialized


def _signature(messages: list[dict]) -> tuple:
    """Identify a trajectory by its sequence of executed code, to drop exact duplicates."""
    codes = []
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            codes.append(tc["function"]["arguments"])
    return tuple(codes)


def collect_item(item, model: str, max_steps: int, temperature: float):
    tracer = Tracer()
    start = time.perf_counter()
    try:
        answer = analyze(
            item.task_text(), item.csv_path,
            model=model, max_steps=max_steps, temperature=temperature, tracer=tracer,
        )
    except Exception as exc:
        return None, {"id": item.id, "level": item.level, "failed": True,
                      "error": f"{type(exc).__name__}: {exc}"}

    score = score_item(item.labels, answer)
    meta = {
        "id": item.id,
        "level": item.level,
        "failed": False,
        "correct": score["correct"],
        "steps": sum(1 for e in tracer.events if e["type"] == "usage"),
        "exec_errors": sum(1 for e in tracer.events if e["type"] == "tool_call" and not e["ok"]),
        "latency_s": time.perf_counter() - start,
        "cost_usd": run_cost_usd(model, tracer.totals),
    }
    if not score["correct"]:
        return None, meta

    trajectory = {
        "id": item.id,
        "level": item.level,
        "model": model,
        "question": item.question,
        "file_name": item.file_name,
        "steps": meta["steps"],
        "cost_usd": meta["cost_usd"],
        "tools": [RUN_PYTHON_TOOL],
        "messages": serialize_messages(tracer.messages),
    }
    return trajectory, meta


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Collect teacher trajectories from DABench.")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Teacher model id.")
    parser.add_argument("--max-steps", type=int, default=12, help="Agent step limit per question.")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes over the train split.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; use >0 with multiple passes for variety.")
    parser.add_argument("--limit", type=int, default=None, help="Cap train questions (for smoke runs).")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing output and start over.")
    parser.add_argument("--repass", action="store_true",
                        help="Re-attempt every train question and append; combine with --temperature "
                             "to collect alternative solutions.")
    args = parser.parse_args()

    test_ids = {it.id for it in load_items(n=TEST_N, seed=TEST_SEED)}
    train_items = [it for it in load_items() if it.id not in test_ids]

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    traj_path = TRAJ_DIR / f"{safe_model}.jsonl"
    log_path = TRAJ_DIR / f"{safe_model}.log.jsonl"

    # Resume by default: skip questions already attempted, append to existing output.
    # --repass keeps the existing output but re-attempts every question, so extra
    # passes can grow the dataset; dedup by signature drops identical solutions.
    seen: set[tuple] = set()
    done_ids: set[int] = set()
    kept = 0
    resuming = not args.fresh and log_path.exists()
    if resuming:
        if traj_path.exists():
            for line in traj_path.open(encoding="utf-8"):
                traj = json.loads(line)
                seen.add((traj["id"], _signature(traj["messages"])))
                kept += 1
        if args.repass:
            print(f"Re-pass: re-attempting all {len(train_items)} train questions, {kept} kept so far.")
        else:
            for line in log_path.open(encoding="utf-8"):
                done_ids.add(json.loads(line)["id"])
            train_items = [it for it in train_items if it.id not in done_ids]
            print(f"Resuming: {len(done_ids)} already attempted, {kept} kept.")

    if args.limit is not None:
        train_items = train_items[: args.limit]

    total_cost = 0.0
    attempts = 0
    file_mode = "a" if resuming else "w"

    with traj_path.open(file_mode, encoding="utf-8") as traj_out, \
         log_path.open(file_mode, encoding="utf-8") as log_out:
        for p in range(args.passes):
            for item in train_items:
                attempts += 1
                trajectory, meta = collect_item(item, args.model, args.max_steps, args.temperature)
                total_cost += meta.get("cost_usd") or 0.0
                meta["pass"] = p
                log_out.write(json.dumps(meta, ensure_ascii=False) + "\n")
                log_out.flush()

                status = "FAIL" if meta.get("failed") else ("kept" if trajectory else "wrong")
                if trajectory is not None:
                    sig = (item.id, _signature(trajectory["messages"]))
                    if sig in seen:
                        status = "dup"
                    else:
                        seen.add(sig)
                        traj_out.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                        traj_out.flush()
                        kept += 1
                print(f"[pass {p} | {attempts}] id={meta['id']} {meta['level']:<6} "
                      f"{status} kept={kept} cost=${total_cost:.3f}")

    by_level: dict[str, int] = {}
    for line in traj_path.open(encoding="utf-8"):
        lvl = json.loads(line)["level"]
        by_level[lvl] = by_level.get(lvl, 0) + 1
    print(json.dumps({
        "trajectories_kept": kept,
        "attempts": attempts,
        "by_level": by_level,
        "total_cost_usd": round(total_cost, 4),
    }, indent=2))
    print(f"Trajectories: {traj_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
