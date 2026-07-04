import json
import sys
from pathlib import Path
from typing import Optional


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def _usage_field(usage, name: str) -> int:
    value = getattr(usage, name, None)
    if value is None and getattr(usage, "model_extra", None):
        value = usage.model_extra.get(name)
    return value or 0


class Tracer:
    """Collect agent steps, optionally writing them to JSONL and/or echoing them live."""

    def __init__(self, path: Optional[str] = None, verbose: bool = False):
        self.path = Path(path) if path else None
        self.verbose = verbose
        self.events: list[dict] = []
        self.messages: list = []
        self.totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }

    def _emit(self, event: dict, console: Optional[str] = None) -> None:
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.verbose and console:
            print(console, file=sys.stderr)

    def log_thought(self, step: int, text: str) -> None:
        self._emit(
            {"step": step, "type": "thought", "text": text},
            console=f"--- step {step} ---\nthinking:\n{_indent(text)}",
        )

    def log_step(self, step: int, code: str, result) -> None:
        status = "ok" if result.ok else "error"
        observed = result.output if result.ok else result.error
        self._emit(
            {
                "step": step,
                "type": "tool_call",
                "code": code,
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
            },
            console=f"run_python ({status}):\n{_indent(code)}\nobserved:\n{_indent(observed)}",
        )

    def log_usage(self, step: int, usage) -> None:
        prompt = _usage_field(usage, "prompt_tokens")
        completion = _usage_field(usage, "completion_tokens")
        cache_hit = _usage_field(usage, "prompt_cache_hit_tokens")
        cache_miss = _usage_field(usage, "prompt_cache_miss_tokens")

        self.totals["prompt_tokens"] += prompt
        self.totals["completion_tokens"] += completion
        self.totals["cache_hit_tokens"] += cache_hit
        self.totals["cache_miss_tokens"] += cache_miss

        self._emit(
            {
                "step": step,
                "type": "usage",
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cache_hit_tokens": cache_hit,
                "cache_miss_tokens": cache_miss,
            },
            console=f"usage: prompt={prompt} (cache hit={cache_hit}, miss={cache_miss}), completion={completion}",
        )

    def log_final(self, step: int, answer: str) -> None:
        self._emit({"step": step, "type": "final", "answer": answer})
