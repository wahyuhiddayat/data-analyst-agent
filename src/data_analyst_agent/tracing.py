import json
from pathlib import Path
from typing import Optional


class Tracer:
    """Collect agent steps in memory and optionally append them to a JSONL file."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []

    def _write(self, event: dict) -> None:
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_step(self, step: int, code: str, result) -> None:
        self._write({
            "step": step,
            "type": "tool_call",
            "code": code,
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
        })

    def log_final(self, step: int, answer: str) -> None:
        self._write({"step": step, "type": "final", "answer": answer})
