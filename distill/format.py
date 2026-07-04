import argparse
import json
from pathlib import Path

# Long tool outputs (big prints from the kernel) inflate sequence length and add
# little training signal. Cap them so trajectories fit a modest context window.
DEFAULT_MAX_TOOL_CHARS = 2000


def truncate_tool_outputs(messages: list[dict], max_chars: int) -> list[dict]:
    """Return messages with over-long tool results clipped and marked."""
    clipped = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and len(content) > max_chars:
            message = {**message, "content": content[:max_chars] + "\n... [output truncated]"}
        clipped.append(message)
    return clipped


def format_file(input_path: Path, output_path: Path, max_tool_chars: int) -> int:
    """Read raw trajectories and write cleaned {messages, tools} examples for training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            trajectory = json.loads(line)
            example = {
                "messages": truncate_tool_outputs(trajectory["messages"], max_tool_chars),
                "tools": trajectory["tools"],
            }
            dst.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare teacher trajectories for SFT.")
    parser.add_argument("--input", required=True, help="Raw trajectories JSONL.")
    parser.add_argument("--output", required=True, help="Formatted training JSONL.")
    parser.add_argument("--max-tool-chars", type=int, default=DEFAULT_MAX_TOOL_CHARS,
                        help="Clip tool outputs longer than this many characters.")
    args = parser.parse_args()

    written = format_file(Path(args.input), Path(args.output), args.max_tool_chars)
    print(f"Wrote {written} training examples to {args.output}")


if __name__ == "__main__":
    main()
