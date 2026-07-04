import argparse
import sys

from dotenv import load_dotenv

from .agent.loop import analyze
from .tracing import Tracer


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Answer a question about a CSV dataset.")
    parser.add_argument("csv", help="Path to the CSV file.")
    parser.add_argument("question", help="Question to answer about the data.")
    parser.add_argument("--model", default=None, help="Override the default model id.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the agent's reasoning and code steps as it works.",
    )
    parser.add_argument("--trace", default=None, help="Write a JSONL trace to this path.")
    args = parser.parse_args()

    tracer = Tracer(path=args.trace, verbose=args.verbose)
    kwargs = {"model": args.model} if args.model else {}
    answer = analyze(args.question, args.csv, tracer=tracer, **kwargs)
    print("\n" + answer)

    if args.verbose:
        t = tracer.totals
        print(
            f"\n[usage] prompt={t['prompt_tokens']} "
            f"(cache hit={t['cache_hit_tokens']}, miss={t['cache_miss_tokens']}), "
            f"completion={t['completion_tokens']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
