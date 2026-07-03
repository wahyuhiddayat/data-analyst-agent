import argparse

from dotenv import load_dotenv

from .agent.loop import analyze


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Answer a question about a CSV dataset.")
    parser.add_argument("csv", help="Path to the CSV file.")
    parser.add_argument("question", help="Question to answer about the data.")
    parser.add_argument("--model", default=None, help="Override the default model id.")
    args = parser.parse_args()

    kwargs = {"model": args.model} if args.model else {}
    answer = analyze(args.question, args.csv, **kwargs)
    print("\n" + answer)


if __name__ == "__main__":
    main()
