# data-analyst-agent

An LLM agent that answers questions about tabular data by writing and running Python, observing the result, and correcting its own errors. Benchmarked on a data-analysis benchmark rather than shipped on vibes.

## Why

Most "chat-with-data" demos are a single prompt with no measurement. This project is built as an agent with an explicit tool-use loop, an isolated code sandbox, and an evaluation harness that reports correctness, cost, and latency across configurations.

## How it works

```
question + dataframe summary (schema, dtypes, sample rows)
        |
        v
  +----------------------------------+
  | 1. model plans / writes code     |
  | 2. run_python tool executes it   |<-----+
  | 3. stdout or traceback returned  |      | on error: feed traceback back, retry
  | 4. enough to answer? ------------+------+
  +----------------------------------+
        |
        v
  final answer in plain language (with the key numbers)
```

The model never sees the full table, only a compact summary, so it must query the data through code.

## Quickstart

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt
pip install -e . --no-deps      # adds the data-analyst-agent CLI entry point

# 3. set the API key
cp .env.example .env            # then edit DEEPSEEK_API_KEY

# 4. ask a question
data-analyst-agent data/sample_sales.csv "What is the total revenue per region?"
```

## Project structure

```
src/data_analyst_agent/
  agent/loop.py       agentic tool-use loop
  agent/prompts.py    system and task prompts
  sandbox/executor.py code execution in a subprocess
  context.py          compact dataframe summary
  tracing.py          per-step JSONL trace
  cli.py              command-line entry point
eval/                 benchmark harness
app/                  Streamlit UI
```

## Roadmap

- Minimal tool-use loop with a CLI (done)
- Harden and measure self-correction from tracebacks
- Multi-step answers with chart capture
- Evaluation harness on an InfiAgent-DABench subset comparing configurations
- Streamlit app, Docker-isolated sandbox, and deployment

Basic error feedback is already in the loop, so simple self-correction works out of the box.

## Evaluation plan

Run a subset of InfiAgent-DABench (question + CSV + closed-form answer) and compare:
- model tier (deepseek-v4-flash vs deepseek-v4-pro, thinking vs non-thinking),
- single-shot code vs. the self-correcting loop,
- with vs. without the dataframe summary in context,
- a local model vs. a hosted model (cost/quality).

Metrics: answer correctness, execution success rate, steps to answer, tokens, cost, latency.

## Security

The executor runs model-generated code in a subprocess with a timeout. This is for local development only and is NOT sandboxed. Do not point it at untrusted input or expose it publicly until it is replaced with an isolated, network-disabled Docker container.

## License

MIT
