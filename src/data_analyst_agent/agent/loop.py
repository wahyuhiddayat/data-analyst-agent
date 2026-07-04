import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from ..context import summarize_dataframe
from ..sandbox.executor import KernelSession
from ..tracing import Tracer
from .prompts import SYSTEM_PROMPT, build_task_message

DEFAULT_MODEL = os.getenv("DAA_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MAX_STEPS = 8

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python code against the preloaded pandas DataFrame `df`. "
            "`df`, `pd`, and `np` are available and state persists across calls. "
            "Use print() to output any value you want to observe; only stdout, "
            "error tracebacks, and saved figures are returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        },
    },
}


def _bootstrap(csv_path: str) -> str:
    return (
        "%matplotlib inline\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        f"df = pd.read_csv({csv_path!r})\n"
    )


def analyze(
    question: str,
    csv_path: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    temperature: float = 0.0,
    tracer: Tracer | None = None,
) -> str:
    """
    Answer a question about a CSV file by letting the model iteratively write and
    run Python in a persistent kernel until it can respond. Returns the model's
    final plain-language answer.
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=DEFAULT_BASE_URL)
    tracer = tracer or Tracer()

    # Resolve to absolute so the kernel can read it from its isolated working dir.
    csv_path = str(Path(csv_path).resolve())
    df = pd.read_csv(csv_path)
    schema = summarize_dataframe(df)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_task_message(question, schema)},
    ]
    tracer.messages = messages

    with KernelSession() as session:
        session.run(_bootstrap(csv_path))

        for step in range(max_steps):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[RUN_PYTHON_TOOL],
                max_tokens=8192,
                temperature=temperature,
            )
            tracer.log_usage(step, response.usage)
            message = response.choices[0].message
            messages.append(message)

            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                tracer.log_thought(step, reasoning)

            if not message.tool_calls:
                answer = message.content or ""
                tracer.log_final(step, answer)
                return answer

            if message.content:
                tracer.log_thought(step, message.content)

            for call in message.tool_calls:
                try:
                    code = json.loads(call.function.arguments)["code"]
                except (json.JSONDecodeError, KeyError):
                    # Truncated or malformed arguments; tell the model to retry.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Error: tool arguments were not valid JSON "
                                   "(possibly truncated). Retry with shorter code.",
                    })
                    continue
                result = session.run(code)
                tracer.log_step(step, code, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.output if result.ok else f"Error:\n{result.error}",
                })

        return "Reached the step limit without producing a final answer."
