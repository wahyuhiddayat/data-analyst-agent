import json
import os

import pandas as pd
from openai import OpenAI

from ..context import summarize_dataframe
from ..sandbox.executor import execute_python
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
            "`df`, `pd`, and `np` are available. Use print() to output any value you "
            "want to observe; only stdout and error tracebacks are returned."
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


def analyze(
    question: str,
    csv_path: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> str:
    """
    Answer a question about a CSV file by letting the model iteratively write and
    run Python until it can respond. Returns the model's final plain-language answer.
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=DEFAULT_BASE_URL)
    tracer = tracer or Tracer()

    df = pd.read_csv(csv_path)
    schema = summarize_dataframe(df)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_task_message(question, schema)},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[RUN_PYTHON_TOOL],
            max_tokens=2048,
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            answer = message.content or ""
            tracer.log_final(step, answer)
            return answer

        for call in message.tool_calls:
            code = json.loads(call.function.arguments)["code"]
            result = execute_python(code, csv_path)
            tracer.log_step(step, code, result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result.output if result.ok else f"Error:\n{result.error}",
            })

    return "Reached the step limit without producing a final answer."
