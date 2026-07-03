import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Local subprocess execution. This is NOT a security sandbox; replace with an
# isolated, network-disabled Docker container before exposing to untrusted input.

RUNNER_TEMPLATE = """\
import pandas as pd
import numpy as np
df = pd.read_csv({csv_path!r})
{code}
"""


@dataclass
class ExecResult:
    ok: bool
    output: str
    error: str


def execute_python(code: str, csv_path: str, timeout: int = 30) -> ExecResult:
    """
    Run model-generated code with the dataframe `df` preloaded from csv_path.

    The code is expected to print any values it wants to observe. Returns the
    captured stdout on success, or the traceback on failure/timeout.
    """
    script = RUNNER_TEMPLATE.format(csv_path=csv_path, code=code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, output="", error=f"Execution timed out after {timeout}s.")
    finally:
        os.unlink(script_path)

    if proc.returncode != 0:
        return ExecResult(ok=False, output=proc.stdout, error=proc.stderr.strip())

    output = proc.stdout.strip() or "(no output; remember to print() the values you need)"
    return ExecResult(ok=True, output=output, error="")
