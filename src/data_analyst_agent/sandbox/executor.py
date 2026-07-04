import base64
import queue
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client import KernelManager

# Code runs in a persistent IPython kernel so state carries across calls within a
# session. This is NOT a security sandbox; run the kernel inside an isolated,
# network-disabled container before exposing it to untrusted input.

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


@dataclass
class ExecResult:
    ok: bool
    output: str
    error: str
    images: list[str] = field(default_factory=list)


class KernelSession:
    """
    A persistent IPython kernel. Variables defined in one run() are visible to the
    next, and matplotlib figures are captured and saved as PNG files.
    """

    def __init__(self, output_dir: str = "outputs", exec_timeout: int = 30):
        self._km = KernelManager(kernel_name="python3")
        # Discard the kernel process's own stderr (startup banners/warnings);
        # code errors still arrive over the iopub channel, not here.
        self._km.start_kernel(stderr=subprocess.DEVNULL)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=60)
        self._exec_timeout = exec_timeout
        self._output_dir = Path(output_dir)
        self._fig_count = 0
        # Run generated code in a throwaway directory so files it writes do not
        # land in the project tree. Figures are still saved to _output_dir, which
        # is resolved against the host process, not the kernel.
        self._workdir = tempfile.mkdtemp(prefix="daa_kernel_")
        self.run(f"import os as _os; _os.chdir({self._workdir!r})")

    def run(self, code: str) -> ExecResult:
        msg_id = self._kc.execute(code)
        stdout_parts: list[str] = []
        error_text = ""
        images: list[str] = []

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=self._exec_timeout)
            except queue.Empty:
                return ExecResult(
                    ok=False,
                    output="".join(stdout_parts).strip(),
                    error=f"Execution timed out after {self._exec_timeout}s.",
                    images=images,
                )
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "stream":
                stdout_parts.append(content["text"])
            elif mtype in ("execute_result", "display_data"):
                data = content["data"]
                if "image/png" in data:
                    images.append(self._save_png(data["image/png"]))
                elif "text/plain" in data:
                    stdout_parts.append(data["text/plain"] + "\n")
            elif mtype == "error":
                error_text = _strip_ansi("\n".join(content["traceback"]))
            elif mtype == "status" and content["execution_state"] == "idle":
                break

        output = "".join(stdout_parts).strip()
        if images:
            note = "\n".join(f"[figure saved: {p}]" for p in images)
            output = f"{output}\n{note}" if output else note

        if error_text:
            return ExecResult(ok=False, output=output, error=error_text, images=images)
        if not output:
            output = "(no output; remember to print() the values you need)"
        return ExecResult(ok=True, output=output, error="", images=images)

    def _save_png(self, b64: str) -> str:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fig_count += 1
        path = self._output_dir / f"figure_{self._fig_count}.png"
        path.write_bytes(base64.b64decode(b64))
        return str(path)

    def shutdown(self) -> None:
        try:
            self._kc.stop_channels()
            self._km.shutdown_kernel(now=True)
        except Exception:
            pass
        shutil.rmtree(self._workdir, ignore_errors=True)

    def __enter__(self) -> "KernelSession":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
