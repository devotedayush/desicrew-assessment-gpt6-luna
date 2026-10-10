"""Execute a deliberately restricted pandas query in a disposable OS sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import pandas as pd

from ._sandbox_runner import SandboxValidationError, validate_code


class SandboxExecutionError(RuntimeError):
    """The generated query failed without changing the parent process."""


def execute_query(code: str, sheets: dict[str, pd.DataFrame], timeout: float = 8.0) -> dict:
    """Return a JSON-safe result; raise on invalid code, limits, or missing isolation.

    Python -I ignores user site-packages, PYTHONPATH and startup hooks. The child
    gets no API keys or application environment and cannot open files, create
    sockets, launch subprocesses, or write to disk after setup. Linux/libseccomp
    is required; this function deliberately has no insecure fallback.
    """
    validate_code(code)
    if not sheets or any(not isinstance(frame, pd.DataFrame) for frame in sheets.values()):
        raise SandboxExecutionError("Supply one or more pandas worksheets.")
    if not 0 < timeout <= 30:
        raise ValueError("Timeout must be greater than zero and at most 30 seconds.")
    payload = json.dumps({"code": code, "sheets": {str(name): frame.to_json(orient="table", date_format="iso", index=False) for name, frame in sheets.items()}, "cpu_seconds": min(10, max(1, int(timeout) - 1))}, ensure_ascii=False)
    if len(payload.encode("utf-8")) > 32 * 1024 * 1024:
        raise SandboxExecutionError("Worksheet payload exceeds the 32 MiB sandbox limit.")
    env = {
        "PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    runner = Path(__file__).with_name("_sandbox_runner.py").resolve()
    with tempfile.TemporaryDirectory(prefix="desicrew-query-") as cwd:
        process = subprocess.Popen([sys.executable, "-I", "-B", str(runner)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd, close_fds=True, start_new_session=True)
        try:
            stdout, _stderr = process.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise SandboxExecutionError(f"Query exceeded its {timeout:g}-second time limit.") from None
    if len(stdout) > 110_000:
        raise SandboxExecutionError("Query exceeded the output limit.")
    try:
        response = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        if process.returncode and process.returncode < 0:
            raise SandboxExecutionError("Query was stopped by an operating-system resource limit.") from None
        raise SandboxExecutionError("The sandbox did not return a valid bounded result.") from None
    if process.returncode != 0 or not response.get("ok"):
        raise SandboxExecutionError(response.get("error", "Query execution failed."))
    if response.get("isolation") != "linux-seccomp":
        raise SandboxExecutionError("The required Linux sandbox was not verified.")
    return response["result"]


__all__ = ["execute_query", "validate_code", "SandboxValidationError", "SandboxExecutionError"]
