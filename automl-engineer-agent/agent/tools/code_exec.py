"""
Sandboxed Python code execution for dynamic analysis/transformations.
"""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr


def exec_python(code: str, globals_: dict | None = None) -> tuple[str, str, dict]:
    """
    Execute Python code in a restricted namespace.
    Returns (stdout, stderr, namespace).
    """
    globals_ = globals_ or {}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_)
    except Exception as e:
        stderr_capture.write(str(e))

    return stdout_capture.getvalue(), stderr_capture.getvalue(), globals_
