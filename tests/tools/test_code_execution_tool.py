"""Tests for CodeExecutionTool."""

from __future__ import annotations

from pathlib import Path

from darwin.tools.code_execution import CodeExecutionTool


def test_simple_print_succeeds(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path)
    result = code.execute({"source": "print('it works')"})
    assert result.success
    assert "it works" in result.output


def test_exception_returns_failed_result(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path)
    result = code.execute({"source": "raise RuntimeError('boom')"})
    assert not result.success
    assert "RuntimeError" in result.error
    assert "boom" in result.error


def test_timeout_enforced(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path, timeout_seconds=0.5)
    result = code.execute({"source": "import time; time.sleep(5)"})
    assert not result.success
    assert "timeout" in result.error.lower()


def test_empty_source_rejected(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path)
    result = code.execute({"source": ""})
    assert not result.success
    assert "empty" in result.error.lower()


def test_subprocess_cwd_is_sandbox(tmp_path: Path) -> None:
    # `os` is now denylisted by the static-AST inspector; opt into the
    # trusted execution surface to read cwd from the subprocess.
    code = CodeExecutionTool(tmp_path, allow_unsafe=True)
    result = code.execute({
        "source": "import os; print(os.getcwd())"
    })
    assert result.success
    assert str(tmp_path.resolve()) in result.output


def test_dangerous_import_rejected_by_static_inspection(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path)
    result = code.execute({"source": "import os; os.system('echo bad')"})
    assert not result.success
    assert "static inspection" in result.error.lower()


def test_eval_call_rejected_by_static_inspection(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path)
    result = code.execute({"source": "x = eval('1 + 1'); print(x)"})
    assert not result.success
    assert "static inspection" in result.error.lower()


def test_trusted_flag_bypasses_inspection(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path)
    result = code.execute({
        "source": "import os; print(os.getcwd())",
        "trusted": True,
    })
    assert result.success


def test_stdout_truncated_to_configured_size(tmp_path: Path) -> None:
    code = CodeExecutionTool(tmp_path, max_output_bytes=128)
    result = code.execute({
        "source": "print('X' * 5000)"
    })
    assert result.success
    assert len(result.output) <= 128
