"""The package must import cleanly when the ``mcp`` extra is not installed."""

import subprocess
import sys

_SCRIPT = """
import sys

sys.modules["mcp"] = None  # make every "import mcp..." raise ImportError

import champi_tts
import champi_tts.mcp.server as server

assert server.MCP_AVAILABLE is False
assert server.mcp is None
try:
    server.main()
except ImportError as exc:
    assert "champi-tts[mcp]" in str(exc), exc
else:
    raise AssertionError("main() should require the mcp extra")
print("ok")
"""


def test_imports_without_mcp_extra() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
