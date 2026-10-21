"""Tests for the ``champi-tts mcp serve`` CLI command."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from champi_tts.cli.main import app

runner = CliRunner()


@pytest.fixture
def fake_server():
    """Install a stand-in ``champi_tts.mcp.server`` module with a mocked main."""
    module = types.ModuleType("champi_tts.mcp.server")
    module.MCP_AVAILABLE = True  # type: ignore[attr-defined]
    module.main = MagicMock()  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"champi_tts.mcp.server": module}):
        yield module


def test_mcp_help_lists_serve() -> None:
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_mcp_serve_help_lists_options_and_env_vars() -> None:
    result = runner.invoke(app, ["mcp", "serve", "--help"])
    assert result.exit_code == 0
    for text in (
        "--transport",
        "--host",
        "--port",
        "8766",
        "CHAMPI_TTS_PROVIDER",
        "CHAMPI_OUTPUT_DEVICE_INDEX",
        "KOKORO_",
    ):
        assert text in result.output


def test_serve_defaults_forwarded(fake_server) -> None:
    result = runner.invoke(app, ["mcp", "serve"])
    assert result.exit_code == 0, result.output
    fake_server.main.assert_called_once_with(
        transport="stdio", host="localhost", port=8766
    )


def test_serve_options_forwarded(fake_server) -> None:
    result = runner.invoke(
        app,
        ["mcp", "serve", "--transport", "sse", "--host", "0.0.0.0", "--port", "9001"],
    )
    assert result.exit_code == 0, result.output
    fake_server.main.assert_called_once_with(transport="sse", host="0.0.0.0", port=9001)


@pytest.mark.parametrize(
    "args",
    [
        ["--transport", "websocket"],
        ["--port", "not-a-number"],
    ],
)
def test_serve_rejects_invalid_options(fake_server, args) -> None:
    result = runner.invoke(app, ["mcp", "serve", *args])
    assert result.exit_code == 2
    fake_server.main.assert_not_called()


def test_serve_exits_when_mcp_unavailable(fake_server) -> None:
    fake_server.MCP_AVAILABLE = False
    result = runner.invoke(app, ["mcp", "serve"])
    assert result.exit_code == 1
    assert "champi-tts[mcp]" in result.stderr
    fake_server.main.assert_not_called()


def test_serve_exits_when_server_import_fails() -> None:
    with patch.dict(sys.modules, {"champi_tts.mcp.server": None}):
        result = runner.invoke(app, ["mcp", "serve"])
    assert result.exit_code == 1
    assert "champi-tts[mcp]" in result.stderr
