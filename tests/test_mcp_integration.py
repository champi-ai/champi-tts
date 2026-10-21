"""Integration tests for the MCP server subprocess stdio handshake.

These tests spawn the real ``champi-tts-mcp`` process and speak raw,
newline-delimited JSON-RPC over its stdin/stdout, complementing the
``mcp``-client based tests in ``tests/test_mcp/test_server.py``. No model
files or audio hardware are needed: the provider loads lazily on the first
synthesis call, which these tests never make.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

_TIMEOUT = 30  # seconds — CI guard per response

_EXPECTED_TOOLS = {
    "list_providers",
    "get_provider_status",
    "synthesize_to_file",
    "list_voices",
    "speak_tool",
    "speak_streaming_tool",
    "stop_speaking_tool",
    "list_audio_devices_tool",
}


def _server_command() -> list[str]:
    """Return the ``champi-tts-mcp`` command, preferring the active venv's copy."""
    entry = Path(sys.executable).parent / "champi-tts-mcp"
    if entry.exists():
        return [str(entry)]
    found = shutil.which("champi-tts-mcp")
    if found is None:
        pytest.skip("champi-tts-mcp entry point is not installed")
    return [found]


def _send(proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _start_reader(proc: subprocess.Popen[str]) -> queue.Queue[dict[str, Any]]:
    """Pump decoded stdout lines into a queue so reads can time out."""
    lines: queue.Queue[dict[str, Any]] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.strip()
            if stripped:
                lines.put(json.loads(stripped))

    threading.Thread(target=_pump, daemon=True).start()
    return lines


def _response(lines: queue.Queue[dict[str, Any]], request_id: int) -> dict[str, Any]:
    """Return the JSON-RPC response for ``request_id``, skipping notifications."""
    while True:
        try:
            message = lines.get(timeout=_TIMEOUT)
        except queue.Empty:
            pytest.fail(f"MCP server sent no response to id={request_id}")
        if message.get("id") == request_id:
            return message


@pytest.fixture(scope="module")
def exchange() -> dict[str, dict[str, Any]]:
    """Spawn the server once, run initialize + tools/list, return both responses."""
    proc = subprocess.Popen(
        _server_command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        lines = _start_reader(proc)
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0.1"},
                },
            },
        )
        initialize = _response(lines, 1)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools_list = _response(lines, 2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {"initialize": initialize, "tools/list": tools_list}


class TestStdioHandshake:
    def test_initialize_returns_result(self, exchange: dict) -> None:
        response = exchange["initialize"]
        assert response["jsonrpc"] == "2.0"
        assert "error" not in response
        assert "result" in response

    def test_server_name_is_champi_tts(self, exchange: dict) -> None:
        assert exchange["initialize"]["result"]["serverInfo"]["name"] == "champi-tts"

    def test_server_version_is_non_empty(self, exchange: dict) -> None:
        version = exchange["initialize"]["result"]["serverInfo"]["version"]
        assert isinstance(version, str)
        assert version

    def test_protocol_version_is_negotiated(self, exchange: dict) -> None:
        assert exchange["initialize"]["result"]["protocolVersion"] == "2024-11-05"

    def test_advertises_tools_capability(self, exchange: dict) -> None:
        assert "tools" in exchange["initialize"]["result"]["capabilities"]


class TestStdioToolsList:
    def test_tools_list_returns_exact_tool_set(self, exchange: dict) -> None:
        names = {tool["name"] for tool in exchange["tools/list"]["result"]["tools"]}
        assert names == _EXPECTED_TOOLS

    def test_every_tool_has_input_schema(self, exchange: dict) -> None:
        for tool in exchange["tools/list"]["result"]["tools"]:
            assert tool["inputSchema"]["type"] == "object", tool["name"]
