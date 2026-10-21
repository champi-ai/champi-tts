"""JSON schema and protocol-level tests for the MCP tools.

Checks the argument schemas FastMCP advertises to clients for every tool
registered by ``create_mcp_server()``, that ``list_tools`` exposes them with
descriptions, and that invalid arguments are rejected before any provider
is touched. Tool behaviour lives in ``test_server.py`` and
``test_speaker_tools.py``; the provider is mocked so no GPU is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp")

from mcp.server.fastmcp.exceptions import ToolError

CORE_TOOLS = (
    "list_providers",
    "get_provider_status",
    "synthesize_to_file",
    "list_voices",
)
SPEAKER_TOOLS = (
    "speak_tool",
    "speak_streaming_tool",
    "stop_speaking_tool",
    "list_audio_devices_tool",
)

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_NULLABLE_INTEGER = {"anyOf": [{"type": "integer"}, {"type": "null"}]}

# tool -> (required params, {param: expected type schema, default if optional})
EXPECTED_SCHEMAS: dict[str, tuple[list[str], dict[str, tuple[dict, Any]]]] = {
    "list_providers": ([], {}),
    "get_provider_status": (["provider"], {"provider": ({"type": "string"}, ...)}),
    "synthesize_to_file": (
        ["text", "output_path"],
        {
            "text": ({"type": "string"}, ...),
            "output_path": ({"type": "string"}, ...),
            "voice": (_NULLABLE_STRING, None),
            "speed": (_NULLABLE_NUMBER, None),
            "provider": (_NULLABLE_STRING, None),
            "overwrite": ({"type": "boolean"}, False),
        },
    ),
    "list_voices": ([], {"provider": (_NULLABLE_STRING, None)}),
    "speak_tool": (
        ["text"],
        {
            "text": ({"type": "string"}, ...),
            "voice": (_NULLABLE_STRING, None),
            "speed": (_NULLABLE_NUMBER, None),
            "provider": (_NULLABLE_STRING, None),
            "device_index": (_NULLABLE_INTEGER, None),
            "wait": ({"type": "boolean"}, True),
        },
    ),
    "speak_streaming_tool": (
        ["text"],
        {
            "text": ({"type": "string"}, ...),
            "voice": (_NULLABLE_STRING, None),
            "speed": (_NULLABLE_NUMBER, None),
            "provider": (_NULLABLE_STRING, None),
            "device_index": (_NULLABLE_INTEGER, None),
        },
    ),
    "stop_speaking_tool": ([], {}),
    "list_audio_devices_tool": ([], {}),
}


@pytest.fixture()
def server() -> Any:
    """Fresh FastMCP server instance per test."""
    from champi_tts.mcp.server import create_mcp_server

    return create_mcp_server()


def _schema(server: Any, tool_name: str) -> dict:
    return server._tool_manager._tools[tool_name].parameters


class TestRegistry:
    def test_registers_exactly_the_expected_tools(self, server: Any) -> None:
        assert set(server._tool_manager._tools) == set(CORE_TOOLS + SPEAKER_TOOLS)

    @pytest.mark.asyncio
    async def test_list_tools_advertises_every_tool_with_description(
        self, server: Any
    ) -> None:
        tools = {t.name: t for t in await server.list_tools()}
        assert set(tools) == set(CORE_TOOLS + SPEAKER_TOOLS)
        for name, tool in tools.items():
            assert tool.description, name
            assert tool.inputSchema == _schema(server, name)


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_SCHEMAS))
class TestToolSchemas:
    def test_schema_is_object(self, server: Any, tool_name: str) -> None:
        assert _schema(server, tool_name)["type"] == "object"

    def test_required_params(self, server: Any, tool_name: str) -> None:
        required, _ = EXPECTED_SCHEMAS[tool_name]
        assert _schema(server, tool_name).get("required", []) == required

    def test_property_names(self, server: Any, tool_name: str) -> None:
        _, props = EXPECTED_SCHEMAS[tool_name]
        assert list(_schema(server, tool_name)["properties"]) == list(props)

    def test_property_types_and_defaults(self, server: Any, tool_name: str) -> None:
        _, props = EXPECTED_SCHEMAS[tool_name]
        schema_props = _schema(server, tool_name)["properties"]
        for name, (type_schema, default) in props.items():
            prop = schema_props[name]
            for key, value in type_schema.items():
                assert prop[key] == value, (tool_name, name)
            if default is ...:
                assert "default" not in prop, (tool_name, name)
            else:
                assert prop["default"] == default, (tool_name, name)


class TestArgumentValidation:
    """Invalid arguments are rejected by FastMCP before the tool body runs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "args", "field"),
        [
            ("get_provider_status", {}, "provider"),
            ("synthesize_to_file", {"text": "hi"}, "output_path"),
            ("synthesize_to_file", {"output_path": "a.wav"}, "text"),
            ("speak_tool", {}, "text"),
            ("speak_streaming_tool", {}, "text"),
        ],
    )
    async def test_missing_required_argument_raises_tool_error(
        self, server: Any, tool_name: str, args: dict, field: str
    ) -> None:
        with (
            patch("champi_tts.get_provider") as mock_get,
            pytest.raises(ToolError, match=field),
        ):
            await server.call_tool(tool_name, args)
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "args", "field"),
        [
            (
                "synthesize_to_file",
                {"text": "hi", "output_path": "a.wav", "speed": "fast"},
                "speed",
            ),
            ("speak_tool", {"text": "hi", "device_index": "left"}, "device_index"),
            (
                "speak_streaming_tool",
                {"text": "hi", "speed": "slow"},
                "speed",
            ),
        ],
    )
    async def test_wrongly_typed_argument_raises_tool_error(
        self, server: Any, tool_name: str, args: dict, field: str
    ) -> None:
        with (
            patch("champi_tts.get_provider") as mock_get,
            pytest.raises(ToolError, match=field),
        ):
            await server.call_tool(tool_name, args)
        mock_get.assert_not_called()


class TestCoreToolsOverProtocol:
    """Smoke-test the four core tools through ``call_tool`` with a mocked provider."""

    @pytest.mark.asyncio
    async def test_list_providers_returns_strings(self, server: Any) -> None:
        _, structured = await server.call_tool("list_providers", {})
        assert structured["result"]
        assert all(isinstance(name, str) for name in structured["result"])

    @pytest.mark.asyncio
    async def test_get_provider_status_passes_provider_name(self, server: Any) -> None:
        prov = MagicMock()
        prov.get_model_info = AsyncMock(return_value={"status": "not_loaded"})
        with patch("champi_tts.get_provider", return_value=prov) as mock_get:
            _, structured = await server.call_tool(
                "get_provider_status", {"provider": "kokoro"}
            )
        assert structured == {"status": "not_loaded"}
        mock_get.assert_called_once_with("kokoro")

    @pytest.mark.asyncio
    async def test_list_voices_defaults_to_kokoro(
        self, server: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CHAMPI_TTS_PROVIDER", raising=False)
        prov = MagicMock()
        prov.list_voices = AsyncMock(return_value=["af_bella"])
        with patch("champi_tts.get_provider", return_value=prov) as mock_get:
            _, structured = await server.call_tool("list_voices", {})
        assert structured["result"] == ["af_bella"]
        mock_get.assert_called_once_with("kokoro")
        prov.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_to_file_coerces_numeric_speed(
        self, server: Any, tmp_path: Any
    ) -> None:
        import numpy as np

        prov = MagicMock()
        prov.config.sample_rate = 24000
        prov.initialize = AsyncMock()
        prov.synthesize = AsyncMock(return_value=np.zeros(240, np.float32))
        out = tmp_path / "speech.wav"
        with patch("champi_tts.get_provider", return_value=prov):
            _, structured = await server.call_tool(
                "synthesize_to_file",
                {"text": "hi", "output_path": str(out), "speed": 1},
            )
        assert structured["result"] == str(out)
        assert out.exists()
        prov.synthesize.assert_awaited_once_with("hi", voice=None, speed=1.0)
