"""JSON schema contract tests for MCP tool parameter schemas.

``EXPECTED_SCHEMAS`` freezes the parameter schema of every registered MCP
tool. Any change to a tool's parameters (name, type, default, required-ness,
added or removed argument) or to the registered tool set fails here and must
be made deliberately by updating the contract. Behaviour is covered by
``tests/test_mcp/``; this file covers the contract only.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mcp")


def _string(title: str) -> dict[str, Any]:
    return {"title": title, "type": "string"}


def _nullable(json_type: str, title: str) -> dict[str, Any]:
    return {
        "anyOf": [{"type": json_type}, {"type": "null"}],
        "default": None,
        "title": title,
    }


def _boolean(title: str, default: bool) -> dict[str, Any]:
    return {"default": default, "title": title, "type": "boolean"}


def _object(
    tool: str,
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "properties": properties,
        "title": f"{tool}Arguments",
        "type": "object",
    }
    if required:
        schema["required"] = required
    return schema


_SPEAK_PROPERTIES: dict[str, dict[str, Any]] = {
    "text": _string("Text"),
    "voice": _nullable("string", "Voice"),
    "speed": _nullable("number", "Speed"),
    "provider": _nullable("string", "Provider"),
    "device_index": _nullable("integer", "Device Index"),
}

EXPECTED_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_providers": _object("list_providers", {}),
    "get_provider_status": _object(
        "get_provider_status",
        {"provider": _string("Provider")},
        required=["provider"],
    ),
    "synthesize_to_file": _object(
        "synthesize_to_file",
        {
            "text": _string("Text"),
            "output_path": _string("Output Path"),
            "voice": _nullable("string", "Voice"),
            "speed": _nullable("number", "Speed"),
            "provider": _nullable("string", "Provider"),
            "overwrite": _boolean("Overwrite", False),
        },
        required=["text", "output_path"],
    ),
    "list_voices": _object(
        "list_voices", {"provider": _nullable("string", "Provider")}
    ),
    "speak_tool": _object(
        "speak_tool",
        {**_SPEAK_PROPERTIES, "wait": _boolean("Wait", True)},
        required=["text"],
    ),
    "speak_streaming_tool": _object(
        "speak_streaming_tool", dict(_SPEAK_PROPERTIES), required=["text"]
    ),
    "stop_speaking_tool": _object("stop_speaking_tool", {}),
    "list_audio_devices_tool": _object("list_audio_devices_tool", {}),
}

_TOOL_NAMES = sorted(EXPECTED_SCHEMAS)

# (tool, parameter) pairs whose value may be omitted and defaults to ``None``.
_NULLABLE_PARAMS = [
    (tool, name)
    for tool, schema in EXPECTED_SCHEMAS.items()
    for name, prop in schema["properties"].items()
    if "anyOf" in prop
]


@pytest.fixture(scope="module")
def tools() -> dict[str, Any]:
    """Return the tool registry from a freshly created MCP server."""
    from champi_tts.mcp.server import create_mcp_server

    return create_mcp_server()._tool_manager._tools


def _schema(tools: dict[str, Any], name: str) -> dict[str, Any]:
    return tools[name].parameters


class TestToolSet:
    def test_exact_tool_set_registered(self, tools: dict[str, Any]) -> None:
        assert set(tools) == set(EXPECTED_SCHEMAS)

    def test_exactly_eight_tools(self, tools: dict[str, Any]) -> None:
        assert len(tools) == 8


@pytest.mark.parametrize("tool", _TOOL_NAMES)
class TestToolSchemaContract:
    def test_schema_matches_frozen_contract(
        self, tools: dict[str, Any], tool: str
    ) -> None:
        assert _schema(tools, tool) == EXPECTED_SCHEMAS[tool]

    def test_schema_is_object_type(self, tools: dict[str, Any], tool: str) -> None:
        assert _schema(tools, tool)["type"] == "object"

    def test_required_params(self, tools: dict[str, Any], tool: str) -> None:
        expected = EXPECTED_SCHEMAS[tool].get("required", [])
        assert _schema(tools, tool).get("required", []) == expected

    def test_no_unknown_properties(self, tools: dict[str, Any], tool: str) -> None:
        props = set(_schema(tools, tool).get("properties", {}))
        assert props == set(EXPECTED_SCHEMAS[tool]["properties"])

    def test_property_types(self, tools: dict[str, Any], tool: str) -> None:
        actual = _schema(tools, tool).get("properties", {})
        for name, expected in EXPECTED_SCHEMAS[tool]["properties"].items():
            assert actual[name].get("type") == expected.get("type"), name
            assert actual[name].get("anyOf") == expected.get("anyOf"), name


@pytest.mark.parametrize(("tool", "param"), _NULLABLE_PARAMS)
class TestNullableParams:
    def test_accepts_null(self, tools: dict[str, Any], tool: str, param: str) -> None:
        prop = _schema(tools, tool)["properties"][param]
        assert {"type": "null"} in prop["anyOf"]

    def test_defaults_to_null(
        self, tools: dict[str, Any], tool: str, param: str
    ) -> None:
        assert _schema(tools, tool)["properties"][param]["default"] is None

    def test_not_required(self, tools: dict[str, Any], tool: str, param: str) -> None:
        assert param not in _schema(tools, tool).get("required", [])
