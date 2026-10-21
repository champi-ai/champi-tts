"""Tests for the MCP server — factory, lifespan, tools, and main()."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mcp")


def _make_provider(
    *,
    model_info: dict | None = None,
    voices: list[str] | None = None,
    audio: np.ndarray | None = None,
    sample_rate: int = 24000,
    initialize_raises: Exception | None = None,
    synthesize_raises: Exception | None = None,
    list_voices_raises: Exception | None = None,
) -> MagicMock:
    prov = MagicMock()
    prov.config.sample_rate = sample_rate
    prov.initialize = AsyncMock(side_effect=initialize_raises)
    prov.shutdown = AsyncMock()
    prov.synthesize = AsyncMock(
        side_effect=synthesize_raises,
        return_value=audio if audio is not None else np.zeros(2400, np.float32),
    )
    prov.list_voices = AsyncMock(
        side_effect=list_voices_raises,
        return_value=voices if voices is not None else ["af_bella", "bm_george"],
    )
    prov.get_model_info = AsyncMock(
        return_value=model_info or {"status": "loaded", "provider": "mock"}
    )
    return prov


def _raw(call_tool_result: tuple) -> dict:
    """Extract the structured result dict from FastMCP call_tool return value."""
    return call_tool_result[1]


@pytest.fixture()
def server():
    """Fresh FastMCP server instance per test."""
    from champi_tts.mcp.server import create_mcp_server

    return create_mcp_server()


class TestModuleConstants:
    def test_mcp_available_flag_is_true(self) -> None:
        from champi_tts.mcp.server import MCP_AVAILABLE

        assert MCP_AVAILABLE is True

    def test_require_mcp_raises_when_unavailable(self) -> None:
        with patch("champi_tts.mcp.server.MCP_AVAILABLE", False):
            from champi_tts.mcp.server import _require_mcp

            with pytest.raises(ImportError, match=r"champi-tts\[mcp\]"):
                _require_mcp()

    def test_main_raises_import_error_when_mcp_unavailable(self) -> None:
        with patch("champi_tts.mcp.server.MCP_AVAILABLE", False):
            from champi_tts.mcp.server import main

            with pytest.raises(ImportError, match="mcp"):
                main()


class TestCreateMcpServer:
    def test_returns_fastmcp_instance(self, server: MagicMock) -> None:
        from mcp.server.fastmcp import FastMCP

        assert isinstance(server, FastMCP)

    def test_registers_all_four_tools(self, server: MagicMock) -> None:
        tools = set(server._tool_manager._tools.keys())
        assert {
            "list_providers",
            "get_provider_status",
            "synthesize_to_file",
            "list_voices",
        } <= tools

    def test_exposes_state_for_test_injection(self, server: MagicMock) -> None:
        assert server._state["provider"] is None

    def test_servers_do_not_share_state(self) -> None:
        from champi_tts.mcp.server import create_mcp_server

        assert create_mcp_server()._state is not create_mcp_server()._state

    @pytest.mark.asyncio
    async def test_lifespan_shuts_down_and_clears_provider(
        self, server: MagicMock
    ) -> None:
        mock_prov = _make_provider()
        server._state["provider"] = mock_prov

        async with server.settings.lifespan(server):
            pass

        mock_prov.shutdown.assert_awaited_once()
        assert server._state["provider"] is None

    @pytest.mark.asyncio
    async def test_lifespan_suppresses_shutdown_errors(self, server: MagicMock) -> None:
        mock_prov = _make_provider()
        mock_prov.shutdown = AsyncMock(side_effect=RuntimeError("GPU gone"))
        server._state["provider"] = mock_prov

        async with server.settings.lifespan(server):
            pass

        assert server._state["provider"] is None

    @pytest.mark.asyncio
    async def test_lifespan_noop_when_no_provider(self, server: MagicMock) -> None:
        async with server.settings.lifespan(server):
            pass


class TestToolListProviders:
    @pytest.mark.asyncio
    async def test_contains_kokoro(self, server: MagicMock) -> None:
        result = _raw(await server.call_tool("list_providers", {}))
        assert "kokoro" in result["result"]


class TestToolGetProviderStatus:
    @pytest.mark.asyncio
    async def test_returns_model_info(self, server: MagicMock) -> None:
        prov = _make_provider(model_info={"status": "not_initialized", "x": 1})
        with patch("champi_tts.get_provider", return_value=prov):
            result = _raw(
                await server.call_tool("get_provider_status", {"provider": "kokoro"})
            )
        assert result["status"] == "not_initialized"
        prov.initialize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_error_dict(self, server: MagicMock) -> None:
        result = _raw(
            await server.call_tool("get_provider_status", {"provider": "bad"})
        )
        assert result["error"] is True
        assert result["error_type"] == "ValueError"
        assert result["provider"] == "bad"


class TestToolSynthesizeToFile:
    @pytest.mark.asyncio
    async def test_writes_wav_and_returns_path(
        self, server: MagicMock, tmp_path: Path
    ) -> None:
        import soundfile as sf

        prov = _make_provider(audio=np.zeros(1200, np.float32), sample_rate=22050)
        out = tmp_path / "nested" / "out.wav"
        with patch("champi_tts.get_provider", return_value=prov):
            result = _raw(
                await server.call_tool(
                    "synthesize_to_file",
                    {
                        "text": "hello",
                        "output_path": str(out),
                        "voice": "af_bella",
                        "speed": 1.2,
                    },
                )
            )
        assert result["result"] == str(out.resolve())
        data, rate = sf.read(str(out))
        assert rate == 22050
        assert len(data) == 1200
        prov.synthesize.assert_awaited_once_with("hello", voice="af_bella", speed=1.2)

    @pytest.mark.asyncio
    async def test_provider_initialised_once(
        self, server: MagicMock, tmp_path: Path
    ) -> None:
        prov = _make_provider()
        with patch("champi_tts.get_provider", return_value=prov) as mock_get:
            for name in ("a.wav", "b.wav"):
                await server.call_tool(
                    "synthesize_to_file",
                    {"text": "hi", "output_path": str(tmp_path / name)},
                )
        prov.initialize.assert_awaited_once()
        mock_get.assert_called_once_with("kokoro")
        assert server._state["provider"] is prov

    @pytest.mark.asyncio
    async def test_env_var_selects_provider(
        self, server: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHAMPI_TTS_PROVIDER", "other")
        prov = _make_provider()
        with patch("champi_tts.get_provider", return_value=prov) as mock_get:
            await server.call_tool(
                "synthesize_to_file",
                {"text": "hi", "output_path": str(tmp_path / "o.wav")},
            )
        mock_get.assert_called_with("other")

    @pytest.mark.asyncio
    async def test_explicit_provider_overrides_env_var(
        self, server: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHAMPI_TTS_PROVIDER", "other")
        prov = _make_provider()
        with patch("champi_tts.get_provider", return_value=prov) as mock_get:
            await server.call_tool(
                "synthesize_to_file",
                {
                    "text": "hi",
                    "output_path": str(tmp_path / "o.wav"),
                    "provider": "kokoro",
                },
            )
        mock_get.assert_called_with("kokoro")

    @pytest.mark.asyncio
    async def test_empty_text_returns_error_string(
        self, server: MagicMock, tmp_path: Path
    ) -> None:
        result = _raw(
            await server.call_tool(
                "synthesize_to_file",
                {"text": "   ", "output_path": str(tmp_path / "o.wav")},
            )
        )
        assert result["result"].startswith("error:")

    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_error_string(
        self, server: MagicMock, tmp_path: Path
    ) -> None:
        prov = _make_provider(synthesize_raises=RuntimeError("boom"))
        with patch("champi_tts.get_provider", return_value=prov):
            result = _raw(
                await server.call_tool(
                    "synthesize_to_file",
                    {"text": "hi", "output_path": str(tmp_path / "o.wav")},
                )
            )
        assert result["result"] == "error: RuntimeError: boom"

    @pytest.mark.asyncio
    async def test_initialize_failure_returns_error_and_allows_retry(
        self, server: MagicMock, tmp_path: Path
    ) -> None:
        prov = _make_provider(initialize_raises=RuntimeError("no model"))
        with patch("champi_tts.get_provider", return_value=prov):
            result = _raw(
                await server.call_tool(
                    "synthesize_to_file",
                    {"text": "hi", "output_path": str(tmp_path / "o.wav")},
                )
            )
        assert result["result"] == "error: RuntimeError: no model"
        assert server._state["provider"] is None

    @pytest.mark.asyncio
    async def test_writes_nothing_to_stdout(
        self, server: MagicMock, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        prov = _make_provider()
        with patch("champi_tts.get_provider", return_value=prov):
            await server.call_tool(
                "synthesize_to_file",
                {"text": "hi", "output_path": str(tmp_path / "o.wav")},
            )
        assert capfd.readouterr().out == ""


class TestToolListVoices:
    @pytest.mark.asyncio
    async def test_returns_voices_without_initialising(self, server: MagicMock) -> None:
        prov = _make_provider(voices=["af_bella", "jf_alpha"])
        with patch("champi_tts.get_provider", return_value=prov) as mock_get:
            result = _raw(await server.call_tool("list_voices", {}))
        assert result["result"] == ["af_bella", "jf_alpha"]
        mock_get.assert_called_once_with("kokoro")
        prov.initialize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_returns_error_entry(self, server: MagicMock) -> None:
        prov = _make_provider(list_voices_raises=FileNotFoundError("no dir"))
        with patch("champi_tts.get_provider", return_value=prov):
            result = _raw(await server.call_tool("list_voices", {}))
        assert result["result"] == ["error: FileNotFoundError: no dir"]

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_error_entry(
        self, server: MagicMock
    ) -> None:
        result = _raw(await server.call_tool("list_voices", {"provider": "bad"}))
        assert result["result"][0].startswith("error: ValueError:")


class TestMain:
    def test_redirects_loguru_to_stderr(self) -> None:
        mock_mcp = MagicMock()
        with patch("champi_tts.mcp.server.mcp", mock_mcp):
            from loguru import logger

            with (
                patch.object(logger, "remove") as mock_remove,
                patch.object(logger, "add") as mock_add,
            ):
                from champi_tts.mcp.server import main

                main()

            mock_remove.assert_called_once_with()
            mock_add.assert_called_once_with(sys.stderr)

    def test_calls_run_with_stdio_default(self) -> None:
        mock_mcp = MagicMock()
        with (
            patch("champi_tts.mcp.server.mcp", mock_mcp),
            patch("loguru.logger.remove"),
            patch("loguru.logger.add"),
        ):
            from champi_tts.mcp.server import main

            main()
        mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_sets_host_port_for_sse(self) -> None:
        mock_mcp = MagicMock()
        with (
            patch("champi_tts.mcp.server.mcp", mock_mcp),
            patch("loguru.logger.remove"),
            patch("loguru.logger.add"),
        ):
            from champi_tts.mcp.server import main

            main(transport="sse", host="0.0.0.0", port=9000)
        assert mock_mcp.settings.host == "0.0.0.0"
        assert mock_mcp.settings.port == 9000
        mock_mcp.run.assert_called_once_with(transport="sse")

    def test_default_sse_port_differs_from_champi_stt(self) -> None:
        import inspect

        from champi_tts.mcp.server import main

        assert inspect.signature(main).parameters["port"].default == 8766


class TestMCPInit:
    def test_exports(self) -> None:
        from champi_tts.mcp import create_mcp_server, main, mcp

        assert callable(create_mcp_server)
        assert callable(main)
        assert mcp is not None
