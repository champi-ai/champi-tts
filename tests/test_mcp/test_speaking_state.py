"""Tests for speaking state tracking and MCP progress notifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from champi_tts.mcp.speaking_state import (
    SpeakingPhase,
    SpeakingStateTracker,
    speaking_state_changed,
)


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.report_progress = AsyncMock()
    ctx.info = AsyncMock()
    return ctx


@pytest.fixture()
def received() -> Any:
    events: list[dict[str, Any]] = []

    def _on_change(sender: Any, state: dict[str, Any]) -> None:
        events.append(state)

    speaking_state_changed.connect(_on_change)
    yield events
    speaking_state_changed.disconnect(_on_change)


class TestSpeakingStateTracker:
    def test_starts_idle(self) -> None:
        snap = SpeakingStateTracker().snapshot()
        assert snap["state"] == "idle"
        assert snap["text"] is None
        assert snap["voice"] is None

    @pytest.mark.asyncio
    async def test_set_phase_records_text_and_voice(self) -> None:
        tracker = SpeakingStateTracker()
        await tracker.set_phase(SpeakingPhase.SYNTHESIZING, text="hi", voice="af")
        await tracker.set_phase(SpeakingPhase.SPEAKING)
        snap = tracker.snapshot()
        assert snap == {**snap, "state": "speaking", "text": "hi", "voice": "af"}

    @pytest.mark.asyncio
    async def test_idle_clears_text_and_voice(self) -> None:
        tracker = SpeakingStateTracker()
        await tracker.set_phase(SpeakingPhase.SPEAKING, text="hi", voice="af")
        await tracker.set_phase(SpeakingPhase.IDLE)
        assert tracker.text is None
        assert tracker.voice is None

    @pytest.mark.asyncio
    async def test_publishes_signal(self, received: list[dict[str, Any]]) -> None:
        tracker = SpeakingStateTracker()
        await tracker.set_phase(SpeakingPhase.SPEAKING, text="hi")
        assert received[-1]["state"] == "speaking"
        assert received[-1]["text"] == "hi"

    @pytest.mark.asyncio
    async def test_notifies_context_with_increasing_progress(self) -> None:
        ctx = _ctx()
        tracker = SpeakingStateTracker()
        for phase in (
            SpeakingPhase.SYNTHESIZING,
            SpeakingPhase.SPEAKING,
            SpeakingPhase.IDLE,
        ):
            await tracker.set_phase(phase, ctx, text="hi")
        progress = [c.args[0] for c in ctx.report_progress.await_args_list]
        assert progress == [1.0, 2.0, 3.0]
        assert ctx.report_progress.await_args_list[0].args[1] == 3.0
        messages = [c.args[0] for c in ctx.info.await_args_list]
        assert messages == [
            "champi-tts: synthesizing",
            "champi-tts: speaking",
            "champi-tts: idle",
        ]

    @pytest.mark.asyncio
    async def test_notification_errors_are_suppressed(self) -> None:
        ctx = _ctx()
        ctx.report_progress.side_effect = ValueError("no request")
        ctx.info.side_effect = ValueError("no request")
        tracker = SpeakingStateTracker()
        await tracker.set_phase(SpeakingPhase.SPEAKING, ctx)
        assert tracker.phase is SpeakingPhase.SPEAKING

    @pytest.mark.asyncio
    async def test_session_returns_to_idle_on_error(self) -> None:
        tracker = SpeakingStateTracker()
        with pytest.raises(RuntimeError):
            async with tracker.session("hi", "af"):
                assert tracker.phase is SpeakingPhase.SYNTHESIZING
                raise RuntimeError("boom")
        assert tracker.phase is SpeakingPhase.IDLE


class TestServerIntegration:
    @pytest.fixture()
    def server(self) -> Any:
        pytest.importorskip("mcp")
        from champi_tts.mcp.server import create_mcp_server

        return create_mcp_server()

    @staticmethod
    def _provider(**synth: Any) -> MagicMock:
        prov = MagicMock()
        prov.config.sample_rate = 24000
        prov.initialize = AsyncMock()
        prov.shutdown = AsyncMock()
        prov.synthesize = AsyncMock(
            **({"return_value": np.zeros(240, np.float32)} | synth)
        )
        return prov

    def test_get_speaking_state_registered(self, server: Any) -> None:
        assert "get_speaking_state" in server._tool_manager._tools

    @pytest.mark.asyncio
    async def test_get_speaking_state_idle_by_default(self, server: Any) -> None:
        result = (await server.call_tool("get_speaking_state", {}))[1]
        assert result["state"] == "idle"
        assert result["text"] is None

    @pytest.mark.asyncio
    async def test_get_speaking_state_reflects_tracker(self, server: Any) -> None:
        await server._state["speaking"].set_phase(
            SpeakingPhase.SPEAKING, text="hello", voice="af_bella"
        )
        result = (await server.call_tool("get_speaking_state", {}))[1]
        assert result["state"] == "speaking"
        assert result["text"] == "hello"
        assert result["voice"] == "af_bella"

    @pytest.mark.asyncio
    async def test_synthesize_to_file_is_synthesizing_while_running(
        self, server: Any, tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}

        async def _synth(*args: Any, **kwargs: Any) -> np.ndarray:
            seen.update(server._state["speaking"].snapshot())
            return np.zeros(240, np.float32)

        prov = self._provider(side_effect=_synth)
        with patch("champi_tts.get_provider", return_value=prov):
            await server.call_tool(
                "synthesize_to_file",
                {
                    "text": "hello",
                    "output_path": str(tmp_path / "o.wav"),
                    "voice": "af_bella",
                },
            )
        assert seen["state"] == "synthesizing"
        assert seen["text"] == "hello"
        assert seen["voice"] == "af_bella"
        assert server._state["speaking"].phase is SpeakingPhase.IDLE

    @pytest.mark.asyncio
    async def test_synthesize_failure_returns_to_idle(
        self, server: Any, tmp_path: Path
    ) -> None:
        prov = self._provider(side_effect=RuntimeError("boom"))
        with patch("champi_tts.get_provider", return_value=prov):
            result = (
                await server.call_tool(
                    "synthesize_to_file",
                    {"text": "hi", "output_path": str(tmp_path / "o.wav")},
                )
            )[1]
        assert result["result"] == "error: RuntimeError: boom"
        assert server._state["speaking"].phase is SpeakingPhase.IDLE

    @pytest.mark.asyncio
    async def test_client_receives_progress_and_log_notifications(
        self, server: Any, tmp_path: Path
    ) -> None:
        from mcp.shared.memory import create_connected_server_and_client_session

        progress: list[tuple[float, float | None, str | None]] = []
        logs: list[Any] = []

        async def _on_progress(p: float, total: float | None, msg: str | None) -> None:
            progress.append((p, total, msg))

        async def _on_log(params: Any) -> None:
            logs.append(params.data)

        prov = self._provider()
        with patch("champi_tts.get_provider", return_value=prov):
            async with create_connected_server_and_client_session(
                server, logging_callback=_on_log
            ) as client:
                result = await client.call_tool(
                    "synthesize_to_file",
                    {"text": "hi", "output_path": str(tmp_path / "o.wav")},
                    progress_callback=_on_progress,
                )
        assert not result.isError
        assert [p for p, _, _ in progress] == [1.0, 3.0]
        assert progress[0][2] == "champi-tts: synthesizing"
        assert "champi-tts: synthesizing" in logs
        assert "champi-tts: idle" in logs
