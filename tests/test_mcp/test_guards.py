"""Tests for MCP input guards and the speech concurrency policy."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from champi_tts.mcp.guards import (
    OUTPUT_DIR_ENV,
    SpeechGate,
    resolve_output_path,
)


@pytest.fixture(autouse=True)
def _no_output_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OUTPUT_DIR_ENV, raising=False)


class TestResolveOutputPath:
    @pytest.mark.parametrize(("name", "fmt"), [("a.wav", "wav"), ("b.FLAC", "flac")])
    def test_supported_extension_returns_format(
        self, tmp_path: Path, name: str, fmt: str
    ) -> None:
        path, got = resolve_output_path(str(tmp_path / name))
        assert path == (tmp_path / name).resolve()
        assert got == fmt

    @pytest.mark.parametrize("name", ["a.mp3", "a.sh", "noext"])
    def test_unsupported_extension_raises(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="unsupported output extension"):
            resolve_output_path(str(tmp_path / name))

    def test_existing_file_refused_without_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "a.wav"
        target.write_bytes(b"keep")
        with pytest.raises(FileExistsError, match="overwrite=True"):
            resolve_output_path(str(target))
        assert target.read_bytes() == b"keep"

    def test_existing_file_allowed_with_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "a.wav"
        target.write_bytes(b"old")
        path, _ = resolve_output_path(str(target), overwrite=True)
        assert path == target.resolve()

    def test_directory_refused(self, tmp_path: Path) -> None:
        (tmp_path / "d.wav").mkdir()
        with pytest.raises(IsADirectoryError):
            resolve_output_path(str(tmp_path / "d.wav"), overwrite=True)

    def test_relative_path_placed_under_output_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path))
        path, _ = resolve_output_path("sub/a.wav")
        assert path == (tmp_path / "sub" / "a.wav").resolve()

    def test_absolute_path_inside_output_root_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OUTPUT_DIR_ENV, str(tmp_path))
        path, _ = resolve_output_path(str(tmp_path / "a.wav"))
        assert path == (tmp_path / "a.wav").resolve()

    @pytest.mark.parametrize("escape", ["../a.wav", "/etc/a.wav"])
    def test_path_outside_output_root_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escape: str
    ) -> None:
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.setenv(OUTPUT_DIR_ENV, str(root))
        with pytest.raises(ValueError, match=OUTPUT_DIR_ENV):
            resolve_output_path(escape)

    def test_symlink_out_of_output_root_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "link").symlink_to(outside)
        monkeypatch.setenv(OUTPUT_DIR_ENV, str(root))
        with pytest.raises(ValueError, match=OUTPUT_DIR_ENV):
            resolve_output_path("link/a.wav")


async def _speak(
    gate: SpeechGate,
    name: str,
    log: list[str],
    release: asyncio.Event,
    policy: str = "queue",
    interrupt: AsyncMock | None = None,
) -> None:
    async with gate.turn(policy, interrupt) as turn:
        if turn.preempted:
            log.append(f"{name}:skipped")
            return
        log.append(f"{name}:start")
        await release.wait()
        log.append(f"{name}:end")


class TestSpeechGate:
    @pytest.mark.asyncio
    async def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown speech policy"):
            async with SpeechGate().turn("shout"):
                pass

    @pytest.mark.asyncio
    async def test_queue_policy_never_overlaps(self) -> None:
        gate, log, release = SpeechGate(), [], asyncio.Event()
        tasks = [asyncio.create_task(_speak(gate, name, log, release)) for name in "ab"]
        await asyncio.sleep(0.01)
        assert log == ["a:start"]
        assert gate.busy

        release.set()
        await asyncio.gather(*tasks)
        assert log == ["a:start", "a:end", "b:start", "b:end"]
        assert not gate.busy

    @pytest.mark.asyncio
    async def test_interrupt_policy_stops_active_speech(self) -> None:
        gate, log = SpeechGate(), []
        first_release, second_release = asyncio.Event(), asyncio.Event()
        interrupt = AsyncMock(side_effect=first_release.set)

        first = asyncio.create_task(_speak(gate, "a", log, first_release))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(
            _speak(gate, "b", log, second_release, "interrupt", interrupt)
        )
        await asyncio.sleep(0.01)
        second_release.set()
        await asyncio.gather(first, second)

        interrupt.assert_awaited_once()
        assert log == ["a:start", "a:end", "b:start", "b:end"]

    @pytest.mark.asyncio
    async def test_interrupt_policy_skips_queued_turns(self) -> None:
        gate, log = SpeechGate(), []
        active_release, done = asyncio.Event(), asyncio.Event()
        interrupt = AsyncMock(side_effect=active_release.set)

        active = asyncio.create_task(_speak(gate, "a", log, active_release))
        await asyncio.sleep(0.01)
        queued = asyncio.create_task(_speak(gate, "b", log, done))
        await asyncio.sleep(0.01)
        newest = asyncio.create_task(
            _speak(gate, "c", log, done, "interrupt", interrupt)
        )
        await asyncio.sleep(0.01)
        done.set()
        await asyncio.gather(active, queued, newest)

        assert log == ["a:start", "a:end", "b:skipped", "c:start", "c:end"]

    @pytest.mark.asyncio
    async def test_interrupt_not_called_when_idle(self) -> None:
        gate, interrupt = SpeechGate(), AsyncMock()
        async with gate.turn("interrupt", interrupt) as turn:
            assert not turn.preempted
        interrupt.assert_not_awaited()
