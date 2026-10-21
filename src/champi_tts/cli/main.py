"""
Command-line interface for champi-tts.
"""

import asyncio
import sys
from enum import StrEnum
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from champi_tts import get_provider, get_reader
from champi_tts.providers.kokoro.exceptions import KokoroVoiceError
from champi_tts.providers.kokoro.models import VoiceManager, default_voice_dir

app = typer.Typer(
    name="champi-tts",
    help="Champi TTS - Multi-Provider Text-to-Speech Library",
    add_completion=False,
)
console = Console()

voices_app = typer.Typer(help="Manage Kokoro voice files.")
app.add_typer(voices_app, name="voices")

mcp_app = typer.Typer(help="MCP (Model Context Protocol) server commands.")
app.add_typer(mcp_app, name="mcp")

_MCP_INSTALL_HINT = (
    "Error: mcp package is not installed. Install with: pip install 'champi-tts[mcp]'"
)


@app.command()
def synthesize(
    text: str = typer.Argument(..., help="Text to synthesize"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output audio file path"
    ),
    voice: str | None = typer.Option(None, "--voice", "-v", help="Voice to use"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Speech speed (0.5-2.0)"),
    provider: str = typer.Option(
        "kokoro", "--provider", "-p", help="TTS provider to use"
    ),
    play: bool = typer.Option(
        True, "--play/--no-play", help="Play audio after synthesis"
    ),
):
    """Synthesize text to speech."""

    async def run():
        # Get provider
        tts_provider = get_provider(provider)  # type: ignore[arg-type]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize
            task = progress.add_task("Initializing TTS provider...", total=None)
            await tts_provider.initialize()
            progress.remove_task(task)

            # Synthesize
            task = progress.add_task(f"Synthesizing: {text[:50]}...", total=None)
            audio = await tts_provider.synthesize(text, voice=voice, speed=speed)
            progress.remove_task(task)

        # Save if output specified
        if output:
            from champi_tts.core.audio import save_audio

            await save_audio(audio, output, sample_rate=tts_provider.config.sample_rate)
            console.print(f"✓ Audio saved to: [cyan]{output}[/cyan]")

        # Play if requested
        if play:
            from champi_tts.core.audio import AudioPlayer

            player = AudioPlayer(sample_rate=tts_provider.config.sample_rate)
            console.print("▶ Playing audio...")
            await player.play(audio, blocking=True)
            console.print("✓ Playback complete")

        # Shutdown
        await tts_provider.shutdown()

    asyncio.run(run())


@app.command()
def read(
    file_path: Path | None = typer.Argument(None, help="Text file to read"),
    text: str | None = typer.Option(
        None, "--text", "-t", help="Text to read (alternative to file)"
    ),
    voice: str | None = typer.Option(None, "--voice", "-v", help="Voice to use"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Speech speed (0.5-2.0)"),
    provider: str = typer.Option(
        "kokoro", "--provider", "-p", help="TTS provider to use"
    ),
    show_ui: bool = typer.Option(False, "--show-ui", help="Show visual UI indicator"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode with controls"
    ),
):
    """Read text from file or string with pause/resume/stop controls."""

    if not file_path and not text:
        console.print("[red]Error: Must provide either file_path or --text[/red]")
        raise typer.Exit(1)

    async def run():
        # Get reader
        reader = get_reader(
            provider,  # type: ignore[arg-type]
            show_ui=show_ui,
            default_voice=voice or "am_adam",
            default_speed=speed,
        )

        # Initialize provider
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing TTS provider...", total=None)
            await reader.provider.initialize()
            progress.remove_task(task)

        # Read text
        try:
            if text:
                console.print(f"📖 Reading: [cyan]{text[:100]}...[/cyan]")
                await reader.read_text(text, voice=voice)
            elif file_path:
                console.print(f"📖 Reading file: [cyan]{file_path}[/cyan]")
                await reader.read_file(file_path, voice=voice)

            console.print("✓ Reading complete")

        except KeyboardInterrupt:
            console.print("\n⏹ Stopping...")
            await reader.stop()

        finally:
            await reader.provider.shutdown()

    async def run_interactive():
        """Run in interactive mode with keyboard controls."""
        import sys
        import termios
        import threading
        import tty

        # Get reader
        reader = get_reader(
            provider,  # type: ignore[arg-type]
            show_ui=show_ui,
            default_voice=voice or "am_adam",
            default_speed=speed,
        )

        # Initialize provider
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing TTS provider...", total=None)
            await reader.provider.initialize()
            progress.remove_task(task)

        console.print("\n[bold green]Interactive Mode Controls:[/bold green]")
        console.print("  [cyan]SPACE[/cyan] - Pause/Resume")
        console.print("  [cyan]S[/cyan]     - Stop")
        console.print("  [cyan]Q[/cyan]     - Quit\n")

        # Start reading in background
        reading_task = None
        if text:
            console.print(f"📖 Reading: [cyan]{text[:100]}...[/cyan]")
            reading_task = asyncio.create_task(reader.read_text(text, voice=voice))
        elif file_path:
            console.print(f"📖 Reading file: [cyan]{file_path}[/cyan]")
            reading_task = asyncio.create_task(reader.read_file(file_path, voice=voice))

        # Handle keyboard input
        def get_key():
            """Get a single keypress."""
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        key_queue: asyncio.Queue[str] = asyncio.Queue()

        def key_listener():
            """Listen for keypresses in a separate thread."""
            while True:
                key = get_key()
                asyncio.run_coroutine_threadsafe(
                    key_queue.put(key), asyncio.get_event_loop()
                )
                if key.lower() == "q":
                    break

        # Start key listener thread
        listener_thread = threading.Thread(target=key_listener, daemon=True)
        listener_thread.start()

        try:
            # Process key commands
            while True:
                key = await key_queue.get()

                if key == " ":
                    if reader.state.value == "reading":
                        await reader.pause()
                        console.print("⏸ [yellow]Paused[/yellow]")
                    elif reader.state.value == "paused":
                        await reader.resume()
                        console.print("▶ [green]Resumed[/green]")

                elif key.lower() == "s":
                    await reader.stop()
                    console.print("⏹ [red]Stopped[/red]")
                    break

                elif key.lower() == "q":
                    console.print("👋 Quitting...")
                    break

                # Check if reading task completed
                if reading_task and reading_task.done():
                    console.print("✓ Reading complete")
                    break

        except KeyboardInterrupt:
            console.print("\n⏹ Stopping...")
            await reader.stop()

        finally:
            if reading_task and not reading_task.done():
                reading_task.cancel()
            await reader.provider.shutdown()

    if interactive:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run())


@app.command()
def list_voices(
    provider: str = typer.Option("kokoro", "--provider", "-p", help="TTS provider"),
):
    """List available voices for a provider."""

    async def run():
        tts_provider = get_provider(provider)  # type: ignore[arg-type]
        await tts_provider.initialize()

        voices = await tts_provider.list_voices()

        console.print(f"\n[bold]Available voices for {provider}:[/bold]")
        for voice in voices:
            console.print(f"  • {voice}")

        await tts_provider.shutdown()

    asyncio.run(run())


@app.command()
def test_ui():
    """Test the TTS UI indicator."""
    from champi_tts.ui import run_standalone

    console.print("🎨 Running TTS UI test...")
    console.print("Press Ctrl+C to exit")
    run_standalone()


@app.command()
def version():
    """Show champi-tts version."""
    from champi_tts import __version__

    console.print(f"champi-tts version [cyan]{__version__}[/cyan]")


VOICE_DIR_OPTION = typer.Option(
    None,
    "--voice-dir",
    "--cache-dir",
    help="Override the voice directory (default: $KOKORO_VOICE_DIR or "
    "~/.mcp_champi/kokoro/voices).",
)


@voices_app.command("list")
def voices_list(voice_dir: Path | None = VOICE_DIR_OPTION) -> None:
    """List voice files available in the voice directory."""
    voices = VoiceManager.list_voices(str(voice_dir or default_voice_dir()))
    if not voices:
        console.print(
            "[yellow]No voices found.[/yellow] "
            "Run 'champi-tts voices download <name>' to fetch one."
        )
        raise typer.Exit(0)
    console.print("[bold]Available voices:[/bold]")
    for voice in voices:
        console.print(f"  • {voice}")


@voices_app.command("cache-dir")
def voices_cache_dir(voice_dir: Path | None = VOICE_DIR_OPTION) -> None:
    """Print the voice directory path."""
    console.print(str(voice_dir or default_voice_dir()))


@voices_app.command("download")
def voices_download(
    name: str = typer.Argument(..., help="Voice name, e.g. am_adam"),
    voice_dir: Path | None = VOICE_DIR_OPTION,
    force: bool = typer.Option(False, "--force", help="Re-download if present."),
) -> None:
    """Download a voice from hexgrad/Kokoro-82M into the voice directory."""
    try:
        path = VoiceManager.download_voice(
            name, str(voice_dir or default_voice_dir()), force=force
        )
    except KokoroVoiceError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    console.print(f"✓ Voice saved to: [cyan]{path}[/cyan]")


class MCPTransport(StrEnum):
    """Transports supported by the MCP server."""

    stdio = "stdio"
    sse = "sse"


@mcp_app.command("serve")
def mcp_serve(
    transport: MCPTransport = typer.Option(
        MCPTransport.stdio,
        "--transport",
        help="Transport protocol: stdio for MCP hosts (Claude Desktop) or sse "
        "for HTTP clients.",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        help="Bind address for the SSE HTTP server (SSE transport only).",
    ),
    port: int = typer.Option(
        8766,
        "--port",
        help="TCP port for the SSE HTTP server (SSE transport only).",
    ),
) -> None:
    """Start the MCP server.

    In stdio mode (default) the server reads JSON-RPC messages from stdin and
    writes responses to stdout, the format expected by Claude Desktop and other
    MCP hosts. In sse mode it starts an HTTP server; connect your MCP client
    to http://<host>:<port>/sse.

    Requires the 'mcp' extra: pip install 'champi-tts[mcp]'

    Environment variables:

    * CHAMPI_TTS_PROVIDER: TTS provider used by the tools (default: kokoro)

    * CHAMPI_OUTPUT_DEVICE_INDEX: audio output device index for playback

    * KOKORO_*: Kokoro provider settings (KOKORO_MODEL_DIR, KOKORO_VOICE_DIR,
      KOKORO_DEFAULT_VOICE, KOKORO_DEFAULT_SPEED, KOKORO_USE_GPU, ...)
    """
    try:
        from champi_tts.mcp.server import MCP_AVAILABLE
        from champi_tts.mcp.server import main as _main
    except ImportError:
        typer.echo(_MCP_INSTALL_HINT, err=True)
        raise typer.Exit(1) from None

    if not MCP_AVAILABLE:
        typer.echo(_MCP_INSTALL_HINT, err=True)
        raise typer.Exit(1)

    _main(transport=transport.value, host=host, port=port)


def main():
    """Main CLI entry point."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    app()


if __name__ == "__main__":
    main()
