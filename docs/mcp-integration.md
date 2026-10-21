# MCP Integration

Champi TTS exposes text-to-speech capabilities over the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/), letting
LLM hosts such as Claude Desktop and Claude Code speak text aloud or write it
to an audio file by calling tools directly via stdio.

## Prerequisites

- **Python 3.12 or 3.13** and [uv](https://docs.astral.sh/uv/) (provides `uvx`).
- **PortAudio** for speaker playback (`speak_tool`, `speak_streaming_tool`,
  `list_audio_devices_tool`):

    ```bash
    sudo apt install portaudio19-dev   # Debian/Ubuntu
    ```

    The macOS and Windows `sounddevice` wheels bundle PortAudio; if playback
    still fails on macOS, `brew install portaudio`. Without PortAudio the server still starts and `synthesize_to_file` still
    works; the playback tools return an `error: ImportError: ...` string.

- **The `mcp` and `kokoro` extras.** `mcp` provides the server, `kokoro` the
  default provider (it pulls in `torch`, so the first install takes a while).

### Kokoro model and voices

Nothing has to be downloaded by hand. On the first synthesis call the Kokoro
provider downloads the model (`kokoro-v1_0.pth`, about 85 MB, plus
`config.json`) and the default voice (`am_adam`). The first call is therefore
slow; later calls reuse the files on disk.

| Files | Default location | Override |
|---|---|---|
| Model | `~/.mcp_champi/kokoro/models` (`%APPDATA%\mcp_champi\kokoro\models` on Windows) | — |
| Voices | `~/.mcp_champi/kokoro/voices` (`%APPDATA%\mcp_champi\kokoro\voices` on Windows) | `KOKORO_VOICE_DIR` |

Fetch additional voices ahead of time with the CLI, for example:

```bash
champi-tts voices download af_bella
champi-tts voices list
```

## Claude Desktop / Claude Code Setup

For Claude Desktop, add a server entry to your `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

For Claude Code, put the same `mcpServers` block in a `.mcp.json` file at the
root of your project (see [Example `.mcp.json`](#example-mcpjson)), or register
it from the command line with `claude mcp add`.

### Production (release wheel)

Replace `X.Y.Z` with the version of the
[latest release](https://github.com/champi-ai/champi-tts/releases/latest):

```json
{
  "mcpServers": {
    "champi-tts": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "champi-tts[mcp,kokoro] @ https://github.com/champi-ai/champi-tts/releases/download/vX.Y.Z/champi_tts-X.Y.Z-py3-none-any.whl",
        "champi-tts",
        "mcp",
        "serve"
      ]
    }
  }
}
```

`--python 3.12` is there because the `curated-tokenizers` override that lets
the `kokoro` extra install on Python 3.13 is only applied by uv project
commands, not by `uvx` (see
[Kokoro on Python 3.13](https://github.com/champi-ai/champi-tts#kokoro-on-python-313)).
Drop it if your default Python is 3.12.

The same command works in a terminal:

```bash
uvx --python 3.12 --from "champi-tts[mcp,kokoro] @ https://github.com/champi-ai/champi-tts/releases/download/vX.Y.Z/champi_tts-X.Y.Z-py3-none-any.whl" champi-tts mcp serve
```

If champi-tts is installed permanently with the `mcp` and `kokoro` extras, use
the `champi-tts-mcp` entry point instead:

```json
{
  "mcpServers": {
    "champi-tts": {
      "command": "champi-tts-mcp"
    }
  }
}
```

### Development (`uv run`)

Use this form when working directly from a source checkout (after
`uv sync --extra mcp --extra kokoro`):

```json
{
  "mcpServers": {
    "champi-tts": {
      "command": "uv",
      "args": ["run", "champi-tts", "mcp", "serve"],
      "cwd": "/path/to/champi-tts"
    }
  }
}
```

Replace `/path/to/champi-tts` with the absolute path to your local clone.

### With environment variables

Add an `env` block to change the defaults:

```json
{
  "mcpServers": {
    "champi-tts": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "champi-tts[mcp,kokoro] @ https://github.com/champi-ai/champi-tts/releases/download/vX.Y.Z/champi_tts-X.Y.Z-py3-none-any.whl",
        "champi-tts",
        "mcp",
        "serve"
      ],
      "env": {
        "CHAMPI_TTS_PROVIDER": "kokoro",
        "CHAMPI_OUTPUT_DEVICE_INDEX": "3",
        "KOKORO_VOICE_DIR": "/home/me/kokoro-voices",
        "CHAMPI_TTS_OUTPUT_DIR": "/home/me/tts-output"
      }
    }
  }
}
```

| Variable | Effect |
|---|---|
| `CHAMPI_TTS_PROVIDER` | Provider used when a tool call omits `provider` (default `kokoro`) |
| `CHAMPI_OUTPUT_DEVICE_INDEX` | Output device used when a playback call omits `device_index` (default: system default device). Find indexes with `list_audio_devices_tool` |
| `KOKORO_VOICE_DIR` | Directory voices are read from and downloaded to |
| `CHAMPI_TTS_OUTPUT_DIR` | Confines `synthesize_to_file`: relative paths are placed under it and paths resolving outside it are rejected |

To use a voice other than `am_adam`, pass `voice` in the tool call (for
example `"voice": "af_bella"`). The MCP server builds its Kokoro provider with
the default configuration, so `KOKORO_DEFAULT_VOICE` and the other `KOKORO_*`
settings read by `KokoroConfig.from_env()` are not applied to it yet
([#122](https://github.com/champi-ai/champi-tts/issues/122)).

After editing the file, restart Claude Desktop (or reload the Claude Code
session) for the change to take effect.

### Example `.mcp.json`

A project-level `.mcp.json` that runs champi-tts next to champi-stt:

```json
{
  "mcpServers": {
    "champi-stt": {
      "command": "uvx",
      "args": [
        "--from",
        "champi-stt[mcp] @ https://github.com/champi-ai/champi-stt/releases/download/vX.Y.Z/champi_stt-X.Y.Z-py3-none-any.whl",
        "champi-stt",
        "mcp",
        "serve"
      ]
    },
    "champi-tts": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "champi-tts[mcp,kokoro] @ https://github.com/champi-ai/champi-tts/releases/download/vX.Y.Z/champi_tts-X.Y.Z-py3-none-any.whl",
        "champi-tts",
        "mcp",
        "serve"
      ]
    }
  }
}
```

Each project has its own release versions, so fill in `X.Y.Z` separately for
each wheel URL.

## Smoke-test

Verify the stdio handshake without a full MCP client by piping an
`initialize` request directly to the server:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | champi-tts mcp serve 2>/dev/null
```

(Use the full `uvx ... champi-tts mcp serve` or `uv run champi-tts mcp serve`
command if `champi-tts` is not on your `PATH`.)

Expected output: a JSON-RPC response whose `"result"` contains `serverInfo`
with `"name": "champi-tts"`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "experimental": {},
      "prompts": {"listChanged": false},
      "resources": {"subscribe": false, "listChanged": false},
      "tools": {"listChanged": false}
    },
    "serverInfo": {
      "name": "champi-tts",
      "version": "1.30.0"
    }
  }
}
```

`serverInfo.version` is the version of the installed `mcp` SDK, so it may
differ. The `2>/dev/null` suppresses loguru output, which is written to stderr
and never touches the JSON-RPC stream.

Then, from the MCP host, ask it to call `speak_tool` with
`{"text": "Hello from champi-tts"}`. The first call downloads the model and
voice, then plays the sentence and returns playback metrics such as
`duration_seconds`, `generation_time_seconds`, `voice` and `interrupted`.

## Available Tools

| Tool | Description |
|---|---|
| `list_providers` | Return the names of all available TTS providers |
| `get_provider_status` | Return health and model information for a named provider |
| `list_voices` | List the voices available for a provider (does not load the model) |
| `synthesize_to_file` | Synthesize text to a `.wav`, `.flac` or `.ogg` file and return the written path |
| `speak_tool` | Synthesize text and play it on a speaker; return playback metrics |
| `speak_streaming_tool` | Synthesize text in chunks and play each as it arrives (lower time-to-first-audio) |
| `stop_speaking_tool` | Stop any active speaker playback |
| `list_audio_devices_tool` | List available audio output devices and their indexes |
| `get_speaking_state` | Return the current speaking state (`idle`, `synthesizing`, `speaking`) |

Tools report failures as an `"error: <ExceptionType>: <message>"` string (or
an error entry/dict) instead of raising, so the host always gets a readable
result. `synthesize_to_file` refuses to overwrite an existing file unless
`overwrite` is `true`.

## Speaking state and progress notifications

The champi-tts MCP server tracks what it is saying so that hosts and sibling
services can react to it — champi-stt can ignore the microphone while TTS is
speaking (echo suppression) and champi-imgui can show a speaking indicator.

### States

| State          | Meaning                                         |
|----------------|-------------------------------------------------|
| `idle`         | Nothing is being synthesized or played          |
| `synthesizing` | Text is being turned into audio                 |
| `speaking`     | Audio is playing on the output device           |

### `get_speaking_state` tool

Returns the current state:

```json
{
  "state": "synthesizing",
  "text": "Hello there",
  "voice": "af_bella",
  "updated_at": 1761040000.12
}
```

`text` and `voice` are `null` when `idle`; `updated_at` is the UNIX time of
the last transition.

### Progress notifications

Tools that synthesize or play audio send a `notifications/progress` message
(when the client supplied a `progressToken`) and a `notifications/message`
log entry on every transition:

| Transition     | `progress` / `total` | message                    |
|----------------|----------------------|----------------------------|
| `synthesizing` | 1 / 3                | `champi-tts: synthesizing` |
| `speaking`     | 2 / 3                | `champi-tts: speaking`     |
| `idle`         | 3 / 3                | `champi-tts: idle`         |

`synthesize_to_file` reports `synthesizing` then `idle`. Playback tools also
report `speaking` right before audio starts. Notifications are best-effort:
a client that does not request progress still gets the tool result.

### Gating champi-stt while TTS speaks

**Separate processes (typical MCP host setup).** Poll `get_speaking_state` on
the champi-tts server and skip `listen_*` calls on champi-stt while the state
is `synthesizing` or `speaking`, or call it after the speak tool returns.

**Same process.** Subscribe to the in-process blinker signal instead of
polling:

```python
from champi_tts.mcp.speaking_state import speaking_state_changed


def on_speaking_state(sender, state):
    mic_muted = state["state"] != "idle"


speaking_state_changed.connect(on_speaking_state)
```

The signal is sent with `sender=<SpeakingStateTracker>` and `state=<dict>`
(the same shape `get_speaking_state` returns).

### Adding a tool that speaks

Tools registered in `create_mcp_server()` share the tracker in
`state["speaking"]`. Take a `ctx: Context | None = None` parameter and wrap
the work in a session:

```python
async with state["speaking"].session(text, voice, ctx) as speaking:
    audio = await provider.synthesize(text, voice=voice)
    await speaking.set_phase(SpeakingPhase.SPEAKING, ctx)
    await play(audio)
```

The session always returns to `idle`, including on errors and interrupts.

## SSE Transport

For clients that prefer HTTP/SSE over stdio:

```bash
champi-tts mcp serve --transport sse --port 8766
```

Connect your MCP client to `http://localhost:8766/sse`. The default port is
8766, so champi-tts can run next to champi-stt (8765). Use `--host` to bind
to an address other than `localhost`.
