#!/usr/bin/env python3
"""
Kokoro TTS Console - Textual UI
A modern terminal-based TTS interface using Textual framework
"""

from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header, RichLog, SelectionList, Static, TextArea

from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.events import TTSSignalManager
from champi_tts.providers.kokoro.provider import KokoroProvider


class TTSStatusWidget(Static):
    """Widget displaying TTS status and settings"""

    def __init__(self):
        super().__init__()
        self.current_voice = "Default Voice"
        self.current_language = "English"
        self.is_speaking = False

    def compose(self) -> ComposeResult:
        yield Static(self.get_status_content(), id="status-content")

    def get_status_content(self) -> Text:
        """Generate status display content"""
        status = "🔊 SPEAKING" if self.is_speaking else "📖 IDLE"

        content = Text()
        content.append("🎤 KOKORO TTS STATUS\n", style="bold cyan")
        content.append("=" * 25 + "\n")
        content.append(
            f"Status: {status}\n",
            style="bold green" if not self.is_speaking else "bold red",
        )
        content.append(f"Voice: {self.current_voice}\n")
        content.append(f"Language: {self.current_language}\n")
        content.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")

        return content

    def update_status(
        self,
        is_speaking: bool | None = None,
        voice: str | None = None,
        language: str | None = None,
    ):
        """Update status information"""
        if is_speaking is not None:
            self.is_speaking = is_speaking
        if voice:
            self.current_voice = voice
        if language:
            self.current_language = language

        # Update the display
        status_widget = self.query_one("#status-content")
        status_widget.renderable = self.get_status_content()


class TTSSignalsWidget(RichLog):
    """Widget displaying TTS signals and events"""

    def __init__(self):
        super().__init__(max_lines=100, highlight=True, markup=True)
        self.add_demo_signals()

    def add_demo_signals(self):
        """Add initial demo signals"""
        demo_signals = [
            "[dim]00:54:20[/dim] [green]LIFECYCLE[/green]   INITIALIZED",
            "[dim]00:54:21[/dim] [blue]MODEL[/blue]       LOADED",
            "[dim]00:54:22[/dim] [yellow]PROCESSING[/yellow]  READY",
        ]

        for signal in demo_signals:
            self.write(signal)

    def add_signal(self, signal_type: str, message: str):
        """Add a new signal entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color coding for different signal types
        colors = {
            "TTS": "red",
            "CONSOLE": "cyan",
            "LIFECYCLE": "green",
            "MODEL": "blue",
            "PROCESSING": "yellow",
        }

        color = colors.get(signal_type, "white")
        formatted = f"[dim]{timestamp}[/dim] [bold {color}]{signal_type:<10}[/bold {color}] {message}"
        self.write(formatted)


class ReadingDisplayWidget(RichLog):
    """Widget showing the text being read with progress (scrollable)"""

    def __init__(self):
        super().__init__(highlight=True, markup=True)
        self.reading_text = ""
        self.reading_position = 0
        self.is_speaking = False
        self.show_initial_content()

    def show_initial_content(self):
        """Show initial help content"""
        self.write("[bold green]📖 READING VIEW[/bold green]")
        self.write("=" * 50)
        self.write("")
        self.write("No text loaded.")
        self.write("")
        self.write("Type text and press Enter to start TTS.")
        self.write("Type '/' for commands.")

    def set_text(self, text: str):
        """Set the text to be read"""
        self.reading_text = text
        self.reading_position = 0
        self.is_speaking = False
        self.update_display()

    def update_progress(self, position: int, is_speaking: bool):
        """Update reading progress"""
        self.reading_position = position
        self.is_speaking = is_speaking
        self.update_display()

    def update_display(self):
        """Update the display"""
        self.clear()

        if not self.reading_text:
            self.show_initial_content()
            return

        # Show header
        self.write("[bold green]📖 READING VIEW[/bold green]")
        self.write("=" * 50)
        self.write("")

        if self.is_speaking and self.reading_position > 0:
            # Show progress with highlighting
            before = self.reading_text[: self.reading_position]
            current_word_end = min(
                len(self.reading_text),
                self.reading_text.find(" ", self.reading_position) + 1,
            )
            if current_word_end == 0:  # No space found
                current_word_end = len(self.reading_text)

            current = self.reading_text[self.reading_position : current_word_end]
            after = self.reading_text[current_word_end:]

            # Split text into lines and format each line
            full_text_with_highlight = (
                before
                + f"[bold red on yellow]{current}[/bold red on yellow]"
                + f"[dim]{after}[/dim]"
            )

            # Write the text with proper line breaks
            lines = full_text_with_highlight.split("\n")
            for line in lines:
                self.write(line)
        else:
            # Show full text
            lines = self.reading_text.split("\n")
            for line in lines:
                self.write(line)


class CommandsWidget(RichLog):
    """Widget showing available commands with filtering"""

    def __init__(self, **kwargs):
        super().__init__(highlight=True, markup=True, **kwargs)
        self.commands = [
            ("/clear", "Clear current text"),
            ("/stop", "Stop current speech"),
            ("/voice", "Choose voice"),
            ("/lang", "Choose language"),
            ("/url", "Load text from URL"),
            ("/paste", "Enter paste mode"),
            ("/write", "Write text to file"),
            ("/load", "Load text from file"),
            ("/quit", "Exit console"),
        ]
        self.filtered_commands = self.commands[:]
        self.selected_index = 0
        self.show_commands()

    def show_commands(self, filter_text=""):
        """Show commands, optionally filtered"""
        self.clear()

        # Filter commands based on input
        if filter_text and filter_text.startswith("/"):
            self.filtered_commands = [
                cmd for cmd in self.commands if cmd[0].startswith(filter_text.lower())
            ]
        else:
            self.filtered_commands = self.commands[:]

        # Reset selection if out of bounds
        if self.selected_index >= len(self.filtered_commands):
            self.selected_index = 0

        # Display commands with highlighting
        for i, (cmd, desc) in enumerate(self.filtered_commands):
            if i == self.selected_index:
                # Highlight selected command
                self.write(f"[bold white on blue]► {cmd}[/bold white on blue] - {desc}")
            else:
                self.write(f"  [cyan]{cmd}[/cyan] - {desc}")

    def move_selection(self, direction: int):
        """Move selection up (-1) or down (1)"""
        if not self.filtered_commands:
            return

        self.selected_index = (self.selected_index + direction) % len(
            self.filtered_commands
        )
        self.show_commands()

    def get_selected_command(self):
        """Get the currently selected command"""
        if self.filtered_commands and 0 <= self.selected_index < len(
            self.filtered_commands
        ):
            return self.filtered_commands[self.selected_index][0]
        return None


class VoiceSelectionWidget(SelectionList):
    """Widget for voice selection"""

    def __init__(self, **kwargs):
        voices = [
            "Default Voice",
            "Female Voice",
            "Male Voice",
            "Robotic Voice",
            "Gentle Voice",
        ]

        options = [(voice, voice) for voice in voices]
        super().__init__(*options, **kwargs)


class LanguageSelectionWidget(SelectionList):
    """Widget for language selection"""

    def __init__(self, **kwargs):
        languages = ["English", "Spanish", "French", "German", "Japanese", "Chinese"]

        options = [(lang, lang) for lang in languages]
        super().__init__(*options, **kwargs)


class KokoroTTSApp(App[None]):
    """Main Textual app for Kokoro TTS Console"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 4;
        grid-gutter: 1;
        grid-rows: 1fr 2fr 1fr auto;
        grid-columns: 1fr 1fr;
    }

    #status {
        column-span: 1;
        row-span: 1;
        border: solid $primary;
        padding: 1;
    }

    #signals {
        column-span: 1;
        row-span: 1;
        border: solid $primary;
        padding: 1;
    }

    #reading {
        column-span: 2;
        row-span: 1;
        border: solid $success;
        padding: 1;
        overflow-y: scroll;
    }

    #input-section {
        column-span: 2;
        row-span: 1;
        border: solid $warning;
        padding: 1;
    }

    #commands, #voices, #languages {
        height: 10;
        border: solid $secondary;
        margin-top: 1;
    }

    TextArea {
        margin-bottom: 1;
        min-height: 3;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "escape", "Cancel/Back"),
        Binding("enter", "submit_text", "Submit Text", priority=True),
        ("f1", "toggle_commands", "Commands"),
        ("f2", "select_voice", "Voice"),
        ("f3", "select_language", "Language"),
    ]

    def __init__(self):
        super().__init__()
        self.mode = "normal"  # normal, commands, voice_select, lang_select
        self.is_speaking = False
        self.provider: KokoroProvider | None = (
            None  # Will hold the initialized KokoroProvider
        )
        self.signal_manager = TTSSignalManager()  # Real signal manager from kokoro_svc

    @work(thread=True)
    def start_tts_provider(self):
        """Initialize and store the TTS provider for use in the console"""
        import asyncio

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Load environment from .env.kokoro if it exists
            env_file = Path(__file__).parent.parent.parent / ".env.kokoro"
            if env_file.exists():
                try:
                    from dotenv import load_dotenv

                    load_dotenv(env_file)
                except ImportError:
                    pass  # Continue without dotenv

            # Create config
            config = KokoroConfig()
            config.from_env()  # Load from environment variables

            # Create provider
            self.provider = KokoroProvider(config)

            # Initialize async operations in thread loop
            loop.run_until_complete(self.provider.initialize())
            info = loop.run_until_complete(self.provider.get_info())

            # List voices and update UI from thread
            voices = self.provider.list_voices()
            if voices:
                # Get English voice for default
                english_voices = [
                    v for v in voices if v.startswith(("af_", "am_", "bf_", "bm_"))
                ]
                default_voice = english_voices[0] if english_voices else voices[0]

                # Update status widget from thread using call_from_thread
                def update_status():
                    status_widget = self.query_one(TTSStatusWidget)
                    status_widget.update_status(voice=default_voice)

                self.call_from_thread(update_status)

            return info
        finally:
            loop.close()

    def initialize_tts_provider(self):
        """Initialize the TTS provider (normal method)"""
        # Get signals widget reference
        signals_widget = self.query_one(TTSSignalsWidget)

        try:
            signals_widget.add_signal("TTS", "INITIALIZING_PROVIDER")
            self.start_tts_provider()

        except Exception as e:
            signals_widget.add_signal("TTS", f"PROVIDER_INITIALIZATION_FAILED: {e!s}")

    def compose(self) -> ComposeResult:
        yield Header()

        # Top row - Status and Signals
        with Container(id="status"):
            yield TTSStatusWidget()

        with Container(id="signals"):
            yield TTSSignalsWidget()

        # Middle row - Reading display
        with Container(id="reading"):
            yield ReadingDisplayWidget()

        # Bottom row - Input and commands
        with Container(id="input-section"):
            yield TextArea(text="", id="text-input")
            yield CommandsWidget(id="commands", classes="hidden")
            yield VoiceSelectionWidget(id="voices", classes="hidden")
            yield LanguageSelectionWidget(id="languages", classes="hidden")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app"""
        self.title = "🎤 Kokoro TTS Console"
        self.sub_title = "Modern Terminal TTS Interface"

        # Connect to real TTSSignalManager from kokoro_svc
        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal("DEBUG", "Connecting signals")
        self.setup_signal_handlers()
        # Focus on text input
        self.query_one("#text-input").focus()
        self.initialize_tts_provider()

    # Connect to lifecycle events
    def on_lifecycle_event(self, sender, **kwargs):
        signals_widget = self.query_one(TTSSignalsWidget)
        kwargs.get("event_type", "UNKNOWN")
        sub_event = kwargs.get("sub_event", "")
        data = kwargs.get("data", {})

        message = f"{sub_event}"
        if data:
            message += f" - {data}"

        signals_widget.add_signal("LIFECYCLE", message)

    # Connect to model events
    def on_model_event(self, sender, **kwargs):
        signals_widget = self.query_one(TTSSignalsWidget)
        kwargs.get("event_type", "UNKNOWN")
        sub_event = kwargs.get("sub_event", "")
        data = kwargs.get("data", {})

        message = f"{sub_event}"
        if data:
            message += f" - {data}"

        signals_widget.add_signal("MODEL", message)

    # Connect to processing events
    def on_processing_event(self, sender, **kwargs):
        signals_widget = self.query_one(TTSSignalsWidget)
        event_type = kwargs.get("event_type", "UNKNOWN")
        sub_event = kwargs.get("sub_event", "")
        data = kwargs.get("data", {})

        # Handle TTS requests through processing events
        if event_type == "TTS_REQUEST" and sub_event == "TEXT_TO_SPEECH":
            self.handle_tts_request(data)

        message = f"{sub_event}"
        if data:
            message += f" - {data}"

        signals_widget.add_signal("PROCESSING", message)

    @work(thread=True)
    def handle_tts_request(self, data):
        """Handle TTS request in a separate thread to avoid blocking UI"""
        import asyncio

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            text = data.get("text", "")
            voice = data.get("voice", "am_adam")
            lang_code = data.get("lang_code", "a")

            if self.provider and text:
                # Run TTS in this thread's loop
                _success, _metrics = loop.run_until_complete(
                    self.provider.text_to_speech(
                        message=text, voice=voice, lang_code=lang_code, play_audio=True
                    )
                )

                # Update UI from thread using call_from_thread
                def update_finished():
                    self.is_speaking = False
                    status_widget = self.query_one(TTSStatusWidget)
                    status_widget.update_status(is_speaking=False)
                    signals_widget = self.query_one(TTSSignalsWidget)
                    signals_widget.add_signal("TTS", "SPEECH_FINISHED")
                    reading_widget = self.query_one(ReadingDisplayWidget)
                    reading_widget.update_progress(len(text), False)

                self.call_from_thread(update_finished)
        finally:
            loop.close()

    # Connect to telemetry events
    def on_telemetry_event(self, sender, **kwargs):
        signals_widget = self.query_one(TTSSignalsWidget)
        kwargs.get("event_type", "UNKNOWN")
        sub_event = kwargs.get("sub_event", "")
        data = kwargs.get("data", {})

        message = f"{sub_event}"
        if data:
            message += f" - {data}"

        signals_widget.add_signal("TELEMETRY", message)

    def setup_signal_handlers(self):
        """Connect to real TTSSignalManager signals"""
        self.signal_manager.lifecycle.connect(self.on_lifecycle_event)
        self.signal_manager.model.connect(self.on_model_event)
        self.signal_manager.processing.connect(self.on_processing_event)
        self.signal_manager.telemetry.connect(self.on_telemetry_event)

    def on_text_area_changed(self, event) -> None:
        """Handle text area changes"""
        text = event.text_area.text

        # Auto-show commands when text starts with '/' and filter them
        if text.startswith("/") and self.mode == "normal":
            self.mode = "commands"
            commands_widget = self.query_one("#commands", CommandsWidget)
            commands_widget.remove_class("hidden")
            commands_widget.show_commands(text)
        elif not text.startswith("/") and self.mode == "commands":
            self.mode = "normal"
            self.query_one("#commands").add_class("hidden")
        elif self.mode == "commands" and text.startswith("/"):
            # Update filtering as user types
            commands_widget = self.query_one("#commands", CommandsWidget)
            commands_widget.show_commands(text)

        self.query_one("#text-input").focus()

    def on_key(self, event) -> None:
        """Handle key events - Enter to submit, Arrow keys for command selection"""
        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal(
            "DEBUG", f"KEY_PRESSED: {event.key} in mode: {self.mode}"
        )

        # Handle commands mode navigation
        if self.mode == "commands":
            commands_widget = self.query_one("#commands", CommandsWidget)

            if event.key == "up":
                commands_widget.move_selection(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                commands_widget.move_selection(1)
                event.prevent_default()
                event.stop()
            elif event.key == "enter":
                # Type selected command in input and return to normal mode
                selected_command = commands_widget.get_selected_command()
                if selected_command:
                    text_area = self.query_one("#text-input")
                    text_area.text = selected_command
                    text_area.cursor_position = len(selected_command)
                self.action_escape()  # Return to normal mode
                event.prevent_default()
                event.stop()
            return

        # Handle normal mode
        if event.key == "enter" and self.mode == "normal":
            text_area = self.query_one("#text-input")
            text = text_area.text.strip()

            signals_widget.add_signal("DEBUG", f"ENTER_PRESSED: text='{text[:50]}...'")

            if text.startswith("/"):
                self.process_command(text)
            elif text:
                self.start_tts(text)

            # Clear input
            text_area.text = ""
            event.prevent_default()  # Prevent Enter from adding a new line
            event.stop()  # Stop event propagation

    def on_option_list_option_selected(self, event) -> None:
        """Handle selection from lists (Enter key or click selection)"""
        widget_id = event.option_list.id
        value = event.option.value

        # Add debug signal
        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal("DEBUG", f"OPTION_SELECTED: {widget_id} = {value}")

        if widget_id == "commands":
            self.process_command(value)
            self.action_escape()
        elif widget_id == "voices":
            self.select_voice(value)
            self.action_escape()
        elif widget_id == "languages":
            self.select_language(value)
            self.action_escape()
        else:
            self.action_escape()

    def on_option_list_option_highlighted(self, event) -> None:
        """Debug highlighting"""
        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal(
            "DEBUG", f"OPTION_HIGHLIGHTED: {event.option_list.id}"
        )

    def process_command(self, command: str):
        """Process a command"""
        signals_widget = self.query_one(TTSSignalsWidget)

        if command == "/clear":
            reading_widget = self.query_one(ReadingDisplayWidget)
            reading_widget.set_text("")
            signals_widget.add_signal("CONSOLE", "TEXT_CLEARED")

        elif command == "/stop":
            self.is_speaking = False
            status_widget = self.query_one(TTSStatusWidget)
            status_widget.update_status(is_speaking=False)
            signals_widget.add_signal("CONSOLE", "STOP_REQUESTED")

        elif command == "/voice":
            self.mode = "voice_select"
            self.query_one("#commands").add_class("hidden")
            self.query_one("#voices").remove_class("hidden")
            self.query_one("#voices").focus()
            signals_widget.add_signal("CONSOLE", "VOICE_SELECTION_STARTED")

        elif command == "/lang":
            self.mode = "lang_select"
            self.query_one("#commands").add_class("hidden")
            self.query_one("#languages").remove_class("hidden")
            self.query_one("#languages").focus()
            signals_widget.add_signal("CONSOLE", "LANGUAGE_SELECTION_STARTED")

        elif command == "/quit":
            self.exit()

        else:
            signals_widget.add_signal("CONSOLE", f"UNKNOWN_COMMAND: {command}")

    @work(exclusive=True)
    async def process_command_async(self, command: str):
        """Process a command asynchronously"""
        signals_widget = self.query_one(TTSSignalsWidget)
        if command == "/clear":
            reading_widget = self.query_one(ReadingDisplayWidget)
            reading_widget.set_text("")
            signals_widget.add_signal("CONSOLE", "TEXT_CLEARED")

        elif command == "/stop":
            self.is_speaking = False
            status_widget = self.query_one(TTSStatusWidget)
            status_widget.update_status(is_speaking=False)
            signals_widget.add_signal("CONSOLE", "STOP_REQUESTED")

        elif command == "/voice":
            self.mode = "voice_select"
            self.query_one("#commands").add_class("hidden")
            self.query_one("#voices").remove_class("hidden")
            self.query_one("#voices").focus()
            signals_widget.add_signal("CONSOLE", "VOICE_SELECTION_STARTED")

        elif command == "/lang":
            self.mode = "lang_select"
            self.query_one("#commands").add_class("hidden")
            self.query_one("#languages").remove_class("hidden")
            self.query_one("#languages").focus()
            signals_widget.add_signal("CONSOLE", "LANGUAGE_SELECTION_STARTED")

        elif command == "/quit":
            self.exit()

        else:
            signals_widget.add_signal("CONSOLE", f"UNKNOWN_COMMAND: {command}")

    def select_voice(self, voice: str):
        """Select a voice"""
        status_widget = self.query_one(TTSStatusWidget)
        status_widget.update_status(voice=voice)

        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal("CONSOLE", f"VOICE_CHANGED_TO_{voice.upper()}")

    def select_language(self, language: str):
        """Select a language"""
        status_widget = self.query_one(TTSStatusWidget)
        status_widget.update_status(language=language)

        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal("CONSOLE", f"LANGUAGE_CHANGED_TO_{language.upper()}")

    def start_tts(self, text: str):
        """Start TTS using signal system to avoid blocking"""
        reading_widget = self.query_one(ReadingDisplayWidget)
        status_widget = self.query_one(TTSStatusWidget)
        signals_widget = self.query_one(TTSSignalsWidget)

        # Set UI state immediately
        reading_widget.set_text(text)
        status_widget.update_status(is_speaking=True)
        signals_widget.add_signal("TTS", "SPEECH_REQUESTED")
        self.is_speaking = True

        # Call TTS handler directly (non-blocking)
        if self.provider:
            self.handle_tts_request(
                {"text": text, "voice": "am_adam", "lang_code": "a"}
            )
        else:
            signals_widget.add_signal("TTS", "PROVIDER_NOT_INITIALIZED")

    def action_escape(self) -> None:
        """Handle Escape key - return to normal mode"""
        self.mode = "normal"

        # Hide all selection widgets
        self.query_one("#commands").add_class("hidden")
        self.query_one("#voices").add_class("hidden")
        self.query_one("#languages").add_class("hidden")

    def action_toggle_commands(self) -> None:
        """Toggle commands display"""
        if self.mode == "commands":
            self.action_escape()
        else:
            self.mode = "commands"
            self.query_one("#commands").remove_class("hidden")
            self.query_one("#commands").focus()

    def action_select_voice(self) -> None:
        """Show voice selection"""
        self.mode = "voice_select"
        self.query_one("#voices").remove_class("hidden")
        self.query_one("#voices").focus()

    def action_select_language(self) -> None:
        """Show language selection"""
        self.mode = "lang_select"
        self.query_one("#languages").remove_class("hidden")
        self.query_one("#languages").focus()

    @work(exclusive=True)
    async def action_submit_text(self) -> None:
        """Handle Enter key - submit text or process command (async)"""
        if self.mode != "normal":
            return  # Don't handle Enter in command/selection modes

        signals_widget = self.query_one(TTSSignalsWidget)
        text_area = self.query_one("#text-input")
        text = text_area.text.strip()

        signals_widget.add_signal(
            "DEBUG", f"SUBMIT_ACTION: text='{text[:50]}...' mode={self.mode}"
        )

        if text.startswith("/"):
            self.process_command_async(text)
        elif text:
            self.start_tts(text)  # This is already a @work method, so no await needed

        # Clear input
        text_area.text = ""

    def action_quit(self) -> None:
        """Handle Ctrl+C - quit the application"""
        signals_widget = self.query_one(TTSSignalsWidget)
        signals_widget.add_signal("CONSOLE", "APPLICATION_QUIT_REQUESTED")
        self.exit()


def main():
    """Run the Kokoro TTS Console"""
    app = KokoroTTSApp()
    app.run()


if __name__ == "__main__":
    main()
