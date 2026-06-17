Champi TTS
==========

.. image:: https://badge.fury.io/py/champi-tts.svg
    :target: https://badge.fury.io/py/champi-tts

.. image:: https://github.com/divagnz/champi-tts/workflows/CI/badge.svg
    :target: https://github.com/divagnz/champi-tts/actions

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/psf/black/badge.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

Multi-Provider Text-to-Speech Library with Voice Reading Features
-----------------------------------------------------------------

A modular, extensible Python library for text-to-speech synthesis with support for multiple backends, full text reading capabilities with interruption support, and visual UI indicators.

.. contents:: Table of Contents
    :local:
    :backlinks: none

Features
--------

.. note::

    **Multi-Provider TTS Support**

    - **Kokoro** (Local, neural TTS) - Implemented
    - **OpenAI TTS API** - Coming Soon
    - **ElevenLabs** - Coming Soon
    - Unified interface across all providers

**Text Reading Service**

- Read text files with paragraph-by-paragraph processing
- Queue management for long documents
- Pause/resume/stop controls
- Interruption support for immediate stopping
- Event-driven architecture for state tracking

**Visual UI Indicator**

- Real-time visual status indicator using GLFW/ImGui
- Visual states:
    - **Idle** - Waiting for text
    - **Processing** - Processing text
    - **Speaking** - Currently speaking
    - **Paused** - Reading paused
    - **Error** - Error occurred
- Pulsing animations for active states
- Standalone testing mode

**Audio Features**

- High-quality audio synthesis
- Multiple voice options
- Adjustable speech speed
- Audio file export (WAV, MP3, etc.)
- Real-time audio playback with interruption

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install champi-tts

or using uv:

.. code-block:: bash

    uv pip install champi-tts

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/divagnz/champi-tts.git
    cd champi-tts
    uv sync --extra dev

Dependencies
~~~~~~~~~~~~

Core dependencies are:
- numpy
- sounddevice
- soundfile
- loguru
- blinker
- champi-signals

For full functionality including the Kokoro TTS provider, UI components, and CLI interface, install with extras:

.. code-block:: bash

    pip install champi-tts[kokoro,ui,cli]

or

.. code-block:: bash

    pip install "champi-tts[all]"

Usage
-----

Simple Synthesis
~~~~~~~~~~~~~~~~

.. code-block:: python

    from champi_tts import get_provider

    # Get default provider (Kokoro)
    provider = get_provider()
    await provider.initialize()

    # Synthesize audio
    audio = await provider.synthesize("Hello, world!")

    # Save to file
    from champi_tts.core.audio import save_audio
    await save_audio(audio, "hello.wav", sample_rate=provider.config.sample_rate)

    await provider.shutdown()

CLI Synthesis
~~~~~~~~~~~~

.. code-block:: bash

    # Synthesize and play
    champi-tts synthesize "Hello, world!" --voice af_bella

    # Synthesize and save
    champi-tts synthesize "Hello, world!" --output hello.wav --no-play

    # List available voices
    champi-tts list-voices

Text Reading
~~~~~~~~~~~~

.. code-block:: python

    from champi_tts import get_reader

    # Create reader with UI
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read text file
    await reader.read_file("document.txt", voice="af_bella")

    # Control playback
    await reader.pause()
    await reader.resume()
    await reader.stop()

    await reader.provider.shutdown()

CLI Text Reading
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Read text file with UI
    champi-tts read document.txt --voice af_bella --show-ui

    # Read text directly
    champi-tts read --text "This is a test" --voice am_adam

    # Interactive mode (with pause/resume controls)
    champi-tts read document.txt --interactive --show-ui

Test UI Indicator
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run standalone UI test
    champi-tts test-ui

Quick Start Examples
--------------------

See the `examples` directory for more complete examples.

API Documentation
-----------------

For detailed API documentation, see the inline docstrings in the source code or the project's documentation.

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (``git checkout -b feature/AmazingFeature``)
3. Commit your changes using Conventional Commits (https://www.conventionalcommits.org/)
4. Push to the branch (``git push origin feature/AmazingFeature``)
5. Open a Pull Request

Development Setup
-----------------

.. code-block:: bash

    # Clone repository
    git clone https://github.com/divagnz/champi-tts.git
    cd champi-tts

    # Install with development dependencies
    uv sync --extra dev

    # Install pre-commit hooks
    uv run pre-commit install

Running Tests
-------------

.. code-block:: bash

    # Run all tests
    uv run pytest

    # Run with coverage
    uv run pytest --cov=src/champi_tts --cov-report=term --cov-report=html

    # Run specific test file
    uv run pytest tests/test_provider.py

Code Quality
------------

.. code-block:: bash

    # Format code
    uv run ruff format src/

    # Lint code
    uv run ruff check src/

    # Type checking
    uv run mypy src/

    # Run all pre-commit hooks
    uv run pre-commit run --all-files

Project Structure
-----------------

.. code-block:: text

    champi-tts/
    ├── src/champi_tts/
    │   ├── core/                  # Generic abstractions
    │   │   ├── base_config.py
    │   │   ├── base_provider.py
    │   │   ├── base_synthesizer.py
    │   │   └── audio.py
    │   ├── providers/             # TTS implementations
    │   │   └── kokoro/
    │   ├── reader/                # Text reading service
    │   ├── ui/                    # Visual UI indicator
    │   ├── cli/                   # CLI interface
    │   └── factory.py             # Provider factory
    ├── tests/                     # Test suite
    ├── docs/                      # Documentation
    ├── examples/                  # Usage examples
    ├── .github/workflows/         # CI/CD workflows
    ├── pyproject.toml             # Project configuration
    └── README.rst                 # This file

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

Acknowledgments
--------------

Built with:
- `Kokoro TTS <https://github.com/thewh1teagle/kokoro-onnx>`_ - Neural text-to-speech engine
- `champi-signals <https://github.com/divagnz/champi-signals>`_ - Event-driven signal management
- `imgui-bundle <https://github.com/pthom/imgui_bundle>`_ - ImGui bindings for Python
- `Typer <https://typer.tiangolo.com/>`_ - CLI framework
- `Rich <https://rich.readthedocs.io/>`_ - Beautiful terminal output

Contact
-------

For questions, issues, or feature requests, please `open an issue <https://github.com/divagnz/champi-tts/issues>`_.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
