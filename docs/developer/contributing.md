# Contributing Guide

Thank you for your interest in contributing to Champi TTS! This guide will help you get started with making contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style Guidelines](#code-style-guidelines)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Committing Changes](#committing-changes)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)
- [Getting Help](#getting-help)

---

## Getting Started

### Fork the Repository

1. Go to [Champi TTS GitHub repository](https://github.com/divagnz/champi-tts)
2. Click "Fork" to create your own fork
3. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/champi-tts.git
cd champi-tts
```


### Set Up Branch

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Or use a different branch naming convention
# feature/description
# fix/description
# docs/description
# test/description
# refactor/description
```

---

## Development Environment

### Prerequisites

- Python 3.12 or higher
- Git
- pip or uv

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/champi-tts.git
cd champi-tts

# Install development dependencies
pip install -e ".[dev]"
```


### Optional Tools

- **Black**: Code formatter
- **Ruff**: Fast Python linter
- **Mypy**: Type checker
- **Pytest**: Testing framework
- **Pydub**: Audio processing

Install with extras:

```bash
pip install -e ".[dev,ui]"
```


### IDE Setup

We recommend using **VS Code** with these extensions:

- Python
- Pylance
- Black Formatter
- Ruff
- pytest

---

## Code Style Guidelines

### Python Style

We follow **PEP 8** guidelines with some specific preferences:

```python
# Use 4 spaces for indentation
def example_function(arg1, arg2):
    """Docstring here"""
    pass

# Type hints for all functions
def example_function(arg1: str, arg2: int) -> bool:
    """Docstring here"""
    return True

# Async functions use 'async'
async def example_async_function():
    """Async docstring"""
    pass

# Imports sorted and grouped
import asyncio
import logging

from champi_tts import get_provider
from champi_tts.core.audio import save_audio
```


### Docstrings

Use **Google-style** docstrings:

```python
async def synthesize(text: str, voice: str = None) -> np.ndarray:
    """Synthesize text to speech.

    Args:
        text: The text to synthesize
        voice: Voice name to use (optional)

    Returns:
        Audio array as numpy array

    Raises:
        InitializationError: If provider fails to initialize
        SynthesisError: If synthesis fails

    Example:
        >>> audio = await provider.synthesize("Hello, world!")
    """
    pass
```


### Naming Conventions

- **Functions**: `snake_case()`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private variables**: `_leading_underscore`
- **Protected variables**: `__leading_underscore` (optional)

---

### File Structure

```python
# champi_tts/
# ├── __init__.py
# ├── providers/
# │   ├── __init__.py
# │   ├── base.py
# │   └── kokoro.py
# ├── core/
# │   ├── __init__.py
# │   ├── audio.py
# │   ├── reader.py
# │   └── types.py
# ├── ui/
# │   ├── __init__.py
# │   └── indicator.py
# └── cli/
#     ├── __init__.py
#     └── main.py
```


---

## Development Workflow

### Making Changes

1. **Create a branch** with descriptive name
2. **Make your changes**
3. **Run tests** to ensure nothing breaks
4. **Format your code** with Black
5. **Run linter** with Ruff
6. **Commit** with descriptive message
7. **Push** to your fork
8. **Open PR** for review

### Example Workflow

```bash
# 1. Create branch
git checkout -b feature/new-provider

# 2. Make changes

# 3. Run tests
pytest tests/

# 4. Format code
black .

# 5. Run linter
ruff check .

# 6. Commit
git add .
git commit -m "feat: Add new provider support"

# 7. Push
git push origin feature/new-provider
```


---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_provider.py -v

# Run with coverage
pytest tests/ --cov=champi_tts --cov-report=html

# Run with verbose output
pytest tests/ -v -s
```


### Test Structure

```
tests/
├── __init__.py
├── test_provider.py
├── test_reader.py
├── test_audio.py
└── fixtures/
    └── sample_audio.wav
```


### Test Example

```python
import pytest
import asyncio
from champi_tts import get_provider

@pytest.mark.asyncio
async def test_basic_synthesis():
    """Test basic text synthesis"""
    async with get_provider() as provider:
        await provider.initialize()

        audio = await provider.synthesize("Hello, world!")
        assert audio is not None
        assert len(audio) > 0

@pytest.mark.asyncio
async def test_synthesize_with_voice():
    """Test synthesis with specific voice"""
    async with get_provider() as provider:
        await provider.initialize()

        audio = await provider.synthesize("Test", voice="af_bella")
        assert audio is not None
```


---

## Committing Changes

### Commit Message Format

Use **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:

```bash
# Feature
git commit -m "feat(kokoro): Add streaming support for large texts"

# Bug fix
git commit -m "fix(audio): Fix audio normalization bug"

# Documentation
git commit -m "docs(api): Add comprehensive API documentation"

# Test
git commit -m "test(reader): Add unit tests for reader service"

# Refactor
git commit -m "refactor(core): Improve audio player memory management"
```


---

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
2. **Run linter and formatter**
3. **Add/update tests** for new features
4. **Update documentation** if needed
5. **Review existing PRs** if applicable

### PR Title Format

```
<type>: <description>
```

Example: `feat: Add OpenAI provider support`

### PR Description

Include:
- Description of changes
- Type of change
- Related issues (if any)
- Breaking changes (if any)
- Screenshots (if UI changes)

Example:

```markdown
## Summary
- Add OpenAI provider support
- Implement streaming synthesis
- Add comprehensive error handling

## Type of Change
- New feature

## Breaking Changes
None

## Test Plan
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing completed
```


### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] PR has descriptive title and description
- [ ] Issue linked (if applicable)

---

## Code of Conduct

### Our Values

- Respectful communication
- Inclusive environment
- Constructive feedback
- Shared responsibility

### Behavior Expectations

- Be professional
- Focus on what's best for the community
- Be direct and honest
- Assume good intentions
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment
- Offensive language
- Personal attacks
- Discriminatory comments
- Inappropriate behavior

### Reporting Violations

If you experience or witness unacceptable behavior, please:

1. Contact project maintainers
2. Report via GitHub issue
3. Provide details: date, location, description

---

## Getting Help

### Documentation

- [API Reference](../api.md)
- [Developer Guide](developer.md)
- [Examples](../examples/)
- [FAQ](../user/faq.md)

### Communication

- GitHub Discussions: [Discussions](https://github.com/divagnz/champi-tts/discussions)
- GitHub Issues: [Issues](https://github.com/divagnz/champi-tts/issues)
- Email: (if provided in README)

### Asking Questions

When asking questions, include:
- Clear description of the problem
- Code examples
- Expected vs actual behavior
- Your environment (Python version, OS)

---

## Project Guidelines

### Feature Requests

We welcome feature requests! Submit them via:
- GitHub Issue with "enhancement" label
- GitHub Discussions

When requesting features:
- Explain the use case
- Provide examples
- Consider alternatives

### Issue Reporting

Report bugs via GitHub Issues with:
- Clear title and description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
- Error messages
- Screenshots (if applicable)

### Code Review

PRs are reviewed by maintainers. Be prepared to:
- Address feedback
- Make changes
- Explain your decisions

---

## Additional Resources

- [Python Documentation](https://docs.python.org/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)

---

## Thank You!

Contributing to Champi TTS is a great way to help the community and improve the project. Your contributions make a difference!

Thank you for your time and effort! 🎉
