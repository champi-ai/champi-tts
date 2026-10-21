"""
Packaging guards: every third-party import is declared, and optional
extras are not needed just to import the package.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "champi_tts"

# Import names whose distribution name differs from the module name.
IMPORT_TO_DIST = {
    "bs4": "beautifulsoup4",
    "champi_signals": "champi-signals",
    "dotenv": "python-dotenv",
    "glm": "pyglm",
    "imgui_bundle": "imgui-bundle",
    "phonemizer": "phonemizer-fork",
}

# Modules that only ship with the kokoro extra.
KOKORO_ONLY_MODULES = [
    "torch",
    "torchaudio",
    "kokoro",
    "misaki",
    "phonemizer",
    "inflect",
    "scipy",
    "awaitlet",
]


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    requirements = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        requirements.extend(extra)
    names = set()
    for requirement in requirements:
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        assert match, f"Unparseable requirement: {requirement}"
        names.add(_normalize(match.group(0)))
    return names


def _third_party_imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top == "champi_tts" or top in sys.stdlib_module_names:
                    continue
                found.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def test_every_third_party_import_is_declared():
    declared = _declared_distributions()
    undeclared = {
        module: sorted(files)
        for module, files in _third_party_imports().items()
        if _normalize(IMPORT_TO_DIST.get(module, module)) not in declared
    }
    assert not undeclared, f"Imports not covered by any dependency/extra: {undeclared}"


def _run_without_kokoro_extra(code: str) -> subprocess.CompletedProcess[str]:
    blocker = "".join(f"sys.modules[{name!r}] = None\n" for name in KOKORO_ONLY_MODULES)
    return subprocess.run(
        [sys.executable, "-c", f"import sys\n{blocker}{code}"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_kokoro_star_import_without_kokoro_extra():
    result = _run_without_kokoro_extra(
        "from champi_tts.providers.kokoro import *\n"
        "import champi_tts.providers.kokoro as k\n"
        "missing = [n for n in k.__all__ if not hasattr(k, n)]\n"
        "assert not missing, missing\n"
    )
    assert result.returncode == 0, result.stderr
    assert "Missing dependencies" not in result.stderr


def test_kokoro_all_names_resolve():
    import champi_tts.providers.kokoro as kokoro

    assert "TextProcessingEvents" in kokoro.__all__
    assert [name for name in kokoro.__all__ if not hasattr(kokoro, name)] == []
