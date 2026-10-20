"""The documentation is current: every command in it is accepted by the parser of the module it calls, and the links between the documents
resolve."""

import argparse
import importlib
import importlib.util
import inspect
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


class _Parsed(Exception):
    """Raised right after argparse accepted the arguments, so that nothing of the command is run."""


def _commands(path: Path):
    """(file, line, argv) of every `python -m …` / `python tools/….py` command in the ```bash blocks of a markdown file."""
    text = path.read_text(encoding="utf-8")
    for block in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", text, flags=re.S):
        variables: dict[str, str] = {}
        for line in re.sub(r"\\\n\s*", " ", block).splitlines():
            line = line.strip()
            m = re.fullmatch(r'(\w+)="(.*)"', line)
            if m:
                variables[m.group(1)] = m.group(2)
                continue
            for name, value in variables.items():
                line = line.replace(f"${name}", value)
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            while tokens and re.fullmatch(r"\w+=\S*", tokens[0]):  # PYTHONPATH=src python …
                tokens = tokens[1:]
            if len(tokens) >= 3 and tokens[0] == "python" and tokens[1] == "-m" and tokens[2].startswith("objeval3d."):
                yield path.name, tokens[2], tokens[3:]
            elif len(tokens) >= 2 and tokens[0] == "python" and tokens[1].startswith("tools/"):
                yield path.name, tokens[1], tokens[2:]


COMMANDS = [c for p in DOCS for c in _commands(p)]


def test_the_documentation_has_commands_to_check():
    assert len(COMMANDS) >= 30, f"only {len(COMMANDS)} commands found in the docs"


@pytest.mark.parametrize("doc, target, argv", COMMANDS, ids=lambda v: v if isinstance(v, str) else None)
def test_every_command_of_the_docs_is_accepted_by_its_parser(doc, target, argv, monkeypatch):
    if target.startswith("tools/"):
        spec = importlib.util.spec_from_file_location("tool", ROOT / target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as e:  # a GPU-only module in the CPU environment
            pytest.skip(f"{target} needs {e.name}")
    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        original(self, args, namespace)
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    monkeypatch.setattr(sys, "argv", [target, *argv])
    if "--help" in argv or "-h" in argv:  # a command that only asks for the help text exits with 0
        with pytest.raises(SystemExit) as e:
            module.main(argv) if _takes_argv(module.main) else module.main()
        assert e.value.code == 0
        return
    with pytest.raises(_Parsed):
        module.main(argv) if _takes_argv(module.main) else module.main()


def _takes_argv(fn) -> bool:
    return len(inspect.signature(fn).parameters) > 0


# ---------------------------------------------------------------------------------------------------------------------------------
# the links between the documents
# ---------------------------------------------------------------------------------------------------------------------------------
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _relative_links(path: Path):
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        yield target.split("#")[0]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_of_the_docs_resolve(doc):
    missing = [t for t in _relative_links(doc) if t and not (doc.parent / t).exists()]
    assert not missing, f"{doc.name} links to files that do not exist: {missing}"
