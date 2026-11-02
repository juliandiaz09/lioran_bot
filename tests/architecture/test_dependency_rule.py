"""Regla de dependencias hexagonal: el núcleo no importa frameworks ni adaptadores."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "chatbot"
FORBIDDEN = {
    "fastapi",
    "starlette",
    "uvicorn",
    "sqlalchemy",
    "asyncpg",
    "supabase",
    "transformers",
    "torch",
    "sentence_transformers",
    "openai",
    "httpx",
    "pydantic_settings",
    "structlog",
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _files(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", ["domain", "application"])
def test_el_nucleo_no_importa_frameworks(layer):
    for f in _files(layer):
        malos = {i for i in _imports(f) if i.split(".")[0] in FORBIDDEN}
        assert not malos, f"{f.relative_to(SRC)} importa {malos}"


@pytest.mark.parametrize("layer", ["domain", "application"])
def test_el_nucleo_no_importa_adaptadores_ni_config(layer):
    for f in _files(layer):
        malos = {i for i in _imports(f) if i.startswith(("chatbot.adapters", "chatbot.config"))}
        assert not malos, f"{f.relative_to(SRC)} importa {malos}"


def test_domain_no_importa_application():
    for f in _files("domain"):
        assert not {i for i in _imports(f) if i.startswith("chatbot.application")}, f
