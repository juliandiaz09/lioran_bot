"""Fase 5: las librerías de los extras `ml` y `db` se importan SOLO dentro de funciones/métodos.

Motivo: `pytest` (y el servicio con adaptadores falsos) deben funcionar sin instalar esos extras.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "chatbot"
HEAVY = {"torch", "transformers", "sentence_transformers", "sqlalchemy", "asyncpg", "pgvector"}


def _module_level_imports(path: Path) -> set[str]:
    """Imports ejecutados al importar el módulo (fuera de funciones y de `if TYPE_CHECKING`)."""
    found: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.If):
                test = ast.unparse(node.test)
                if "TYPE_CHECKING" not in test:
                    visit(node.body)
                    visit(node.orelse)
            elif isinstance(node, ast.Try):
                for block in (node.body, node.orelse, node.finalbody, *(h.body for h in node.handlers)):
                    visit(block)
            elif isinstance(node, ast.ClassDef):
                visit(node.body)  # el cuerpo de la clase sí se ejecuta al importar

    visit(ast.parse(path.read_text(encoding="utf-8")).body)
    return found


@pytest.mark.parametrize("path", sorted((SRC).rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_ningun_modulo_importa_librerias_pesadas_a_nivel_de_modulo(path):
    assert not (_module_level_imports(path) & HEAVY), (
        f"{path.relative_to(SRC)} importa {HEAVY} a nivel de módulo"
    )


def test_el_detector_funciona(tmp_path):
    malo = tmp_path / "m.py"
    malo.write_text("import torch\n\ndef f():\n    import sqlalchemy\n")
    assert _module_level_imports(malo) == {"torch"}
