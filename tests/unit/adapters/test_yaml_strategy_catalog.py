from pathlib import Path

import pytest

from chatbot.adapters.outbound.memory.seed import seed_strategies
from chatbot.adapters.outbound.strategies.yaml_strategy_catalog import (
    StrategyCatalogError,
    YamlStrategyCatalog,
)
from chatbot.domain.models import Emotion

REAL = Path(__file__).resolve().parents[3] / "config" / "strategies.yaml"

VALIDO = """
version: "1.0"
estrategias:
  - {id: a, nombre: A, tipo: respiracion, emociones_objetivo: [miedo], descripcion: "d"}
  - {id: b, nombre: B, tipo: grounding, emociones_objetivo: [miedo, ira]}
"""


def write(tmp_path, content):
    f = tmp_path / "s.yaml"
    f.write_text(content, encoding="utf-8")
    return f


async def test_carga_valida(tmp_path):
    cat = YamlStrategyCatalog.from_file(write(tmp_path, VALIDO))
    assert cat.version == "1.0"
    assert [s.id for s in await cat.list_all()] == ["a", "b"]
    assert [s.id for s in await cat.for_emotion(Emotion.IRA)] == ["b"]
    assert [s.id for s in await cat.for_emotion(Emotion.MIEDO)] == ["a", "b"]
    assert await cat.for_emotion(Emotion.ALEGRIA) == []


def test_archivo_ausente(tmp_path):
    with pytest.raises(StrategyCatalogError, match="No se encontró"):
        YamlStrategyCatalog.from_file(tmp_path / "no-existe.yaml")


@pytest.mark.parametrize(
    "content, fragmento",
    [
        ("version: [sin cerrar", "No se pudo leer"),
        ("- solo una lista", "se esperaba un mapa"),
        ("estrategias: []", "version"),
        ('version: "1"\nestrategias: []', "lista no vacía"),
        ('version: "1"\nestrategias: texto', "lista no vacía"),
        ('version: "1"\nestrategias: [1]', "se esperaba un mapa"),
        ('version: "1"\nestrategias:\n  - {id: a, nombre: A}', "faltan campos"),
        (
            'version: "1"\nestrategias:\n  - {id: a, nombre: A, tipo: nuevo, emociones_objetivo: [miedo]}',
            "estrategia #1",
        ),
        (
            'version: "1"\nestrategias:\n  - {id: a, nombre: A, tipo: grounding,emociones_objetivo: [x1]}',
            "estrategia #1",
        ),
        (
            'version: "1"\nestrategias:\n  - {id: a, nombre: A, tipo: grounding, emociones_objetivo: x}',
            "debe ser una lista",
        ),
        (
            'version: "1"\nestrategias:\n  - {id: a, nombre: A, tipo: grounding, emociones_objetivo: []}',
            "estrategia #1",
        ),
        (
            'version: "1"\nestrategias:\n'
            "  - {id: a, nombre: A, tipo: grounding, emociones_objetivo: [miedo]}\n"
            "  - {id: a, nombre: B, tipo: grounding, emociones_objetivo: [ira]}",
            "repetidos",
        ),
    ],
)
def test_archivo_invalido(tmp_path, content, fragmento):
    with pytest.raises(StrategyCatalogError, match=fragmento):
        YamlStrategyCatalog.from_file(write(tmp_path, content))


async def test_el_yaml_real_coincide_con_seed_strategies():
    """Mismas estrategias que `seed_strategies()` (que queda solo para pruebas)."""
    real = {s.id: s for s in await YamlStrategyCatalog.from_file(REAL).list_all()}
    seed = {s.id: s for s in seed_strategies()}
    assert real == seed
