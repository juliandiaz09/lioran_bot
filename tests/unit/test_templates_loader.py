"""Carga y fail-fast de las plantillas externalizadas."""

from pathlib import Path

import pytest

from chatbot.config.templates_loader import ARCHIVOS, InvalidTemplate, load_response_templates
from helpers import RAIZ

ORIGINALES = RAIZ / "config" / "templates"


def _copiar(tmp_path: Path) -> Path:
    for archivo in ARCHIVOS.values():
        (tmp_path / archivo).write_text((ORIGINALES / archivo).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_las_plantillas_oficiales_cargan_y_estan_pendientes_de_aprobacion():
    t = load_response_templates(ORIGINALES)
    assert "{contactos}" in t.crisis and t.crisis_sin_contactos and t.segura
    assert set(t.versiones) == set(ARCHIVOS)
    assert all("PENDIENTE_APROBACION" in v for v in t.versiones.values())


def test_render_con_contactos_y_sin_ellos():
    t = load_response_templates(ORIGINALES)
    con = t.render_crisis(["Bienestar: [[TELEFONO_BIENESTAR]]"])
    assert "Bienestar: [[TELEFONO_BIENESTAR]]" in con and "{contactos}" not in con
    sin = t.render_crisis([])
    assert t.crisis_sin_contactos in sin and "{contactos}" not in sin


def test_archivo_ausente_falla(tmp_path):
    base = _copiar(tmp_path)
    (base / "safe_response.md").unlink()
    with pytest.raises(InvalidTemplate, match="No se pudo leer"):
        load_response_templates(base)


def test_sin_cabecera_falla(tmp_path):
    base = _copiar(tmp_path)
    (base / "safe_response.md").write_text("solo texto", encoding="utf-8")
    with pytest.raises(InvalidTemplate, match="cabecera"):
        load_response_templates(base)


def test_cabecera_sin_version_o_estado_falla(tmp_path):
    base = _copiar(tmp_path)
    (base / "safe_response.md").write_text("---\nversion: '1'\n---\nhola", encoding="utf-8")
    with pytest.raises(InvalidTemplate, match="'version' y 'estado'"):
        load_response_templates(base)


def test_cabecera_yaml_invalida_falla(tmp_path):
    base = _copiar(tmp_path)
    (base / "safe_response.md").write_text("---\nversion: [\n---\nhola", encoding="utf-8")
    with pytest.raises(InvalidTemplate, match="YAML"):
        load_response_templates(base)


def test_cuerpo_vacio_falla(tmp_path):
    base = _copiar(tmp_path)
    (base / "safe_response.md").write_text("---\nversion: '1'\nestado: X\n---\n  \n", encoding="utf-8")
    with pytest.raises(InvalidTemplate, match="vacía"):
        load_response_templates(base)


def test_crisis_sin_marcador_de_contactos_falla(tmp_path):
    base = _copiar(tmp_path)
    (base / "crisis_response.md").write_text(
        "---\nversion: '1'\nestado: X\n---\nsin marcador", encoding="utf-8"
    )
    with pytest.raises(InvalidTemplate, match="marcador"):
        load_response_templates(base)
