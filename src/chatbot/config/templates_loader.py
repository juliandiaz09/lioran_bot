"""Carga de las plantillas de respuesta desde archivos versionados (`config/templates/*.md`).

Cada archivo lleva una cabecera YAML entre `---` con `version` y `estado`. El arranque falla
(`InvalidTemplate`) si falta un archivo, la cabecera es inválida o la plantilla de crisis no trae
el marcador `{contactos}`: una respuesta de crisis sin lugar para los contactos sería un hueco
silencioso en la ruta más sensible.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from chatbot.application.dto import ResponseTemplates
from chatbot.domain.errors import DomainError

ARCHIVOS = {
    "crisis": "crisis_response.md",
    "crisis_sin_contactos": "crisis_no_contacts.md",
    "segura": "safe_response.md",
}


class InvalidTemplate(DomainError):
    codigo = "plantilla_invalida"


def _leer(ruta: Path) -> tuple[dict[str, str], str]:
    try:
        crudo = ruta.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidTemplate(f"No se pudo leer la plantilla {ruta}: {exc.strerror or exc}") from exc
    partes = crudo.split("---", 2)
    if len(partes) != 3 or partes[0].strip():
        raise InvalidTemplate(f"La plantilla {ruta.name} debe empezar con una cabecera '---' ... '---'")
    try:
        cabecera = yaml.safe_load(partes[1])
    except yaml.YAMLError as exc:
        raise InvalidTemplate(f"Cabecera YAML inválida en {ruta.name}") from exc
    if not isinstance(cabecera, dict) or not cabecera.get("version") or not cabecera.get("estado"):
        raise InvalidTemplate(f"La cabecera de {ruta.name} debe tener 'version' y 'estado'")
    cuerpo = partes[2].strip()
    if not cuerpo:
        raise InvalidTemplate(f"La plantilla {ruta.name} está vacía")
    return {k: str(v) for k, v in cabecera.items()}, cuerpo


def load_response_templates(directorio: str | Path) -> ResponseTemplates:
    base = Path(directorio)
    textos: dict[str, str] = {}
    versiones: dict[str, str] = {}
    for clave, archivo in ARCHIVOS.items():
        cabecera, cuerpo = _leer(base / archivo)
        textos[clave] = cuerpo
        versiones[clave] = f"{cabecera['version']} ({cabecera['estado']})"
    if ResponseTemplates.MARCADOR_CONTACTOS not in textos["crisis"]:
        raise InvalidTemplate(
            f"crisis_response.md debe contener el marcador {ResponseTemplates.MARCADOR_CONTACTOS}"
        )
    return ResponseTemplates(
        crisis=textos["crisis"],
        crisis_sin_contactos=textos["crisis_sin_contactos"],
        segura=textos["segura"],
        versiones=versiones,
    )
