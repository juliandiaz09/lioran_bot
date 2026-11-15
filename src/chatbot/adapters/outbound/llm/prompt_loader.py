"""Carga de prompts desde archivos `.md` con cabecera YAML (`version`) y variables `{{nombre}}`.

Falla de forma explícita (en vez de enviar un prompt a medias al LLM) si:
- el archivo no existe o no tiene cabecera con `version`;
- el prompt usa una variable que no se le entrega;
- se le entrega una variable que el prompt no usa.

La sustitución es de una sola pasada: si el valor de una variable contiene `{{algo}}` (p. ej. texto del
usuario), NO se vuelve a expandir.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_HEADER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.DOTALL)


class PromptError(Exception):
    """Prompt ausente, mal formado o usado con variables que no coinciden."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: str
    body: str
    variables: frozenset[str]

    def render(self, **values: str) -> str:
        provided = set(values)
        missing = self.variables - provided
        extra = provided - self.variables
        if missing:
            raise PromptError(f"Prompt '{self.name}': faltan variables {sorted(missing)}")
        if extra:
            raise PromptError(f"Prompt '{self.name}': variables que el prompt no usa {sorted(extra)}")
        return _VARIABLE.sub(lambda m: values[m.group(1)], self.body)


class PromptLoader:
    def __init__(self, prompts_dir: str | Path) -> None:
        self._dir = Path(prompts_dir)
        self._cache: dict[str, PromptTemplate] = {}

    def get(self, name: str) -> PromptTemplate:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def render(self, name: str, **values: str) -> str:
        return self.get(name).render(**values)

    def version(self, name: str) -> str:
        return self.get(name).version

    def _load(self, name: str) -> PromptTemplate:
        path = self._dir / f"{name}.md"
        if not path.is_file():
            raise PromptError(f"No se encontró el prompt '{name}' en {path}")
        raw = path.read_text(encoding="utf-8")
        match = _HEADER.match(raw)
        if match is None:
            raise PromptError(f"El prompt '{name}' no tiene cabecera YAML (--- ... ---) con 'version'")
        try:
            header = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise PromptError(f"Cabecera YAML inválida en el prompt '{name}'") from exc
        if not isinstance(header, Mapping) or not str(header.get("version", "")).strip():
            raise PromptError(f"El prompt '{name}' no declara 'version' en su cabecera")
        body = match.group(2)
        return PromptTemplate(
            name=name,
            version=str(header["version"]).strip(),
            body=body,
            variables=frozenset(_VARIABLE.findall(body)),
        )
