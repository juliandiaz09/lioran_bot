"""Catálogo de estrategias leído de un YAML versionado (`config/strategies.yaml`).

Falla al construirse (y por tanto al arrancar el servicio) si el archivo falta o es inválido.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from chatbot.domain.errors import DomainError
from chatbot.domain.models import Emotion, Strategy, StrategyType


class StrategyCatalogError(Exception):
    """El archivo del catálogo de estrategias falta o no es válido."""


class YamlStrategyCatalog:
    def __init__(self, strategies: Sequence[Strategy], version: str) -> None:
        if not strategies:
            raise StrategyCatalogError("El catálogo de estrategias no puede estar vacío")
        ids = [s.id for s in strategies]
        if len(set(ids)) != len(ids):
            raise StrategyCatalogError("El catálogo tiene ids de estrategia repetidos")
        self._strategies = list(strategies)
        self.version = version

    @classmethod
    def from_file(cls, path: str | Path) -> YamlStrategyCatalog:
        file = Path(path)
        if not file.is_file():
            raise StrategyCatalogError(f"No se encontró el catálogo de estrategias: {file}")
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise StrategyCatalogError(f"No se pudo leer el catálogo {file}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise StrategyCatalogError(f"{file}: se esperaba un mapa con 'version' y 'estrategias'")
        version = str(data.get("version", "")).strip()
        if not version:
            raise StrategyCatalogError(f"{file}: falta la cabecera 'version'")
        items = data.get("estrategias")
        if not isinstance(items, list) or not items:
            raise StrategyCatalogError(f"{file}: 'estrategias' debe ser una lista no vacía")
        return cls([cls._parse(i, item, file) for i, item in enumerate(items)], version)

    @staticmethod
    def _parse(index: int, item: Any, file: Path) -> Strategy:
        where = f"{file}: estrategia #{index + 1}"
        if not isinstance(item, Mapping):
            raise StrategyCatalogError(f"{where}: se esperaba un mapa")
        missing = [k for k in ("id", "nombre", "tipo", "emociones_objetivo") if k not in item]
        if missing:
            raise StrategyCatalogError(f"{where}: faltan campos {missing}")
        raw_emotions = item["emociones_objetivo"]
        if not isinstance(raw_emotions, list):
            raise StrategyCatalogError(f"{where}: 'emociones_objetivo' debe ser una lista")
        try:
            return Strategy(
                id=str(item["id"]),
                nombre=str(item["nombre"]),
                tipo=StrategyType(item["tipo"]),
                emociones_objetivo=frozenset(Emotion(e) for e in raw_emotions),
                descripcion=str(item.get("descripcion", "")),
            )
        except (ValueError, DomainError) as exc:  # tipo/emoción desconocidos o estrategia inválida
            raise StrategyCatalogError(f"{where}: {exc}") from exc

    # ------------------------------------------------------------------------- StrategyCatalogPort
    async def list_all(self) -> Sequence[Strategy]:
        return list(self._strategies)

    async def for_emotion(self, emocion: Emotion) -> Sequence[Strategy]:
        return [s for s in self._strategies if s.applies_to(emocion)]
