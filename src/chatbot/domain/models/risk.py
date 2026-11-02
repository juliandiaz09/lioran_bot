"""Nivel de riesgo, ordenado: NONE < LOW < MEDIUM < HIGH."""

from __future__ import annotations

from enum import IntEnum


class RiskLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @classmethod
    def parse(cls, value: str | int | RiskLevel) -> RiskLevel:
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError as exc:
                raise ValueError(f"Nivel de riesgo desconocido: {value!r}") from exc
        return cls(value)

    @property
    def is_crisis(self) -> bool:
        return self is RiskLevel.HIGH
