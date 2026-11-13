"""Esquemas de `GET /v1/strategies`: catálogo de estrategias de regulación."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from chatbot.domain.models import EMOTION_ORDER, Strategy


class StrategyOut(BaseModel):
    """Una estrategia del catálogo."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    nombre: str
    tipo: str
    #: Emociones a las que aplica, en el orden fijo de `EMOTION_ORDER` (salida determinista).
    emociones_objetivo: list[str]
    descripcion: str

    @classmethod
    def from_strategy(cls, strategy: Strategy) -> StrategyOut:
        return cls(
            id=strategy.id,
            nombre=strategy.nombre,
            tipo=strategy.tipo.value,
            emociones_objetivo=[e.value for e in EMOTION_ORDER if e in strategy.emociones_objetivo],
            descripcion=strategy.descripcion,
        )


class StrategyListOut(BaseModel):
    """Catálogo completo, en el orden en que lo entrega el caso de uso."""

    model_config = ConfigDict(extra="forbid", strict=True)

    estrategias: list[StrategyOut]

    @classmethod
    def from_strategies(cls, strategies: Iterable[Strategy]) -> StrategyListOut:
        return cls(estrategias=[StrategyOut.from_strategy(s) for s in strategies])
