from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chatbot.application.dto import StrategySelection
from chatbot.domain.models import BanditContext, Emotion, Strategy


class StrategyCatalogPort(Protocol):
    async def list_all(self) -> Sequence[Strategy]: ...

    async def for_emotion(self, emocion: Emotion) -> Sequence[Strategy]: ...


class StrategySelectorPort(Protocol):
    async def select(self, context: BanditContext, candidates: Sequence[Strategy]) -> StrategySelection: ...

    async def update(self, context: BanditContext, strategy_id: str, reward: float) -> None:
        """Registra la recompensa (retroalimentación) de la estrategia en esa celda."""
        ...
