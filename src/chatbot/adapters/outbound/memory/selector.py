"""Selector por reglas en memoria (provisional). El bandido Thompson llega en la Fase 6."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from chatbot.application.dto import StrategySelection
from chatbot.domain.models import BanditContext, Strategy


class RuleBasedSelector:
    """Elige la primera candidata no usada antes en la conversación; registra recompensas."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.updates: list[tuple[str, str, float]] = []
        self.rewards: dict[tuple[str, str], list[float]] = defaultdict(list)

    async def select(self, context: BanditContext, candidates: Sequence[Strategy]) -> StrategySelection:
        if self.fail:
            raise RuntimeError("bandido caído (simulado)")
        unused = [s for s in candidates if s.id not in context.estrategias_previas]
        return StrategySelection((unused or list(candidates))[0], exploracion=False, metodo="reglas")

    async def update(self, context: BanditContext, strategy_id: str, reward: float) -> None:
        self.updates.append((context.cell_key, strategy_id, reward))
        self.rewards[(context.cell_key, strategy_id)].append(reward)
