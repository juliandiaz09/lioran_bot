"""Política de riesgo: el nivel más alto entre todas las fuentes siempre prevalece."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chatbot.domain.models.risk import RiskLevel


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    level: RiskLevel
    rules: RiskLevel
    llm: RiskLevel
    history: RiskLevel

    @property
    def is_crisis(self) -> bool:
        return self.level.is_crisis

    @property
    def source(self) -> str:
        """Fuente(s) que determinan el nivel final (para auditoría/registro)."""
        named = (("reglas", self.rules), ("llm", self.llm), ("historial", self.history))
        return "+".join(n for n, v in named if v == self.level) or "ninguna"


class RiskPolicy:
    """`riesgo_final = max(riesgo_reglas, riesgo_llm, riesgo_historial)`.

    Prefiere un falso positivo a un falso negativo: nunca se reduce un nivel por consenso.
    """

    def __init__(self, history_window: int = 5) -> None:
        if history_window < 1:
            raise ValueError("history_window debe ser >= 1")
        self._window = history_window

    def history_level(self, recent_levels: Sequence[RiskLevel]) -> RiskLevel:
        """Riesgo por historial: el máximo de la ventana reciente, rebajado un escalón.

        Una conversación con un episodio HIGH reciente se trata al menos como MEDIUM;
        con MEDIUM, como LOW. El historial solo eleva el riesgo del mensaje actual.
        """
        window = list(recent_levels)[-self._window :]
        if not window:
            return RiskLevel.NONE
        return RiskLevel(max(0, max(window) - 1))

    def assess(
        self,
        rules: RiskLevel,
        llm: RiskLevel | None = None,
        recent_levels: Sequence[RiskLevel] = (),
    ) -> RiskAssessment:
        llm_level = llm if llm is not None else RiskLevel.NONE  # LLM caído => no aporta
        history = self.history_level(recent_levels)
        return RiskAssessment(
            level=max(rules, llm_level, history),
            rules=rules,
            llm=llm_level,
            history=history,
        )
