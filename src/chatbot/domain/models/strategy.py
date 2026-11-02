"""Estrategia de regulación emocional (elemento del catálogo)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from chatbot.domain.errors import InvalidStrategy
from chatbot.domain.models.emotion import Emotion


class StrategyType(StrEnum):
    RESPIRACION = "respiracion"
    REENCUADRE_COGNITIVO = "reencuadre_cognitivo"
    VALIDACION_EMOCIONAL = "validacion_emocional"
    GROUNDING = "grounding"
    PLANIFICACION = "planificacion"
    PSICOEDUCACION = "psicoeducacion"
    DERIVACION = "derivacion"


@dataclass(frozen=True, slots=True)
class Strategy:
    id: str
    nombre: str
    tipo: StrategyType
    emociones_objetivo: frozenset[Emotion]
    descripcion: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.nombre.strip():
            raise InvalidStrategy("La estrategia requiere id y nombre no vacíos")
        if not self.emociones_objetivo:
            raise InvalidStrategy(f"La estrategia {self.id!r} requiere al menos una emoción objetivo")

    def applies_to(self, emotion: Emotion) -> bool:
        return emotion in self.emociones_objetivo
