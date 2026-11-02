"""Contexto discretizado del bandido contextual (una "celda" por combinación)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from chatbot.domain.models.emotion import Emotion, EmotionDistribution
from chatbot.domain.models.risk import RiskLevel


class TimeOfDay(StrEnum):
    MADRUGADA = "madrugada"  # 00-05
    MANANA = "manana"  # 06-11
    TARDE = "tarde"  # 12-17
    NOCHE = "noche"  # 18-23

    @classmethod
    def from_hour(cls, hour: int) -> TimeOfDay:
        if not 0 <= hour <= 23:
            raise ValueError("hora fuera de rango 0-23")
        if hour < 6:
            return cls.MADRUGADA
        if hour < 12:
            return cls.MANANA
        if hour < 18:
            return cls.TARDE
        return cls.NOCHE


class Certainty(StrEnum):
    """Qué tan definida está la emoción (inverso de la entropía normalizada)."""

    ALTA = "alta"
    MEDIA = "media"
    BAJA = "baja"

    @classmethod
    def from_entropy(cls, normalized_entropy: float) -> Certainty:
        if normalized_entropy < 0.35:
            return cls.ALTA
        if normalized_entropy < 0.65:
            return cls.MEDIA
        return cls.BAJA


@dataclass(frozen=True, slots=True)
class BanditContext:
    emocion: Emotion
    certeza: Certainty
    riesgo: RiskLevel
    franja: TimeOfDay
    #: Estrategias usadas antes en la conversación. No forma parte de la celda.
    estrategias_previas: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        distribution: EmotionDistribution,
        riesgo: RiskLevel,
        hour: int,
        estrategias_previas: Sequence[str] = (),
    ) -> BanditContext:
        return cls(
            emocion=distribution.top(),
            certeza=Certainty.from_entropy(distribution.normalized_entropy()),
            riesgo=riesgo,
            franja=TimeOfDay.from_hour(hour),
            estrategias_previas=tuple(estrategias_previas),
        )

    @property
    def cell_key(self) -> str:
        """Clave de la celda discretizada, p. ej. 'miedo|alta|LOW|noche'."""
        return f"{self.emocion.value}|{self.certeza.value}|{self.riesgo.name}|{self.franja.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "emocion": self.emocion.value,
            "certeza": self.certeza.value,
            "riesgo": self.riesgo.name,
            "franja": self.franja.value,
            "estrategias_previas": list(self.estrategias_previas),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BanditContext:
        return cls(
            emocion=Emotion(data["emocion"]),
            certeza=Certainty(data["certeza"]),
            riesgo=RiskLevel.parse(data["riesgo"]),
            franja=TimeOfDay(data["franja"]),
            estrategias_previas=tuple(data.get("estrategias_previas", ())),
        )
