"""Emociones primarias (Ekman + neutral) y su distribución de probabilidad."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from chatbot.domain.errors import InvalidEmotionDistribution


class Emotion(StrEnum):
    ALEGRIA = "alegria"
    TRISTEZA = "tristeza"
    MIEDO = "miedo"
    IRA = "ira"
    ASCO = "asco"
    SORPRESA = "sorpresa"
    NEUTRAL = "neutral"


#: Orden fijo de las emociones; sirve también para desempatar de forma determinista.
EMOTION_ORDER: tuple[Emotion, ...] = tuple(Emotion)

SUM_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class EmotionDistribution:
    """Distribución emoción -> probabilidad. Inmutable; valida que sume 1 (con tolerancia).

    Las emociones ausentes en `probabilities` se completan con 0.
    """

    probabilities: Mapping[Emotion, float]

    def __post_init__(self) -> None:
        full: dict[Emotion, float] = {}
        for emotion in EMOTION_ORDER:
            value = float(self.probabilities.get(emotion, 0.0))
            if math.isnan(value) or value < 0.0 or value > 1.0 + SUM_TOLERANCE:
                raise InvalidEmotionDistribution(f"Probabilidad inválida para {emotion.value}: {value}")
            full[emotion] = value
        unknown = set(self.probabilities) - set(EMOTION_ORDER)
        if unknown:
            raise InvalidEmotionDistribution(f"Emociones desconocidas: {sorted(map(str, unknown))}")
        total = sum(full.values())
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise InvalidEmotionDistribution(f"Las probabilidades deben sumar 1 (suman {total:.4f})")
        object.__setattr__(self, "probabilities", MappingProxyType(full))

    # --- construcción -------------------------------------------------------------------
    @classmethod
    def from_scores(cls, scores: Mapping[Emotion, float]) -> EmotionDistribution:
        """Normaliza puntajes no negativos para que sumen 1."""
        if any(v < 0 or math.isnan(v) for v in scores.values()):
            raise InvalidEmotionDistribution("Los puntajes no pueden ser negativos ni NaN")
        total = sum(scores.values())
        if total <= 0:
            raise InvalidEmotionDistribution("La suma de puntajes debe ser mayor que 0")
        return cls({e: v / total for e, v in scores.items()})

    @classmethod
    def uniform(cls) -> EmotionDistribution:
        return cls({e: 1.0 / len(EMOTION_ORDER) for e in EMOTION_ORDER})

    @classmethod
    def certain(cls, emotion: Emotion) -> EmotionDistribution:
        return cls({emotion: 1.0})

    # --- consulta -----------------------------------------------------------------------
    def __getitem__(self, emotion: Emotion) -> float:
        return self.probabilities[emotion]

    def top(self) -> Emotion:
        """Emoción más probable; en empate gana la primera según EMOTION_ORDER."""
        return self.top_k(1)[0]

    def top_k(self, k: int) -> list[Emotion]:
        if k < 1:
            raise ValueError("k debe ser >= 1")
        ordered = sorted(EMOTION_ORDER, key=lambda e: (-self.probabilities[e], EMOTION_ORDER.index(e)))
        return ordered[:k]

    def entropy(self) -> float:
        """Entropía de Shannon en bits (0 = certeza total; log2(7) ≈ 2.807 = uniforme)."""
        return -sum(p * math.log2(p) for p in self.probabilities.values() if p > 0)

    def normalized_entropy(self) -> float:
        """Entropía escalada a [0, 1]; útil como medida de intensidad/incertidumbre."""
        return self.entropy() / math.log2(len(EMOTION_ORDER))

    def to_dict(self) -> dict[str, float]:
        return {e.value: self.probabilities[e] for e in EMOTION_ORDER}
