"""Fusión tardía de distribuciones emocionales: P_final = Σ w_m · P_m."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from chatbot.domain.errors import InvalidFusionConfig
from chatbot.domain.models.emotion import EMOTION_ORDER, EmotionDistribution


@dataclass(frozen=True, slots=True)
class FusionResult:
    distribution: EmotionDistribution
    #: Desacuerdo entre modelos en [0, 1] (divergencia de Jensen-Shannon generalizada,
    #: normalizada). 0 = coinciden; 1 = máximo desacuerdo posible para esos pesos.
    disagreement: float
    #: Modelos que efectivamente participaron (útil cuando uno falla: degradación elegante).
    models_used: tuple[str, ...]


class FusionService:
    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise InvalidFusionConfig("Se requiere al menos un modelo con peso")
        if any(w < 0 or math.isnan(w) for w in weights.values()):
            raise InvalidFusionConfig("Los pesos no pueden ser negativos")
        if sum(weights.values()) <= 0:
            raise InvalidFusionConfig("La suma de pesos debe ser mayor que 0")
        self._weights = dict(weights)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def fuse(self, distributions: Mapping[str, EmotionDistribution]) -> FusionResult:
        """Fusiona las distribuciones disponibles.

        Si falta algún modelo (p. ej. el LLM cayó) los pesos de los restantes se
        renormalizan, de modo que el sistema degrada sin fallar.
        """
        if not distributions:
            raise InvalidFusionConfig("No hay distribuciones para fusionar")
        unknown = set(distributions) - set(self._weights)
        if unknown:
            raise InvalidFusionConfig(f"Modelos sin peso configurado: {sorted(unknown)}")

        active = {m: self._weights[m] for m in distributions}
        total = sum(active.values())
        if total <= 0:
            raise InvalidFusionConfig("Los modelos disponibles tienen peso total 0")
        norm = {m: w / total for m, w in active.items()}

        mixed = {e: sum(norm[m] * distributions[m][e] for m in norm) for e in EMOTION_ORDER}
        final = EmotionDistribution.from_scores(mixed)
        return FusionResult(
            distribution=final,
            disagreement=self._disagreement(norm, distributions, final),
            models_used=tuple(sorted(norm)),
        )

    @staticmethod
    def _disagreement(
        norm: Mapping[str, float],
        distributions: Mapping[str, EmotionDistribution],
        mixed: EmotionDistribution,
    ) -> float:
        weights = [w for w in norm.values() if w > 0]
        if len(weights) < 2:
            return 0.0
        weight_entropy = -sum(w * math.log2(w) for w in weights)
        jsd = mixed.entropy() - sum(w * distributions[m].entropy() for m, w in norm.items())
        return min(1.0, max(0.0, jsd / weight_entropy))
