"""Encoder determinista para pruebas y desarrollo (no es un modelo real)."""

from __future__ import annotations

import asyncio

from chatbot.domain.models import Emotion, EmotionDistribution
from chatbot.domain.services.secondary_emotions import normalize_text

_KEYWORDS: dict[Emotion, tuple[str, ...]] = {
    Emotion.MIEDO: (
        "miedo",
        "nervios",
        "ansiedad",
        "ansios",
        "parcial",
        "examen",
        "panico",
        "preocup",
    ),
    Emotion.TRISTEZA: ("triste", "llorar", "sola", "solo", "deprim", "vacio", "desanim"),
    Emotion.IRA: ("rabia", "ira", "furios", "harto", "harta", "odio", "molest"),
    Emotion.ALEGRIA: ("feliz", "alegre", "contento", "contenta", "genial", "emocionado"),
    Emotion.ASCO: ("asco", "repugn"),
    Emotion.SORPRESA: ("sorpresa", "increible", "no lo puedo creer"),
}


class StubEncoder:
    """Puntúa por palabras clave; con `fixed` devuelve siempre esa distribución."""

    def __init__(
        self,
        fixed: EmotionDistribution | None = None,
        *,
        fail: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        self.fixed = fixed
        self.fail = fail
        self.delay_s = delay_s
        self.calls: list[str] = []

    async def classify(self, texto: str) -> EmotionDistribution:
        self.calls.append(texto)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("encoder caído (simulado)")
        if self.fixed is not None:
            return self.fixed
        norm = normalize_text(texto)
        scores = {e: 0.05 for e in Emotion}
        scores[Emotion.NEUTRAL] = 0.3
        for emotion, words in _KEYWORDS.items():
            scores[emotion] += sum(1.0 for w in words if w in norm)
        return EmotionDistribution.from_scores(scores)
