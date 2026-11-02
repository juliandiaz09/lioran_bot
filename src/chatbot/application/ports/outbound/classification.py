from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chatbot.application.dto import (
    GenerationContext,
    LLMClassification,
    RiskDetection,
)
from chatbot.domain.models import EmotionDistribution, Message


class EmotionClassifierPort(Protocol):
    """Encoder de texto en español (calibrado)."""

    async def classify(self, texto: str) -> EmotionDistribution: ...


class LLMPort(Protocol):
    async def classify_emotion(self, texto: str, contexto: Sequence[Message]) -> LLMClassification:
        """Salida JSON estricta validada por el adaptador (con reintento ante JSON inválido)."""
        ...

    async def generate(self, contexto: GenerationContext) -> str: ...


class RiskDetectorPort(Protocol):
    """Detección de riesgo por reglas. Síncrona y sin E/S: debe ser rápida y no fallar."""

    def detect(self, texto: str) -> RiskDetection: ...
