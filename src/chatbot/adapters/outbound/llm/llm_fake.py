"""LLM falso y configurable para pruebas: registra llamadas y permite simular fallos."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.application.dto import GenerationContext, LLMClassification
from chatbot.domain.models import EmotionDistribution, Message, RiskLevel


class FakeLLM:
    def __init__(
        self,
        classification: LLMClassification | None = None,
        *,
        generate_outputs: Sequence[str] | None = None,
        fail_classify: bool = False,
        fail_generate: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        self.classification = classification
        self.generate_outputs = list(generate_outputs) if generate_outputs else []
        self.fail_classify = fail_classify
        self.fail_generate = fail_generate
        self.delay_s = delay_s
        self.classify_calls: list[str] = []
        self.generate_calls: list[GenerationContext] = []
        self._heuristic = StubEncoder()

    async def classify_emotion(self, texto: str, contexto: Sequence[Message]) -> LLMClassification:
        self.classify_calls.append(texto)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_classify:
            raise RuntimeError("LLM caído (simulado)")
        if self.classification is not None:
            return self.classification
        dist: EmotionDistribution = await self._heuristic.classify(texto)
        return LLMClassification(dist, RiskLevel.NONE, "heurística de prueba")

    async def generate(self, contexto: GenerationContext) -> str:
        self.generate_calls.append(contexto)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_generate:
            raise RuntimeError("LLM caído (simulado)")
        if self.generate_outputs:
            index = min(len(self.generate_calls) - 1, len(self.generate_outputs) - 1)
            return self.generate_outputs[index]
        base = contexto.fragmentos[0].texto if contexto.fragmentos else ""
        return f"Entiendo cómo te sientes. {base} ¿Te sirvió esta sugerencia?"
