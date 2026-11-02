from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chatbot.application.dto import RetrievedChunk, ValidationResult
from chatbot.domain.models import RiskLevel


class ResponseValidatorPort(Protocol):
    def validate(
        self, respuesta: str, fragmentos: Sequence[RetrievedChunk], riesgo: RiskLevel
    ) -> ValidationResult: ...
