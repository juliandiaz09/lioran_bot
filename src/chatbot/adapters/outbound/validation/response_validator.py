"""Validador de respuestas por reglas (antes de devolverlas al usuario)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from chatbot.application.dto import RetrievedChunk, ValidationResult
from chatbot.domain.models import RiskLevel
from chatbot.domain.services.secondary_emotions import normalize_text

_DIAGNOSTICO = (
    "tienes depresion",
    "tienes ansiedad",
    "sufres de",
    "padeces",
    "diagnostico",
    "eres bipolar",
    "trastorno de",
    "tienes un trastorno",
)
_MEDICACION = (
    "medicacion",
    "medicamento",
    "antidepresivo",
    "ansiolitico",
    "pastilla",
    "fluoxetina",
    "sertralina",
    "clonazepam",
    "alprazolam",
    "dosis",
)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-().]{5,}\d)(?!\d)")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


class RuleBasedResponseValidator:
    def __init__(self, max_chars: int = 1200, require_feedback_question: bool = True) -> None:
        self._max = max_chars
        self._feedback_q = require_feedback_question

    def validate(
        self, respuesta: str, fragmentos: Sequence[RetrievedChunk], riesgo: RiskLevel
    ) -> ValidationResult:
        razones: list[str] = []
        text = respuesta.strip()
        norm = normalize_text(text)
        if not text:
            return ValidationResult(False, ("respuesta vacía",))
        if len(text) > self._max:
            razones.append(f"supera {self._max} caracteres; acórtala")
        if any(t in norm for t in _DIAGNOSTICO):
            razones.append("contiene un diagnóstico clínico; no diagnostiques")
        if any(t in norm for t in _MEDICACION):
            razones.append("menciona medicación; no recomiendes medicamentos")
        context_digits = " ".join(_digits(c.texto) for c in fragmentos)
        for match in _PHONE.findall(text):
            if _digits(match) not in context_digits:
                razones.append("incluye un teléfono que no está en el contexto recuperado")
                break
        if self._feedback_q and not text.rstrip().endswith("?"):
            razones.append("debe cerrar con una pregunta de retroalimentación")
        return ValidationResult(not razones, tuple(razones))
