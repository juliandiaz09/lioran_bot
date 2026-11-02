"""Errores de la capa de aplicación (heredan de DomainError para el mapeo uniforme de la API)."""

from __future__ import annotations

from chatbot.domain.errors import DomainError


class EmptyMessage(DomainError):
    codigo = "mensaje_vacio"


class AnalysisNotFound(DomainError):
    codigo = "analisis_no_encontrado"


class FeedbackNotApplicable(DomainError):
    codigo = "feedback_no_aplicable"


class DuplicateFeedback(DomainError):
    codigo = "feedback_duplicado"


class IngestError(DomainError):
    codigo = "ingesta_invalida"
