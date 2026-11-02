"""Errores del dominio. Sin dependencias externas."""

from __future__ import annotations


class DomainError(Exception):
    """Base de todos los errores de dominio. `codigo` se usa en la respuesta uniforme de la API."""

    codigo = "error_dominio"

    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje


class InvalidEmotionDistribution(DomainError):
    codigo = "distribucion_invalida"


class InvalidFusionConfig(DomainError):
    codigo = "fusion_invalida"


class InvalidStrategy(DomainError):
    codigo = "estrategia_invalida"


class InvalidConversation(DomainError):
    codigo = "conversacion_invalida"


class InvalidFeedback(DomainError):
    codigo = "feedback_invalido"


class InvalidRule(DomainError):
    codigo = "regla_invalida"
