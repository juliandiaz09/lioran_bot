"""Errores propios del adaptador LLM. Son excepciones normales: `_guard` las convierte en degradación."""

from __future__ import annotations


class LLMError(Exception):
    """Base de los errores del adaptador LLM."""


class LLMInvalidResponse(LLMError):
    """El modelo respondió, pero no con el JSON/contenido esperado (tras el reintento)."""


class LLMUnavailable(LLMError):
    """No se pudo obtener respuesta: reintentos agotados ante 429/5xx/timeout/conexión."""


class LLMRequestRejected(LLMError):
    """El proveedor rechazó la petición con un 4xx distinto de 429 (no se reintenta)."""
