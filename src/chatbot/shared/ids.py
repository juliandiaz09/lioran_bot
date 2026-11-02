"""Utilidades neutras para identificadores."""

from __future__ import annotations

import uuid


def new_trace_id() -> str:
    return str(uuid.uuid4())


def parse_trace_id(value: str | None) -> str:
    """Devuelve el trace_id recibido si es un UUID válido; si no, genera uno nuevo."""
    if value:
        try:
            return str(uuid.UUID(value.strip()))
        except ValueError:
            pass
    return new_trace_id()
