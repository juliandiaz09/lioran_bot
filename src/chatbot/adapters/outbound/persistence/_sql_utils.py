"""Utilidades comunes de los repositorios SQL (sin dependencias externas)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def aware(moment: datetime) -> datetime:
    """`timestamptz` exige fecha con zona; una fecha ingenua se interpreta como UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Any) -> Any:
    """asyncpg/SQLAlchemy devuelven jsonb ya decodificado; si llegara como texto, se decodifica."""
    return json.loads(value) if isinstance(value, str | bytes) else value
