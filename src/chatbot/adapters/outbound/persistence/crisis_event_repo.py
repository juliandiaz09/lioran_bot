"""Registro de eventos de crisis en la tabla `crisis_events`. Cumple `CrisisEventPort`.

Mantiene el log WARNING estructurado de la Fase 4 (`evento_crisis`) y se escribe ANTES de tocar la BD: si la
BD falla, el aviso queda al menos en el log. Sin texto del usuario: solo ids seudónimos, categorías y la
fuente del riesgo. EXTENSIÓN FUTURA (no implementada): avisar a humanos con un webhook (n8n / Bienestar).
"""

from __future__ import annotations

import logging

from chatbot.adapters.outbound.persistence._sql_utils import aware, to_json
from chatbot.adapters.outbound.persistence.database import SqlDatabase
from chatbot.application.dto import CrisisEvent

logger = logging.getLogger(__name__)

_INSERT = """
INSERT INTO crisis_events (trace_id, conversation_id, user_ref, categorias, fuente_riesgo, created_at)
VALUES (:trace_id, :conversation_id, :user_ref, CAST(:categorias AS jsonb), :fuente_riesgo, :created_at)
ON CONFLICT (trace_id) DO NOTHING
"""


class CrisisEventRepository:
    def __init__(self, database: SqlDatabase) -> None:
        self._db = database

    async def register(self, event: CrisisEvent) -> None:
        logger.warning(
            "evento_crisis",
            extra={
                "trace_id": event.trace_id,
                "conversation_id": event.conversation_id,
                "user_ref": event.user_ref,
                "categorias": list(event.categorias),
                "fuente_riesgo": event.fuente_riesgo,
            },
        )
        async with self._db.transaction() as tx:
            await tx.execute(
                _INSERT,
                {
                    "trace_id": event.trace_id,
                    "conversation_id": event.conversation_id,
                    "user_ref": event.user_ref,
                    "categorias": to_json(list(event.categorias)),
                    "fuente_riesgo": event.fuente_riesgo,
                    "created_at": aware(event.timestamp),
                },
            )
