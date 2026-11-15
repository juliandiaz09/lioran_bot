"""Registro de eventos de crisis en memoria + log WARNING estructurado (`evento_crisis`).

Sin texto del usuario: solo identificadores seudónimos, categorías y fuente del riesgo. La versión SQL
(tabla `crisis_events`) llega en la Fase 5. EXTENSIÓN FUTURA (no implementada, fuera del alcance de la
Fase 4): avisar a humanos con un webhook a n8n/Bienestar; el protocolo de seguimiento (quién, en qué
plazo) lo define Bienestar.
"""

from __future__ import annotations

import logging

from chatbot.application.dto import CrisisEvent

logger = logging.getLogger(__name__)


class InMemoryCrisisEventRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[CrisisEvent] = []

    async def register(self, event: CrisisEvent) -> None:
        if self.fail:
            raise RuntimeError("Registro de crisis caído (simulado)")
        self.events.append(event)
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
