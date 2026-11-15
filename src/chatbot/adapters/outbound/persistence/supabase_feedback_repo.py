"""Repositorio de feedback sobre Postgres. Cumple `FeedbackRepositoryPort`.

`trace_id` es la clave primaria: un solo feedback por análisis. Un duplicado hace fallar `save` (el caso de
uso `RegisterFeedback` ya consulta `exists` antes, esto es la red de seguridad ante carreras).

Con `store_user_text=False` el comentario (texto libre del usuario) tampoco se guarda.
"""

from __future__ import annotations

from chatbot.adapters.outbound.persistence.database import SqlDatabase
from chatbot.domain.models import Feedback

_INSERT = """
INSERT INTO feedback (trace_id, message_id, strategy_id, util_bool, util_escala, comentario, reward)
VALUES (:trace_id, :message_id, :strategy_id, :util_bool, :util_escala, :comentario, :reward)
"""


class SupabaseFeedbackRepository:
    def __init__(self, database: SqlDatabase, *, store_user_text: bool = True) -> None:
        self._db = database
        self._store_user_text = store_user_text

    async def exists(self, trace_id: str) -> bool:
        async with self._db.transaction() as tx:
            row = await tx.fetch_one("SELECT 1 AS ok FROM feedback WHERE trace_id = :t", {"t": trace_id})
        return row is not None

    async def save(self, trace_id: str, feedback: Feedback) -> None:
        is_bool = isinstance(feedback.util, bool)
        async with self._db.transaction() as tx:
            await tx.execute(
                _INSERT,
                {
                    "trace_id": trace_id,
                    "message_id": feedback.message_id,
                    "strategy_id": feedback.strategy_id,
                    "util_bool": feedback.util if is_bool else None,
                    "util_escala": None if is_bool else int(feedback.util),
                    "comentario": feedback.comentario if self._store_user_text else None,
                    "reward": feedback.reward,
                },
            )
