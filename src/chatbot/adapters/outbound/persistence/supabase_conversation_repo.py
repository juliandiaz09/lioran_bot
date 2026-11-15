"""Repositorio de conversaciones sobre Postgres (Supabase). Cumple `ConversationRepositoryPort`.

Privacidad:
- Solo `user_ref` seudónimo. Ni nombre ni usuario de Telegram.
- Con `store_user_text=False` el mensaje del USUARIO se guarda como `[omitido]` (la columna `longitud`
  conserva el largo real). La respuesta del asistente sí se guarda.
  Consecuencia: el historial que ve el LLM ya no
  incluye lo que el usuario escribió en turnos anteriores.
- Ningún log lleva el texto ni los parámetros de las consultas.
"""

from __future__ import annotations

from typing import Any

from chatbot.adapters.outbound.persistence._sql_utils import aware, from_json, to_json
from chatbot.adapters.outbound.persistence.database import SqlDatabase
from chatbot.application.dto import AnalysisRecord, ConversationHistory, SourceRef
from chatbot.domain.models import (
    BanditContext,
    Emotion,
    EmotionDistribution,
    Message,
    RiskLevel,
    Role,
)

OMITTED_TEXT = "[omitido]"

_INSERT_CONVERSATION = """
INSERT INTO conversations (id, user_ref, created_at)
VALUES (:id, :user_ref, :created_at)
ON CONFLICT (id) DO NOTHING
"""

_INSERT_MESSAGE = """
INSERT INTO messages (id, conversation_id, rol, texto, longitud, created_at)
VALUES (:id, :conversation_id, :rol, :texto, :longitud, :created_at)
"""

_INSERT_ANALYSIS = """
INSERT INTO analyses (
    trace_id, conversation_id, user_ref, user_message_id, assistant_message_id, distribucion,
    emociones_secundarias, desacuerdo_modelos, riesgo, riesgo_fuente, estrategia_id, bandit_context,
    es_crisis, degradado, fuentes, created_at
) VALUES (
    :trace_id, :conversation_id, :user_ref, :user_message_id, :assistant_message_id,
    CAST(:distribucion AS jsonb), CAST(:emociones_secundarias AS jsonb), :desacuerdo_modelos, :riesgo,
    :riesgo_fuente, :estrategia_id, CAST(:bandit_context AS jsonb), :es_crisis, :degradado,
    CAST(:fuentes AS jsonb), :created_at
)
"""

_SELECT_MESSAGES = """
SELECT id, conversation_id, texto, rol, created_at
FROM messages WHERE conversation_id = :conversation_id
ORDER BY seq DESC LIMIT :n
"""

_SELECT_RECENT_ANALYSES = """
SELECT riesgo, estrategia_id
FROM analyses WHERE conversation_id = :conversation_id
ORDER BY seq DESC LIMIT :n
"""

_SELECT_ANALYSIS = """
SELECT trace_id, conversation_id, user_ref, user_message_id, assistant_message_id, distribucion,
       emociones_secundarias, desacuerdo_modelos, riesgo, riesgo_fuente, estrategia_id, bandit_context,
       es_crisis, degradado, fuentes, created_at
FROM analyses WHERE trace_id = :trace_id
"""


class SupabaseConversationRepository:
    def __init__(self, database: SqlDatabase, *, store_user_text: bool = True) -> None:
        self._db = database
        self._store_user_text = store_user_text

    async def history(self, conversation_id: str, limit: int) -> ConversationHistory:
        """Mensajes de los últimos `limit` turnos y riesgos/estrategias de los últimos `limit` análisis."""
        if limit < 1:
            return ConversationHistory()
        async with self._db.transaction() as tx:
            message_rows = await tx.fetch_all(
                _SELECT_MESSAGES, {"conversation_id": conversation_id, "n": limit * 2}
            )
            analysis_rows = await tx.fetch_all(
                _SELECT_RECENT_ANALYSES, {"conversation_id": conversation_id, "n": limit}
            )
        message_rows.reverse()  # de más nuevo a más viejo -> orden cronológico
        analysis_rows.reverse()
        return ConversationHistory(
            messages=tuple(self._to_message(r) for r in message_rows),
            risk_levels=tuple(RiskLevel(int(r["riesgo"])) for r in analysis_rows),
            strategy_ids=tuple(r["estrategia_id"] for r in analysis_rows if r["estrategia_id"]),
        )

    async def record_turn(
        self, user_message: Message, assistant_message: Message, analysis: AnalysisRecord
    ) -> None:
        """UNA transacción: conversación (si no existe) + dos mensajes + análisis. O todo o nada."""
        async with self._db.transaction() as tx:
            await tx.execute(
                _INSERT_CONVERSATION,
                {
                    "id": analysis.conversation_id,
                    "user_ref": analysis.user_ref,
                    "created_at": aware(user_message.timestamp),
                },
            )
            await tx.execute(_INSERT_MESSAGE, self._message_params(user_message, redact=True))
            await tx.execute(_INSERT_MESSAGE, self._message_params(assistant_message, redact=False))
            await tx.execute(_INSERT_ANALYSIS, self._analysis_params(analysis))

    async def get_analysis(self, trace_id: str) -> AnalysisRecord | None:
        async with self._db.transaction() as tx:
            row = await tx.fetch_one(_SELECT_ANALYSIS, {"trace_id": trace_id})
        return self._to_analysis(row) if row is not None else None

    # ------------------------------------------------------------------------- mapeo fila <-> dominio
    def _message_params(self, message: Message, *, redact: bool) -> dict[str, Any]:
        texto = OMITTED_TEXT if redact and not self._store_user_text else message.texto
        return {
            "id": message.id,
            "conversation_id": message.conversation_id,
            "rol": message.rol.value,
            "texto": texto,
            "longitud": len(message.texto),
            "created_at": aware(message.timestamp),
        }

    @staticmethod
    def _analysis_params(a: AnalysisRecord) -> dict[str, Any]:
        return {
            "trace_id": a.trace_id,
            "conversation_id": a.conversation_id,
            "user_ref": a.user_ref,
            "user_message_id": a.user_message_id,
            "assistant_message_id": a.assistant_message_id,
            "distribucion": to_json(a.distribucion.to_dict()) if a.distribucion else None,
            "emociones_secundarias": to_json(list(a.emociones_secundarias)),
            "desacuerdo_modelos": a.desacuerdo_modelos,
            "riesgo": int(a.riesgo),
            "riesgo_fuente": a.riesgo_fuente,
            "estrategia_id": a.estrategia_id,
            "bandit_context": to_json(a.bandit_context.to_dict()) if a.bandit_context else None,
            "es_crisis": a.es_crisis,
            "degradado": a.degradado,
            "fuentes": to_json([{"doc_id": s.doc_id, "titulo": s.titulo} for s in a.fuentes]),
            "created_at": aware(a.timestamp),
        }

    @staticmethod
    def _to_message(row: dict[str, Any]) -> Message:
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            texto=row["texto"],
            rol=Role(row["rol"]),
            timestamp=aware(row["created_at"]),
        )

    @staticmethod
    def _to_analysis(row: dict[str, Any]) -> AnalysisRecord:
        distribution = from_json(row["distribucion"])
        context = from_json(row["bandit_context"])
        return AnalysisRecord(
            trace_id=row["trace_id"],
            conversation_id=row["conversation_id"],
            user_ref=row["user_ref"],
            user_message_id=row["user_message_id"],
            assistant_message_id=row["assistant_message_id"],
            distribucion=(
                EmotionDistribution({Emotion(k): float(v) for k, v in distribution.items()})
                if distribution
                else None
            ),
            emociones_secundarias=tuple(from_json(row["emociones_secundarias"])),
            desacuerdo_modelos=float(row["desacuerdo_modelos"]),
            riesgo=RiskLevel(int(row["riesgo"])),
            riesgo_fuente=row["riesgo_fuente"],
            estrategia_id=row["estrategia_id"],
            bandit_context=BanditContext.from_dict(context) if context else None,
            es_crisis=bool(row["es_crisis"]),
            degradado=bool(row["degradado"]),
            fuentes=tuple(SourceRef(s["doc_id"], s["titulo"]) for s in from_json(row["fuentes"])),
            timestamp=aware(row["created_at"]),
        )
