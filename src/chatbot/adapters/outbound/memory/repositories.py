"""Repositorios en memoria (desarrollo y pruebas). Los reales (Supabase) llegan en la Fase 5."""

from __future__ import annotations

from chatbot.application.dto import AnalysisRecord, ConversationHistory
from chatbot.domain.models import Feedback, Message


class InMemoryConversationRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: dict[str, list[Message]] = {}
        self.analyses: dict[str, AnalysisRecord] = {}
        self._order: dict[str, list[str]] = {}  # conversation_id -> trace_ids

    async def history(self, conversation_id: str, limit: int) -> ConversationHistory:
        if self.fail:
            raise RuntimeError("BD caída (simulado)")
        msgs = self.messages.get(conversation_id, [])[-limit * 2 :]
        records = [self.analyses[t] for t in self._order.get(conversation_id, [])][-limit:]
        return ConversationHistory(
            messages=tuple(msgs),
            risk_levels=tuple(r.riesgo for r in records),
            strategy_ids=tuple(r.estrategia_id for r in records if r.estrategia_id),
        )

    async def record_turn(
        self, user_message: Message, assistant_message: Message, analysis: AnalysisRecord
    ) -> None:
        if self.fail:
            raise RuntimeError("BD caída (simulado)")
        cid = analysis.conversation_id
        self.messages.setdefault(cid, []).extend([user_message, assistant_message])
        self.analyses[analysis.trace_id] = analysis
        self._order.setdefault(cid, []).append(analysis.trace_id)

    async def get_analysis(self, trace_id: str) -> AnalysisRecord | None:
        return self.analyses.get(trace_id)


class InMemoryFeedbackRepository:
    def __init__(self) -> None:
        self.items: dict[str, Feedback] = {}

    async def exists(self, trace_id: str) -> bool:
        return trace_id in self.items

    async def save(self, trace_id: str, feedback: Feedback) -> None:
        self.items[trace_id] = feedback
