from __future__ import annotations

from typing import Protocol

from chatbot.application.dto import AnalysisRecord, ConversationHistory
from chatbot.domain.models import Feedback, Message


class ConversationRepositoryPort(Protocol):
    async def history(self, conversation_id: str, limit: int) -> ConversationHistory: ...

    async def record_turn(
        self,
        user_message: Message,
        assistant_message: Message,
        analysis: AnalysisRecord,
    ) -> None:
        """Persiste el turno completo (mensajes + análisis) de forma atómica."""
        ...

    async def get_analysis(self, trace_id: str) -> AnalysisRecord | None: ...


class FeedbackRepositoryPort(Protocol):
    async def exists(self, trace_id: str) -> bool: ...

    async def save(self, trace_id: str, feedback: Feedback) -> None: ...
