"""Registra la retroalimentación del usuario y actualiza al bandido."""

from __future__ import annotations

import logging

from chatbot.application.dto import FeedbackCommand, FeedbackResult
from chatbot.application.errors import AnalysisNotFound, DuplicateFeedback, FeedbackNotApplicable
from chatbot.application.ports.outbound import (
    ConversationRepositoryPort,
    FeedbackRepositoryPort,
    StrategySelectorPort,
)
from chatbot.domain.models import Feedback

logger = logging.getLogger(__name__)


class RegisterFeedback:
    def __init__(
        self,
        *,
        conversations: ConversationRepositoryPort,
        feedback: FeedbackRepositoryPort,
        selector: StrategySelectorPort,
    ) -> None:
        self._conversations = conversations
        self._feedback = feedback
        self._selector = selector

    async def execute(self, command: FeedbackCommand) -> FeedbackResult:
        analysis = await self._conversations.get_analysis(command.trace_id)
        if analysis is None:
            raise AnalysisNotFound("No existe un análisis con ese trace_id")
        if analysis.estrategia_id is None:
            raise FeedbackNotApplicable("Ese análisis no sugirió ninguna estrategia")
        if await self._feedback.exists(command.trace_id):
            raise DuplicateFeedback("Ya se registró retroalimentación para ese trace_id")

        feedback = Feedback(
            message_id=analysis.assistant_message_id,
            strategy_id=analysis.estrategia_id,
            util=command.util,
            comentario=command.comentario,
        )
        await self._feedback.save(command.trace_id, feedback)

        if analysis.bandit_context is not None:
            try:
                await self._selector.update(analysis.bandit_context, analysis.estrategia_id, feedback.reward)
            except Exception as exc:  # el feedback ya quedó guardado; no se pierde al usuario
                logger.error(
                    "bandido_no_actualizado",
                    extra={"trace_id": command.trace_id, "error": type(exc).__name__},
                )
        return FeedbackResult(command.trace_id, analysis.estrategia_id, feedback.reward)
