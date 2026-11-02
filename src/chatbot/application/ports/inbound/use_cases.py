"""Interfaces de los casos de uso: lo único que los routers de FastAPI conocen."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chatbot.application.dto import (
    AnalysisResult,
    AnalyzeCommand,
    FeedbackCommand,
    FeedbackResult,
    IngestCommand,
    IngestResult,
)
from chatbot.domain.models import Strategy


class AnalyzeAndRespondPort(Protocol):
    async def execute(self, command: AnalyzeCommand) -> AnalysisResult: ...


class RegisterFeedbackPort(Protocol):
    async def execute(self, command: FeedbackCommand) -> FeedbackResult: ...


class IngestDocumentsPort(Protocol):
    async def execute(self, command: IngestCommand) -> IngestResult: ...


class ListStrategiesPort(Protocol):
    async def execute(self) -> Sequence[Strategy]: ...
