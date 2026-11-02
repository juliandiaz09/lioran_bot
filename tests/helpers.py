"""Arnés de pruebas: ensambla los casos de uso con dobles de los puertos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase, InMemoryStrategyCatalog
from chatbot.adapters.outbound.memory.repositories import (
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)
from chatbot.adapters.outbound.memory.seed import seed_documents, seed_strategies
from chatbot.adapters.outbound.memory.selector import RuleBasedSelector
from chatbot.adapters.outbound.rag.chunker import ParagraphChunker
from chatbot.adapters.outbound.rag.embedder import HashEmbedder
from chatbot.adapters.outbound.risk.basic_risk_detector import BasicRiskDetector
from chatbot.adapters.outbound.validation.response_validator import RuleBasedResponseValidator
from chatbot.application.dto import (
    ENCODER_MODEL,
    LLM_MODEL,
    AnalyzeCommand,
    AnalyzeConfig,
    IngestCommand,
    LLMClassification,
)
from chatbot.application.use_cases import AnalyzeAndRespond, IngestDocuments, RegisterFeedback
from chatbot.domain.models import Emotion, EmotionDistribution, RiskLevel
from chatbot.domain.services import FusionService, RiskPolicy, SecondaryEmotionMapper

MIEDO = EmotionDistribution(
    {
        Emotion.MIEDO: 0.70,
        Emotion.TRISTEZA: 0.10,
        Emotion.NEUTRAL: 0.10,
        Emotion.IRA: 0.05,
        Emotion.ALEGRIA: 0.02,
        Emotion.SORPRESA: 0.02,
        Emotion.ASCO: 0.01,
    }
)
TEXTO_NORMAL = "Tengo mucho miedo por el parcial de mañana"
TEXTO_CRISIS = "ya no quiero vivir más"


class FixedClock:
    def __init__(self, hour: int = 21) -> None:
        self._now = datetime(2026, 10, 26, hour, 0, tzinfo=timezone(timedelta(hours=-5)))

    def now(self) -> datetime:
        return self._now


class CounterIds:
    def __init__(self) -> None:
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"id-{self._n}"


def command(texto: str = TEXTO_NORMAL, trace: str = "trace-1", conv: str = "conv-1") -> AnalyzeCommand:
    return AnalyzeCommand(conversation_id=conv, user_ref="u-seudonimo", texto=texto, trace_id=trace)


@dataclass
class Harness:
    use_case: AnalyzeAndRespond
    encoder: StubEncoder
    llm: FakeLLM
    repo: InMemoryConversationRepository
    kb: InMemoryKnowledgeBase
    selector: RuleBasedSelector
    feedback_repo: InMemoryFeedbackRepository
    register_feedback: RegisterFeedback


async def make_harness(
    *,
    encoder: StubEncoder | None = None,
    llm: FakeLLM | None = None,
    repo: InMemoryConversationRepository | None = None,
    kb: InMemoryKnowledgeBase | None = None,
    selector: RuleBasedSelector | None = None,
    config: AnalyzeConfig | None = None,
    seed_kb: bool = True,
    hour: int = 21,
) -> Harness:
    encoder = encoder or StubEncoder(fixed=MIEDO)
    llm = llm or FakeLLM(LLMClassification(MIEDO, RiskLevel.NONE, "ok"))
    repo = repo or InMemoryConversationRepository()
    kb = kb or InMemoryKnowledgeBase()
    selector = selector or RuleBasedSelector()
    feedback_repo = InMemoryFeedbackRepository()
    if seed_kb:
        await IngestDocuments(chunker=ParagraphChunker(), embedder=HashEmbedder(), store=kb).execute(
            IngestCommand(tuple(seed_documents()))
        )
    use_case = AnalyzeAndRespond(
        encoder=encoder,
        llm=llm,
        risk_detector=BasicRiskDetector(),
        risk_policy=RiskPolicy(history_window=5),
        fusion=FusionService({ENCODER_MODEL: 0.5, LLM_MODEL: 0.5}),
        secondary_mapper=SecondaryEmotionMapper(),
        selector=selector,
        catalog=InMemoryStrategyCatalog(seed_strategies()),
        retriever=kb,
        validator=RuleBasedResponseValidator(),
        repository=repo,
        clock=FixedClock(hour),
        ids=CounterIds(),
        config=config,
    )
    return Harness(
        use_case,
        encoder,
        llm,
        repo,
        kb,
        selector,
        feedback_repo,
        RegisterFeedback(conversations=repo, feedback=feedback_repo, selector=selector),
    )
