"""Raíz de composición: único lugar donde se conectan servicios de dominio y adaptadores.

Los adaptadores se eligen por configuración (`ENCODER_BACKEND`, `LLM_BACKEND`). En la Fase 3
solo existen los falsos/en memoria (`stub`/`fake`); los reales (HuggingFace, LLM compatible
con OpenAI, pgvector, Supabase) se añaden en la Fase 5 sin tocar `domain` ni `application`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase, InMemoryStrategyCatalog
from chatbot.adapters.outbound.memory.repositories import (
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)
from chatbot.adapters.outbound.memory.seed import seed_strategies
from chatbot.adapters.outbound.memory.selector import RuleBasedSelector
from chatbot.adapters.outbound.memory.system import SystemClock, UuidGenerator
from chatbot.adapters.outbound.rag.chunker import ParagraphChunker
from chatbot.adapters.outbound.rag.embedder import HashEmbedder
from chatbot.adapters.outbound.risk.basic_risk_detector import BasicRiskDetector
from chatbot.adapters.outbound.validation.response_validator import RuleBasedResponseValidator
from chatbot.application.dto import ENCODER_MODEL, LLM_MODEL, AnalyzeConfig
from chatbot.application.ports.inbound.use_cases import (
    AnalyzeAndRespondPort,
    IngestDocumentsPort,
    ListStrategiesPort,
    RegisterFeedbackPort,
)
from chatbot.application.use_cases import (
    AnalyzeAndRespond,
    IngestDocuments,
    ListStrategies,
    RegisterFeedback,
)
from chatbot.config.settings import Settings
from chatbot.domain.services import (
    DEFAULT_RULES,
    FusionService,
    RiskPolicy,
    SecondaryEmotionMapper,
)


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    fusion_service: FusionService
    risk_policy: RiskPolicy
    secondary_mapper: SecondaryEmotionMapper
    analyze: AnalyzeAndRespondPort
    register_feedback: RegisterFeedbackPort
    ingest_documents: IngestDocumentsPort
    list_strategies: ListStrategiesPort

    def readiness(self) -> dict[str, str]:
        """Estado de los componentes. Se amplía con BD y modelos en la Fase 5."""
        return {
            "dominio": "ok",
            "casos_de_uso": "ok",
            "base_de_datos": "pendiente_fase_5",
            "modelos": "pendiente_fase_5",
        }


def _build_secondary_mapper(settings: Settings) -> SecondaryEmotionMapper:
    if settings.secondary_emotions_file:
        rules = json.loads(Path(settings.secondary_emotions_file).read_text(encoding="utf-8"))
        return SecondaryEmotionMapper.from_dicts(rules)
    return SecondaryEmotionMapper(DEFAULT_RULES)


def build_container(settings: Settings) -> Container:
    if settings.encoder_backend != "stub" or settings.llm_backend != "fake":
        raise NotImplementedError("Los adaptadores reales (encoder HF, LLM, BD) llegan en la Fase 5")

    fusion = FusionService(
        {ENCODER_MODEL: settings.fusion_weight_encoder, LLM_MODEL: settings.fusion_weight_llm}
    )
    policy = RiskPolicy(history_window=settings.risk_history_window)
    mapper = _build_secondary_mapper(settings)

    # --- adaptadores falsos / en memoria
    knowledge = InMemoryKnowledgeBase()
    catalog = InMemoryStrategyCatalog(seed_strategies())
    conversations = InMemoryConversationRepository()
    selector = RuleBasedSelector()

    analyze = AnalyzeAndRespond(
        encoder=StubEncoder(),
        llm=FakeLLM(),
        risk_detector=BasicRiskDetector(),
        risk_policy=policy,
        fusion=fusion,
        secondary_mapper=mapper,
        selector=selector,
        catalog=catalog,
        retriever=knowledge,
        validator=RuleBasedResponseValidator(),
        repository=conversations,
        clock=SystemClock(),
        ids=UuidGenerator(),
        config=AnalyzeConfig(
            max_text_length=settings.max_message_length,
            history_window=settings.risk_history_window,
            rag_top_k=settings.rag_top_k,
            classification_timeout_s=settings.classification_timeout_s,
            generation_timeout_s=settings.generation_timeout_s,
            persistence_timeout_s=settings.persistence_timeout_s,
            crisis_lookup_timeout_s=settings.crisis_lookup_timeout_s,
            max_regenerations=settings.max_regenerations,
        ),
    )
    return Container(
        settings=settings,
        fusion_service=fusion,
        risk_policy=policy,
        secondary_mapper=mapper,
        analyze=analyze,
        register_feedback=RegisterFeedback(
            conversations=conversations, feedback=InMemoryFeedbackRepository(), selector=selector
        ),
        ingest_documents=IngestDocuments(
            chunker=ParagraphChunker(), embedder=HashEmbedder(), store=knowledge
        ),
        list_strategies=ListStrategies(catalog=catalog),
    )
