"""Raíz de composición: único lugar donde se conectan servicios de dominio y adaptadores.

Cada puerto tiene una fábrica que decide el adaptador SOLO por configuración (`ENCODER_BACKEND`,
`LLM_BACKEND`, `PERSISTENCE_BACKEND`, `RETRIEVER_BACKEND`). Con `stub`/`fake`/`memory` el sistema no necesita
dependencias externas; los reales (HuggingFace, LLM compatible con OpenAI, Postgres, pgvector) se conectan
aquí sin tocar `domain` ni `application`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatbot.adapters.outbound.classifier.encoder_hf import HFEncoder
from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.llm.llm_openai_compatible import OpenAICompatibleLLM
from chatbot.adapters.outbound.llm.prompt_loader import PromptLoader
from chatbot.adapters.outbound.memory.crisis import InMemoryCrisisEventRepository
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase
from chatbot.adapters.outbound.memory.repositories import (
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)
from chatbot.adapters.outbound.memory.selector import RuleBasedSelector
from chatbot.adapters.outbound.memory.system import SystemClock, UuidGenerator
from chatbot.adapters.outbound.persistence.crisis_event_repo import CrisisEventRepository
from chatbot.adapters.outbound.persistence.database import Database, SqlDatabase
from chatbot.adapters.outbound.persistence.supabase_conversation_repo import SupabaseConversationRepository
from chatbot.adapters.outbound.persistence.supabase_feedback_repo import SupabaseFeedbackRepository
from chatbot.adapters.outbound.rag.chunker import ParagraphChunker
from chatbot.adapters.outbound.rag.embedder import HashEmbedder, SentenceTransformerEmbedder
from chatbot.adapters.outbound.rag.pgvector_retriever import PgVectorRetriever
from chatbot.adapters.outbound.risk.rules_risk_detector import RulesRiskDetector
from chatbot.adapters.outbound.strategies.yaml_strategy_catalog import YamlStrategyCatalog
from chatbot.adapters.outbound.validation.response_validator import RuleBasedResponseValidator
from chatbot.application.dto import ENCODER_MODEL, LLM_MODEL, AnalyzeConfig, ResponseTemplates
from chatbot.application.ports.inbound.use_cases import (
    AnalyzeAndRespondPort,
    IngestDocumentsPort,
    ListStrategiesPort,
    RegisterFeedbackPort,
)
from chatbot.application.ports.outbound import (
    ConversationRepositoryPort,
    CrisisEventPort,
    EmbedderPort,
    EmotionClassifierPort,
    FeedbackRepositoryPort,
    LLMPort,
    RiskDetectorPort,
    StrategyCatalogPort,
)
from chatbot.application.use_cases import (
    AnalyzeAndRespond,
    IngestDocuments,
    ListStrategies,
    RegisterFeedback,
)
from chatbot.config.settings import Settings
from chatbot.config.templates_loader import load_response_templates
from chatbot.domain.services import (
    DEFAULT_RULES,
    FusionService,
    RiskPolicy,
    SecondaryEmotionMapper,
)

logger = logging.getLogger(__name__)

#: Tiempo máximo de cada verificación de dependencias en `/ready`.
READINESS_TIMEOUT_S = 3.0


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
    risk_detector: RiskDetectorPort
    templates: ResponseTemplates
    crisis_events: CrisisEventPort
    catalog: StrategyCatalogPort | None = None
    #: Conexión a Postgres (None con `PERSISTENCE_BACKEND=memory`).
    database: SqlDatabase | None = None
    #: Componentes con ciclo de vida, por nombre: "encoder", "embedder", "llm" (los que se hayan creado).
    components: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------------- ciclo de vida
    async def warmup(self) -> None:
        """Carga encoder y embedder reales (los falsos no tienen `warmup`). Si falla, no se arranca."""
        for name in ("encoder", "embedder"):
            warmup = getattr(self.components.get(name), "warmup", None)
            if warmup is not None:
                await warmup()

    async def close(self) -> None:
        """Cierra el cliente HTTP del LLM y el pool de la BD. Intenta cerrar todo aunque algo falle."""
        closers = [getattr(self.components.get("llm"), "aclose", None)]
        if self.database is not None:
            closers.append(self.database.close)
        for close in closers:
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:
                logger.warning("fallo_al_cerrar", extra={"error": type(exc).__name__})

    # ------------------------------------------------------------------------- readiness
    def readiness(self) -> dict[str, str]:
        """Verificaciones síncronas (sin red). `check_readiness` agrega las de BD."""
        version = getattr(self.risk_detector, "version", "desconocida")
        s = self.settings
        return {
            "dominio": "ok",
            "casos_de_uso": "ok",
            "lexico_riesgo": f"ok ({version})",
            "plantillas": "ok (" + "; ".join(f"{k}={v}" for k, v in self.templates.versiones.items()) + ")",
            "encoder": self._component_state("encoder", s.encoder_backend == "stub"),
            "embedder": self._component_state("embedder", s.retriever_backend == "memory"),
        }

    def _component_state(self, name: str, is_fake: bool) -> str:
        if is_fake:
            return "no_aplica"
        is_ready = getattr(self.components.get(name), "is_ready", None)
        return "ok" if is_ready is not None and is_ready() else "no_listo"

    async def check_readiness(self) -> tuple[bool, dict[str, str]]:
        """Verificación completa para `/ready`: (todo_ok, detalle por componente)."""
        checks = self.readiness()
        if self.database is None:
            checks["base_de_datos"] = "no_aplica"
            checks["extension_vector"] = "no_aplica"
        else:
            checks["base_de_datos"] = await self._probe(self.database.ping)
            checks["extension_vector"] = await self._probe(self._vector_check)
        if self.catalog is not None:
            checks["catalogo_estrategias"] = await self._probe(self._catalog_check)
        ok = all(v == "ok" or v.startswith("ok ") or v == "no_aplica" for v in checks.values())
        return ok, checks

    async def _vector_check(self) -> None:
        assert self.database is not None
        if not await self.database.has_vector_extension():
            raise LookupError("extensión vector ausente")

    async def _catalog_check(self) -> None:
        assert self.catalog is not None
        if len(await self.catalog.list_all()) < 1:
            raise LookupError("catálogo vacío")

    @staticmethod
    async def _probe(check: Any) -> str:
        try:
            await asyncio.wait_for(check(), READINESS_TIMEOUT_S)
        except Exception as exc:  # solo el tipo: el mensaje podría incluir datos de conexión
            return f"error: {type(exc).__name__}"
        return "ok"


def _build_secondary_mapper(settings: Settings) -> SecondaryEmotionMapper:
    if settings.secondary_emotions_file:
        rules = json.loads(Path(settings.secondary_emotions_file).read_text(encoding="utf-8"))
        return SecondaryEmotionMapper.from_dicts(rules)
    return SecondaryEmotionMapper(DEFAULT_RULES)


def _build_database(settings: Settings) -> Database | None:
    if settings.persistence_backend == "memory":
        return None
    return Database(settings.database_url, pool_size=settings.db_pool_size)


def _build_encoder(settings: Settings) -> EmotionClassifierPort:
    if settings.encoder_backend == "hf":
        return HFEncoder(
            model_path=settings.encoder_model_path,
            temperature_file=settings.encoder_temperature_file,
            device=settings.encoder_device,
            model_name=settings.encoder_model_name,
        )
    return StubEncoder()


def _build_llm(settings: Settings) -> LLMPort:
    if settings.llm_backend == "openai_compatible":
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            prompts=PromptLoader(settings.prompts_dir),
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    return FakeLLM()


def _build_embedder(settings: Settings) -> EmbedderPort:
    if settings.retriever_backend == "pgvector":
        return SentenceTransformerEmbedder(
            settings.embedding_model_name, settings.embedding_dim, device=settings.encoder_device
        )
    return HashEmbedder()


def _build_knowledge(
    settings: Settings, database: Database | None, embedder: EmbedderPort
) -> InMemoryKnowledgeBase | PgVectorRetriever:
    if settings.retriever_backend == "pgvector":
        assert database is not None  # garantizado por las validaciones de Settings
        return PgVectorRetriever(database, embedder, embedding_dim=settings.embedding_dim)
    return InMemoryKnowledgeBase()


def _build_repositories(
    settings: Settings, database: Database | None
) -> tuple[ConversationRepositoryPort, FeedbackRepositoryPort, CrisisEventPort]:
    if settings.persistence_backend == "postgres":
        assert database is not None
        return (
            SupabaseConversationRepository(database, store_user_text=settings.store_user_text),
            SupabaseFeedbackRepository(database, store_user_text=settings.store_user_text),
            CrisisEventRepository(database),
        )
    return InMemoryConversationRepository(), InMemoryFeedbackRepository(), InMemoryCrisisEventRepository()


def build_container(settings: Settings) -> Container:
    fusion = FusionService(
        {ENCODER_MODEL: settings.fusion_weight_encoder, LLM_MODEL: settings.fusion_weight_llm}
    )
    policy = RiskPolicy(history_window=settings.risk_history_window)
    mapper = _build_secondary_mapper(settings)

    # --- adaptadores: cada fábrica elige el real o el falso según la configuración
    database = _build_database(settings)
    encoder = _build_encoder(settings)
    llm = _build_llm(settings)
    embedder = _build_embedder(settings)
    knowledge = _build_knowledge(settings, database, embedder)
    conversations, feedback_repo, crisis_events = _build_repositories(settings, database)
    # Fail-fast: sin catálogo válido el servicio no arranca. `seed_strategies` queda solo para pruebas.
    catalog = YamlStrategyCatalog.from_file(settings.strategies_file)
    selector = RuleBasedSelector()
    # Fail-fast: sin léxico o plantillas válidos el servicio no arranca (ruta de crisis sin red de seguridad).
    risk_detector = RulesRiskDetector.from_file(settings.risk_lexicon_file)
    templates = load_response_templates(settings.templates_dir)

    analyze = AnalyzeAndRespond(
        encoder=encoder,
        llm=llm,
        risk_detector=risk_detector,
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
        templates=templates,
        crisis_events=crisis_events,
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
            conversations=conversations, feedback=feedback_repo, selector=selector
        ),
        ingest_documents=IngestDocuments(chunker=ParagraphChunker(), embedder=embedder, store=knowledge),
        list_strategies=ListStrategies(catalog=catalog),
        risk_detector=risk_detector,
        templates=templates,
        crisis_events=crisis_events,
        catalog=catalog,
        database=database,
        components={"encoder": encoder, "embedder": embedder, "llm": llm},
    )
