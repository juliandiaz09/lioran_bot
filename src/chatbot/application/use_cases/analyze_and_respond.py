"""Caso de uso principal: analizar el mensaje y responder (sección 7.1 de la especificación).

Orden del flujo:
  1 normalizar · 2 riesgo por reglas (HIGH => crisis, sin LLM) · 3 clasificar (encoder ‖ LLM)
  + fusión · 4 riesgo final · 5 estrategia (bandido) · 6 RAG · 7 generar · 8 validar
  (1 regeneración, luego plantilla segura) · 9 persistir · 10 devolver.

Resiliencia: ningún fallo de un modelo o de la persistencia produce una excepción hacia el
usuario final; se degrada (un solo modelo, plantilla segura). Los logs no incluyen el texto.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TypeVar

from chatbot.application.dto import (
    ENCODER_MODEL,
    LLM_MODEL,
    AnalysisRecord,
    AnalysisResult,
    AnalyzeCommand,
    AnalyzeConfig,
    ConversationHistory,
    GenerationContext,
    RetrievalQuery,
    RetrievedChunk,
    SourceRef,
    StrategyRef,
)
from chatbot.application.errors import EmptyMessage
from chatbot.application.ports.outbound import (
    ClockPort,
    ConversationRepositoryPort,
    EmotionClassifierPort,
    IdGeneratorPort,
    LLMPort,
    ResponseValidatorPort,
    RetrieverPort,
    RiskDetectorPort,
    StrategyCatalogPort,
    StrategySelectorPort,
)
from chatbot.application.templates import SAFE_TEMPLATE, render_crisis_response
from chatbot.domain.errors import DomainError
from chatbot.domain.models import (
    BanditContext,
    Emotion,
    EmotionDistribution,
    Message,
    RiskLevel,
    Role,
    Strategy,
)
from chatbot.domain.services import (
    FusionService,
    RiskAssessment,
    RiskPolicy,
    SecondaryEmotionMapper,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _Classification:
    distribution: EmotionDistribution | None = None
    disagreement: float = 0.0
    llm_risk: RiskLevel | None = None
    models_used: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Outcome:
    respuesta: str
    assessment: RiskAssessment
    classification: _Classification
    secondary: tuple[str, ...]
    strategy: Strategy | None
    bandit_context: BanditContext | None
    fuentes: tuple[SourceRef, ...]
    es_crisis: bool
    degradado: bool


class AnalyzeAndRespond:
    def __init__(
        self,
        *,
        encoder: EmotionClassifierPort,
        llm: LLMPort,
        risk_detector: RiskDetectorPort,
        risk_policy: RiskPolicy,
        fusion: FusionService,
        secondary_mapper: SecondaryEmotionMapper,
        selector: StrategySelectorPort,
        catalog: StrategyCatalogPort,
        retriever: RetrieverPort,
        validator: ResponseValidatorPort,
        repository: ConversationRepositoryPort,
        clock: ClockPort,
        ids: IdGeneratorPort,
        config: AnalyzeConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._llm = llm
        self._risk_detector = risk_detector
        self._risk_policy = risk_policy
        self._fusion = fusion
        self._secondary = secondary_mapper
        self._selector = selector
        self._catalog = catalog
        self._retriever = retriever
        self._validator = validator
        self._repo = repository
        self._clock = clock
        self._ids = ids
        self._cfg = config or AnalyzeConfig()

    # ------------------------------------------------------------------------- flujo
    async def execute(self, command: AnalyzeCommand) -> AnalysisResult:
        started = time.perf_counter()
        texto = self._normalize(command.texto)  # 1
        rules = self._risk_detector.detect(texto)  # 2

        if rules.level.is_crisis:
            outcome = await self._crisis_outcome(
                self._risk_policy.assess(rules.level), _Classification(), texto, command
            )
            return await self._finish(command, texto, outcome, started)

        history = (
            await self._guard(
                self._repo.history(command.conversation_id, self._cfg.history_window),
                self._cfg.persistence_timeout_s,
                "historial",
                command,
            )
            or ConversationHistory()
        )

        classification = await self._classify(texto, history, command)  # 3
        assessment = self._risk_policy.assess(  # 4
            rules.level, classification.llm_risk, history.risk_levels
        )
        if assessment.is_crisis:
            outcome = await self._crisis_outcome(assessment, classification, texto, command)
        else:
            outcome = await self._respond(assessment, classification, texto, history, command)
        return await self._finish(command, texto, outcome, started)  # 9, 10

    # ------------------------------------------------------------------------- 1. normalizar
    def _normalize(self, texto: str) -> str:
        """Solo espacios y longitud máxima: errores ortográficos y emojis son señal."""
        limpio = " ".join(texto.split())[: self._cfg.max_text_length].strip()
        if not limpio:
            raise EmptyMessage("El mensaje está vacío")
        return limpio

    # ------------------------------------------------------------------------- 3. clasificar
    async def _classify(
        self, texto: str, history: ConversationHistory, command: AnalyzeCommand
    ) -> _Classification:
        t = self._cfg.classification_timeout_s
        enc, llm = await asyncio.gather(
            self._guard(self._encoder.classify(texto), t, "encoder", command),
            self._guard(self._llm.classify_emotion(texto, history.messages), t, "llm_clasif", command),
        )
        distributions: dict[str, EmotionDistribution] = {}
        if enc is not None:
            distributions[ENCODER_MODEL] = enc
        if llm is not None:
            distributions[LLM_MODEL] = llm.distribucion
        llm_risk = llm.riesgo if llm is not None else None
        if not distributions:
            return _Classification(llm_risk=llm_risk)
        try:
            fused = self._fusion.fuse(distributions)
        except DomainError:
            logger.warning("fusion_fallida", extra={"trace_id": command.trace_id})
            only = next(iter(distributions.values()))
            return _Classification(only, 0.0, llm_risk, tuple(sorted(distributions)))
        return _Classification(fused.distribution, fused.disagreement, llm_risk, fused.models_used)

    # ------------------------------------------------------------------------- ruta de crisis
    async def _crisis_outcome(
        self,
        assessment: RiskAssessment,
        classification: _Classification,
        texto: str,
        command: AnalyzeCommand,
    ) -> _Outcome:
        chunks = await self._guard(
            self._retriever.institutional_info(),
            self._cfg.crisis_lookup_timeout_s,
            "info_institucional",
            command,
        )
        fragmentos = tuple(chunks or ())
        secondary = self._secondary_for(classification, texto)
        return _Outcome(
            respuesta=render_crisis_response([c.texto for c in fragmentos]),
            assessment=assessment,
            classification=classification,
            secondary=secondary,
            strategy=None,
            bandit_context=None,
            fuentes=self._sources(fragmentos),
            es_crisis=True,
            degradado=False,
        )

    # ------------------------------------------------------------------------- 5-8 respuesta
    async def _respond(
        self,
        assessment: RiskAssessment,
        classification: _Classification,
        texto: str,
        history: ConversationHistory,
        command: AnalyzeCommand,
    ) -> _Outcome:
        dist = classification.distribution
        degradado = len(classification.models_used) < 2
        secondary = self._secondary_for(classification, texto)

        def safe(strategy: Strategy | None = None, ctx: BanditContext | None = None) -> _Outcome:
            return _Outcome(
                SAFE_TEMPLATE, assessment, classification, secondary, strategy, ctx, (), False, True
            )

        if dist is None:  # ningún modelo respondió
            return safe()

        emotion = dist.top()
        bandit_context = BanditContext.build(
            dist, assessment.level, self._clock.now().hour, history.strategy_ids
        )
        strategy = await self._pick_strategy(bandit_context, emotion, command)  # 5

        chunks = await self._guard(  # 6
            self._retriever.retrieve(
                RetrievalQuery(
                    texto=texto,
                    emocion=emotion,
                    estrategia_id=strategy.id if strategy else None,
                    top_k=self._cfg.rag_top_k,
                    incluir_institucional=assessment.level >= RiskLevel.MEDIUM,
                )
            ),
            self._cfg.classification_timeout_s,
            "rag",
            command,
        )
        fragmentos = tuple(chunks or ())
        if not fragmentos:  # sin contexto recuperado no hay base para generar
            return safe(strategy, bandit_context)

        respuesta = await self._generate_validated(  # 7 y 8
            GenerationContext(
                texto_usuario=texto,
                emocion=emotion,
                emociones_secundarias=secondary,
                riesgo=assessment.level,
                estrategia=strategy,
                fragmentos=fragmentos,
                historial=history.messages,
            ),
            command,
        )
        if respuesta is None:
            return safe(strategy, bandit_context)
        return _Outcome(
            respuesta,
            assessment,
            classification,
            secondary,
            strategy,
            bandit_context,
            self._sources(fragmentos),
            False,
            degradado,
        )

    async def _pick_strategy(
        self, context: BanditContext, emotion: Emotion, command: AnalyzeCommand
    ) -> Strategy | None:
        candidates = await self._guard(
            self._catalog.for_emotion(emotion), self._cfg.persistence_timeout_s, "catalogo", command
        )
        if not candidates:
            candidates = await self._guard(
                self._catalog.list_all(), self._cfg.persistence_timeout_s, "catalogo", command
            )
        if not candidates:
            return None
        selection = await self._guard(
            self._selector.select(context, candidates),
            self._cfg.persistence_timeout_s,
            "bandido",
            command,
        )
        # Si el bandido falla, política por reglas: la primera candidata del catálogo.
        return selection.strategy if selection is not None else candidates[0]

    async def _generate_validated(self, context: GenerationContext, command: AnalyzeCommand) -> str | None:
        correcciones: tuple[str, ...] = ()
        for _ in range(self._cfg.max_regenerations + 1):
            texto = await self._guard(
                self._llm.generate(
                    GenerationContext(
                        texto_usuario=context.texto_usuario,
                        emocion=context.emocion,
                        emociones_secundarias=context.emociones_secundarias,
                        riesgo=context.riesgo,
                        estrategia=context.estrategia,
                        fragmentos=context.fragmentos,
                        historial=context.historial,
                        correcciones=correcciones,
                    )
                ),
                self._cfg.generation_timeout_s,
                "llm_generacion",
                command,
            )
            if texto is None or not texto.strip():
                return None
            verdict = self._validator.validate(texto, context.fragmentos, context.riesgo)
            if verdict.ok:
                return texto
            logger.info(
                "respuesta_rechazada",
                extra={"trace_id": command.trace_id, "razones": list(verdict.razones)},
            )
            correcciones = verdict.razones
        return None

    # ------------------------------------------------------------------------- 9-10
    async def _finish(
        self, command: AnalyzeCommand, texto: str, outcome: _Outcome, started: float
    ) -> AnalysisResult:
        cls = outcome.classification
        user_msg = Message(self._ids.new_id(), command.conversation_id, texto, Role.USER, self._clock.now())
        bot_msg = Message(
            self._ids.new_id(),
            command.conversation_id,
            outcome.respuesta,
            Role.ASSISTANT,
            self._clock.now(),
        )
        record = AnalysisRecord(
            trace_id=command.trace_id,
            conversation_id=command.conversation_id,
            user_ref=command.user_ref,
            user_message_id=user_msg.id,
            assistant_message_id=bot_msg.id,
            distribucion=cls.distribution,
            emociones_secundarias=outcome.secondary,
            desacuerdo_modelos=cls.disagreement,
            riesgo=outcome.assessment.level,
            riesgo_fuente=outcome.assessment.source,
            estrategia_id=outcome.strategy.id if outcome.strategy else None,
            bandit_context=outcome.bandit_context,
            es_crisis=outcome.es_crisis,
            degradado=outcome.degradado,
            fuentes=outcome.fuentes,
            timestamp=bot_msg.timestamp,
        )

        async def persist() -> bool:
            await self._repo.record_turn(user_msg, bot_msg, record)
            return True

        # Un fallo de persistencia queda en el log; no se le niega la respuesta al usuario.
        await self._guard(persist(), self._cfg.persistence_timeout_s, "persistencia", command)

        logger.info(
            "analisis_completado",
            extra={
                "trace_id": command.trace_id,
                "longitud_texto": len(texto),
                "riesgo": outcome.assessment.level.name,
                "riesgo_fuente": outcome.assessment.source,
                "es_crisis": outcome.es_crisis,
                "estrategia": record.estrategia_id,
                "modelos": list(cls.models_used),
                "degradado": outcome.degradado,
                "duracion_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return AnalysisResult(
            trace_id=command.trace_id,
            emocion_dominante=cls.distribution.top() if cls.distribution else None,
            distribucion=cls.distribution,
            emociones_secundarias=outcome.secondary,
            desacuerdo_modelos=cls.disagreement,
            riesgo=outcome.assessment.level,
            estrategia=(
                StrategyRef(outcome.strategy.id, outcome.strategy.nombre) if outcome.strategy else None
            ),
            respuesta=outcome.respuesta,
            es_crisis=outcome.es_crisis,
            fuentes=outcome.fuentes,
            degradado=outcome.degradado,
            modelos_usados=cls.models_used,
        )

    # ------------------------------------------------------------------------- utilidades
    def _secondary_for(self, cls: _Classification, texto: str) -> tuple[str, ...]:
        if cls.distribution is None:
            return ()
        return tuple(self._secondary.derive(cls.distribution, texto))

    @staticmethod
    def _sources(chunks: tuple[RetrievedChunk, ...]) -> tuple[SourceRef, ...]:
        seen: dict[str, SourceRef] = {}
        for c in chunks:
            seen.setdefault(c.doc_id, SourceRef(c.doc_id, c.titulo))
        return tuple(seen.values())

    @staticmethod
    async def _guard(awaitable: Awaitable[T], timeout: float, name: str, command: AnalyzeCommand) -> T | None:
        """Ejecuta con tiempo límite; ante cualquier fallo registra y devuelve None."""
        try:
            return await asyncio.wait_for(awaitable, timeout)
        except Exception as exc:
            logger.warning(
                "fallo_componente",
                extra={
                    "trace_id": command.trace_id,
                    "componente": name,
                    "error": type(exc).__name__,
                },
            )
            return None
