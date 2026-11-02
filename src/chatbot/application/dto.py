"""Objetos de transferencia entre adaptadores y casos de uso."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from chatbot.domain.models import (
    BanditContext,
    Emotion,
    EmotionDistribution,
    Message,
    RiskLevel,
    Strategy,
)

#: Nombres de los modelos en la fusión tardía (deben coincidir con los pesos configurados).
ENCODER_MODEL = "encoder"
LLM_MODEL = "llm"


# --------------------------------------------------------------------------- configuración
@dataclass(frozen=True, slots=True)
class AnalyzeConfig:
    max_text_length: int = 2000
    history_window: int = 5
    rag_top_k: int = 4
    classification_timeout_s: float = 6.0
    generation_timeout_s: float = 8.0
    persistence_timeout_s: float = 3.0
    crisis_lookup_timeout_s: float = 1.0
    max_regenerations: int = 1


# --------------------------------------------------------------------------- analizar
@dataclass(frozen=True, slots=True)
class AnalyzeCommand:
    conversation_id: str
    user_ref: str
    texto: str
    trace_id: str


@dataclass(frozen=True, slots=True)
class StrategyRef:
    id: str
    nombre: str


@dataclass(frozen=True, slots=True)
class SourceRef:
    doc_id: str
    titulo: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    trace_id: str
    #: None cuando no se pudo clasificar o cuando la ruta de crisis cortocircuitó el análisis.
    emocion_dominante: Emotion | None
    distribucion: EmotionDistribution | None
    emociones_secundarias: tuple[str, ...]
    desacuerdo_modelos: float
    riesgo: RiskLevel
    estrategia: StrategyRef | None
    respuesta: str
    es_crisis: bool
    fuentes: tuple[SourceRef, ...]
    #: True si faltó algún modelo o se usó la plantilla segura.
    degradado: bool = False
    modelos_usados: tuple[str, ...] = ()


# --------------------------------------------------------------------------- puertos: datos
@dataclass(frozen=True, slots=True)
class RiskDetection:
    level: RiskLevel
    #: Categorías detectadas (p. ej. "ideacion_suicida"); nunca el texto del usuario.
    categorias: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMClassification:
    distribucion: EmotionDistribution
    riesgo: RiskLevel
    justificacion_breve: str = ""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    doc_id: str
    titulo: str
    texto: str
    tipo: str
    fuente: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    texto: str
    emocion: Emotion | None
    estrategia_id: str | None
    top_k: int
    incluir_institucional: bool = False


@dataclass(frozen=True, slots=True)
class GenerationContext:
    texto_usuario: str
    emocion: Emotion | None
    emociones_secundarias: tuple[str, ...]
    riesgo: RiskLevel
    estrategia: Strategy | None
    fragmentos: tuple[RetrievedChunk, ...]
    historial: tuple[Message, ...] = ()
    #: Razones del validador en un reintento; vacío en el primer intento.
    correcciones: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    razones: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategySelection:
    strategy: Strategy
    exploracion: bool = False
    metodo: str = "reglas"


@dataclass(frozen=True, slots=True)
class ConversationHistory:
    messages: tuple[Message, ...] = ()
    risk_levels: tuple[RiskLevel, ...] = ()
    strategy_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    """Lo que se persiste de cada análisis (permite auditar y registrar el feedback)."""

    trace_id: str
    conversation_id: str
    user_ref: str
    user_message_id: str
    assistant_message_id: str
    distribucion: EmotionDistribution | None
    emociones_secundarias: tuple[str, ...]
    desacuerdo_modelos: float
    riesgo: RiskLevel
    riesgo_fuente: str
    estrategia_id: str | None
    bandit_context: BanditContext | None
    es_crisis: bool
    degradado: bool
    fuentes: tuple[SourceRef, ...]
    timestamp: datetime


# --------------------------------------------------------------------------- feedback
@dataclass(frozen=True, slots=True)
class FeedbackCommand:
    trace_id: str
    util: bool | int
    comentario: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    trace_id: str
    strategy_id: str
    reward: float


# --------------------------------------------------------------------------- ingesta RAG
class DocumentType(StrEnum):
    TECNICA = "tecnica"
    PSICOEDUCACION = "psicoeducacion"
    INSTITUCIONAL = "institucional"


@dataclass(frozen=True, slots=True)
class CatalogDocument:
    titulo: str
    texto: str
    tipo: DocumentType
    fuente: str
    version: str
    emociones_objetivo: frozenset[Emotion] = field(default_factory=frozenset)
    estrategia_id: str | None = None


@dataclass(frozen=True, slots=True)
class IngestCommand:
    documents: tuple[CatalogDocument, ...]


@dataclass(frozen=True, slots=True)
class IngestResult:
    ingested: tuple[str, ...]  # "fuente@version"
    skipped: tuple[str, ...]
    total_chunks: int
