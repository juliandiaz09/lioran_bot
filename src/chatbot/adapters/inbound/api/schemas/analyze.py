"""Esquemas de `POST /v1/analyze`: entrada estricta y salida con la forma exacta de la especificación."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chatbot.application.dto import AnalysisResult
from chatbot.domain.models import Emotion, RiskLevel

#: Tope de entrada del texto: el doble de `MAX_MESSAGE_LENGTH` por defecto (2000). El caso de uso ya recorta
#: al largo configurado; este tope solo rechaza cuerpos desmesurados antes de llegar a él.
MAX_TEXT_INPUT_LENGTH = 4000

#: Longitud máxima de los identificadores que envía el cliente (n8n).
MAX_ID_LENGTH = 128

RiskName = Literal["NONE", "LOW", "MEDIUM", "HIGH"]

_RISK_NAMES: dict[RiskLevel, RiskName] = {
    RiskLevel.NONE: "NONE",
    RiskLevel.LOW: "LOW",
    RiskLevel.MEDIUM: "MEDIUM",
    RiskLevel.HIGH: "HIGH",
}

_CRISIS_NULL = "Es null en crisis (riesgo HIGH: se responde con la plantilla sin clasificar)"


class AnalyzeRequest(BaseModel):
    """Mensaje del usuario. Estricto: sin campos extra ni coerción de tipos."""

    model_config = ConfigDict(extra="forbid", strict=True)

    conversation_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    user_ref: str = Field(
        min_length=1,
        max_length=MAX_ID_LENGTH,
        description="Seudónimo estable del usuario; nunca su identidad.",
    )
    texto: str = Field(min_length=1, max_length=MAX_TEXT_INPUT_LENGTH)


class StrategyRefOut(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    nombre: str


class SourceOut(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    doc_id: str
    titulo: str


class AnalyzeResponse(BaseModel):
    """Resultado del análisis tal como lo recibe el cliente."""

    model_config = ConfigDict(extra="forbid", strict=True)

    trace_id: str
    emocion_dominante: Emotion | None = Field(
        description=f"Emoción más probable. {_CRISIS_NULL} o si ningún modelo pudo clasificar."
    )
    distribucion: dict[str, float] | None = Field(
        description=f"Probabilidad por emoción (suma 1). {_CRISIS_NULL} o si ningún modelo pudo clasificar."
    )
    emociones_secundarias: list[str]
    desacuerdo_modelos: float = Field(description="Divergencia de Jensen-Shannon entre modelos, en [0, 1].")
    riesgo: RiskName
    estrategia: StrategyRefOut | None = Field(
        description=f"Estrategia de regulación elegida. {_CRISIS_NULL}."
    )
    respuesta: str
    es_crisis: bool = Field(description="True si el caso queda marcado para seguimiento humano.")
    fuentes: list[SourceOut]
    degradado: bool = Field(
        default=False, description="True si faltó algún modelo o se usó la plantilla segura."
    )
    modelos_usados: list[str] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: AnalysisResult) -> AnalyzeResponse:
        """Traduce el resultado del caso de uso al JSON público."""
        estrategia = result.estrategia
        return cls(
            trace_id=result.trace_id,
            emocion_dominante=result.emocion_dominante,
            distribucion=result.distribucion.to_dict() if result.distribucion is not None else None,
            emociones_secundarias=list(result.emociones_secundarias),
            desacuerdo_modelos=result.desacuerdo_modelos,
            riesgo=_RISK_NAMES[result.riesgo],
            estrategia=StrategyRefOut(id=estrategia.id, nombre=estrategia.nombre) if estrategia else None,
            respuesta=result.respuesta,
            es_crisis=result.es_crisis,
            fuentes=[SourceOut(doc_id=f.doc_id, titulo=f.titulo) for f in result.fuentes],
            degradado=result.degradado,
            modelos_usados=list(result.modelos_usados),
        )
