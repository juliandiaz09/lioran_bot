"""`POST /v1/analyze`: lo llama n8n con el mensaje del estudiante. Solo conoce el puerto de entrada."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from chatbot.adapters.inbound.api.dependencies import ContainerDep, require_service_key
from chatbot.adapters.inbound.api.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from chatbot.application.dto import AnalyzeCommand

router = APIRouter(prefix="/v1", tags=["análisis"], dependencies=[Depends(require_service_key)])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, request: Request, container: ContainerDep) -> AnalyzeResponse:
    """Analiza el mensaje y devuelve la respuesta de apoyo. Con riesgo alto responde con la plantilla de
    crisis (`es_crisis: true`) sin invocar al LLM; en ese caso `emocion_dominante`, `distribucion` y
    `estrategia` pueden ser `null`."""
    command = AnalyzeCommand(
        conversation_id=body.conversation_id,
        user_ref=body.user_ref,
        texto=body.texto,
        trace_id=request.state.trace_id,
    )
    result = await container.analyze.execute(command)
    return AnalyzeResponse.from_result(result)
