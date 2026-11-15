"""`GET /v1/strategies`: catálogo de estrategias de regulación (protegido con la llave de servicio)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from chatbot.adapters.inbound.api.dependencies import ContainerDep, require_service_key
from chatbot.adapters.inbound.api.schemas.strategies import StrategyListOut

router = APIRouter(prefix="/v1", tags=["estrategias"], dependencies=[Depends(require_service_key)])


@router.get("/strategies", response_model=StrategyListOut)
async def list_strategies(container: ContainerDep) -> StrategyListOut:
    return StrategyListOut.from_strategies(await container.list_strategies.execute())
