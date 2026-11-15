from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from chatbot.adapters.inbound.api.dependencies import ContainerDep

router = APIRouter(tags=["utilidades"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: el proceso está vivo."""
    return {"status": "ok"}


@router.get("/ready", response_model=None)
async def ready(container: ContainerDep) -> JSONResponse:
    """Readiness real: 200 si todo está bien; 503 con el detalle por componente si algo falla."""
    ok, checks = await container.check_readiness()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )
