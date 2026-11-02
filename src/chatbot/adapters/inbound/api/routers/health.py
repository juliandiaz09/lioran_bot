from __future__ import annotations

from fastapi import APIRouter

from chatbot.adapters.inbound.api.dependencies import ContainerDep

router = APIRouter(tags=["utilidades"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: el proceso está vivo."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(container: ContainerDep) -> dict[str, object]:
    """Readiness: dependencias listas. (BD y modelos se verifican desde la Fase 5.)"""
    checks = container.readiness()
    return {"status": "ready", "checks": checks}
