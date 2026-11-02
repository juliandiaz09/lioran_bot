"""Dependencias de FastAPI: contenedor y autenticación servicio a servicio."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from chatbot.config.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def _check_key(provided: str | None, expected: str) -> None:
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Llave de API ausente o inválida")


def require_service_key(container: ContainerDep, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """n8n -> backend: cabecera X-API-Key."""
    _check_key(x_api_key, container.settings.service_api_key.get_secret_value())


def require_admin_key(container: ContainerDep, x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Endpoints administrativos (ingesta del RAG)."""
    _check_key(x_api_key, container.settings.admin_api_key.get_secret_value())
