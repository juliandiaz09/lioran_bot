"""Punto de entrada FastAPI. `create_app` permite inyectar settings en las pruebas."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from chatbot.adapters.inbound.api.error_handlers import register_error_handlers
from chatbot.adapters.inbound.api.routers import health
from chatbot.config.container import build_container
from chatbot.config.settings import Settings, get_settings
from chatbot.shared.ids import parse_trace_id
from chatbot.shared.logging import configure_logging

TRACE_HEADER = "X-Trace-Id"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="Chatbot de regulación emocional - Backend", version="0.1.0")
    app.state.container = build_container(settings)

    @app.middleware("http")
    async def trace_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = parse_trace_id(request.headers.get(TRACE_HEADER))
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers[TRACE_HEADER] = trace_id
        return response

    register_error_handlers(app)
    app.include_router(health.router)
    return app


def app_factory() -> FastAPI:
    """Para `uvicorn chatbot.adapters.inbound.api.main:app_factory --factory`."""
    return create_app()
