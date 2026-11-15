"""Errores uniformes: {"error": {"codigo": "...", "mensaje": "..."}}."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from chatbot.application.errors import (
    AnalysisNotFound,
    DuplicateFeedback,
    EmptyMessage,
    FeedbackNotApplicable,
    IngestError,
)
from chatbot.domain.errors import DomainError
from chatbot.shared.ids import new_trace_id

logger = logging.getLogger(__name__)

_CODES = {
    400: "solicitud_invalida",
    401: "no_autorizado",
    403: "prohibido",
    404: "no_encontrado",
    405: "metodo_no_permitido",
    422: "validacion_fallida",
}


#: Estado HTTP por clase de error (capa de adaptador: el dominio no conoce HTTP). El resto, 422.
_DOMAIN_STATUS: dict[type[DomainError], int] = {
    EmptyMessage: 422,
    AnalysisNotFound: 404,
    DuplicateFeedback: 409,
    FeedbackNotApplicable: 409,
    IngestError: 422,
}


def domain_status(exc: DomainError) -> int:
    for clase in type(exc).__mro__:
        if clase in _DOMAIN_STATUS:
            return _DOMAIN_STATUS[clase]
    return 422


def error_body(codigo: str, mensaje: str) -> dict[str, dict[str, str]]:
    return {"error": {"codigo": codigo, "mensaje": mensaje}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codigo = _CODES.get(exc.status_code, "error_http")
        return JSONResponse(
            error_body(codigo, str(exc.detail)), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        campos = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(error_body("validacion_fallida", campos), status_code=422)

    @app.exception_handler(DomainError)
    async def _domain(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(error_body(exc.codigo, exc.mensaje), status_code=domain_status(exc))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        logger.exception("error_no_controlado", extra={"trace_id": trace_id})
        return JSONResponse(
            error_body("error_interno", "Ocurrió un error interno. Intenta de nuevo."),
            status_code=500,
            headers={"X-Trace-Id": trace_id},
        )
