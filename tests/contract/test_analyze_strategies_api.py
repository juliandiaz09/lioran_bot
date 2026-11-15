"""Contrato HTTP de `POST /v1/analyze` y `GET /v1/strategies` (contenedor con dobles inyectado)."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from chatbot.adapters.inbound.api.error_handlers import domain_status
from chatbot.adapters.inbound.api.main import create_app
from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.risk.rules_risk_detector import RulesRiskDetector
from chatbot.application.errors import (
    AnalysisNotFound,
    DuplicateFeedback,
    EmptyMessage,
    FeedbackNotApplicable,
    IngestError,
)
from chatbot.config.container import build_container
from chatbot.config.settings import AppEnv, Settings
from chatbot.domain.errors import DomainError, InvalidStrategy
from helpers import RAIZ, Harness, make_harness

TRACE = "123e4567-e89b-42d3-a456-426614174000"
LLAVE = {"X-API-Key": "llave-n8n"}
CUERPO = {"conversation_id": "c-1", "user_ref": "u-1", "texto": "Tengo mucho miedo por el parcial de mañana"}
CRISIS = {"conversation_id": "c-1", "user_ref": "u-1", "texto": "kiero morirme xq ya no aguanto"}
CLAVES_NORMAL = {
    "trace_id", "emocion_dominante", "distribucion", "emociones_secundarias", "desacuerdo_modelos",
    "riesgo", "estrategia", "respuesta", "es_crisis", "fuentes", "degradado", "modelos_usados",
}  # fmt: skip


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env=AppEnv.TEST,
        service_api_key=SecretStr("llave-n8n"),
        admin_api_key=SecretStr("llave-admin"),
    )


def _cliente(h: Harness) -> TestClient:
    base = build_container(_settings())
    contenedor = dataclasses.replace(base, analyze=h.use_case, crisis_events=h.crisis_events)
    return TestClient(create_app(container=contenedor), raise_server_exceptions=False)


@pytest.fixture
def h() -> Harness:
    return asyncio.run(
        make_harness(risk_detector=RulesRiskDetector.from_file(RAIZ / "config" / "risk_lexicon.yaml"))
    )


@pytest.fixture
def client(h: Harness) -> TestClient:
    return _cliente(h)


# ----------------------------------------------------------------------------- autenticación
def test_analyze_sin_llave_401(client):
    r = client.post("/v1/analyze", json=CUERPO)
    assert r.status_code == 401 and r.json()["error"]["codigo"] == "no_autorizado"


def test_analyze_llave_erronea_401(client):
    assert client.post("/v1/analyze", json=CUERPO, headers={"X-API-Key": "otra"}).status_code == 401


def test_analyze_con_llave_de_administrador_tampoco_sirve(client):
    assert client.post("/v1/analyze", json=CUERPO, headers={"X-API-Key": "llave-admin"}).status_code == 401


# ----------------------------------------------------------------------------- validación (422)
@pytest.mark.parametrize(
    "cuerpo",
    [
        {},
        {"conversation_id": "c", "user_ref": "u"},
        {"conversation_id": "c", "texto": "hola"},
        {**CUERPO, "texto": ""},
        {**CUERPO, "conversation_id": ""},
        {**CUERPO, "user_ref": "x" * 129},
        {**CUERPO, "texto": "x" * 4001},
        {**CUERPO, "texto": 123},
        {**CUERPO, "extra": "no permitido"},
        {**CUERPO, "chat_id": 987654},
    ],
)
def test_analyze_422_con_cuerpo_invalido(client, cuerpo):
    r = client.post("/v1/analyze", json=cuerpo, headers=LLAVE)
    assert r.status_code == 422 and r.json()["error"]["codigo"] == "validacion_fallida"


def test_analyze_422_con_json_roto(client):
    r = client.post(
        "/v1/analyze", content=b"{no es json", headers={**LLAVE, "Content-Type": "application/json"}
    )
    assert r.status_code == 422 and set(r.json()) == {"error"}


def test_analyze_texto_solo_espacios_da_error_uniforme_422(client):
    r = client.post("/v1/analyze", json={**CUERPO, "texto": "   "}, headers=LLAVE)
    assert r.status_code == 422 and r.json()["error"]["codigo"] == "mensaje_vacio"


# ----------------------------------------------------------------------------- forma de la respuesta
def test_analyze_respuesta_normal_tiene_la_forma_de_la_especificacion(client):
    r = client.post("/v1/analyze", json=CUERPO, headers=LLAVE)
    assert r.status_code == 200
    d: dict[str, Any] = r.json()
    assert set(d) == CLAVES_NORMAL
    assert d["riesgo"] in {"NONE", "LOW", "MEDIUM", "HIGH"} and d["es_crisis"] is False
    assert d["emocion_dominante"] == "miedo" and abs(sum(d["distribucion"].values()) - 1) < 1e-3
    assert set(d["estrategia"]) == {"id", "nombre"}
    assert all(set(f) == {"doc_id", "titulo"} for f in d["fuentes"]) and d["fuentes"]
    assert isinstance(d["emociones_secundarias"], list) and 0 <= d["desacuerdo_modelos"] <= 1
    assert d["respuesta"] and d["trace_id"] == r.headers["X-Trace-Id"]


def test_analyze_respuesta_de_crisis_tiene_nulos_y_es_crisis(client, h):
    r = client.post("/v1/analyze", json=CRISIS, headers=LLAVE)
    d = r.json()
    assert r.status_code == 200 and set(d) == CLAVES_NORMAL
    assert d["es_crisis"] is True and d["riesgo"] == "HIGH"
    assert d["emocion_dominante"] is None and d["distribucion"] is None and d["estrategia"] is None
    assert "Bienestar Universitario" in d["respuesta"]
    assert d["fuentes"] and len(h.crisis_events.events) == 1


def test_trace_id_de_entrada_se_respeta_y_se_devuelve(client):
    r = client.post("/v1/analyze", json=CUERPO, headers={**LLAVE, "X-Trace-Id": TRACE})
    assert r.headers["X-Trace-Id"] == TRACE and r.json()["trace_id"] == TRACE


def test_si_no_hay_trace_id_el_backend_genera_uno(client):
    r = client.post("/v1/analyze", json=CUERPO, headers=LLAVE)
    assert r.headers["X-Trace-Id"] and r.json()["trace_id"] == r.headers["X-Trace-Id"]


# ----------------------------------------------------------------------------- ruta de crisis
def test_crisis_por_http_no_invoca_ni_llm_ni_encoder(client, h):
    r = client.post("/v1/analyze", json=CRISIS, headers=LLAVE)
    assert r.json()["es_crisis"] is True
    assert h.llm.generate_calls == [] and h.llm.classify_calls == [] and h.encoder.calls == []


def test_crisis_responde_en_menos_de_2s_aun_con_llm_y_encoder_lentos():
    lento = asyncio.run(
        make_harness(
            encoder=StubEncoder(delay_s=5.0),
            llm=FakeLLM(delay_s=5.0),
            risk_detector=RulesRiskDetector.from_file(RAIZ / "config" / "risk_lexicon.yaml"),
        )
    )
    t0 = time.perf_counter()
    r = _cliente(lento).post("/v1/analyze", json=CRISIS, headers=LLAVE)
    assert r.json()["es_crisis"] is True and time.perf_counter() - t0 <= 2.0


def test_crisis_sin_catalogo_institucional_no_inventa_digitos():
    sin = asyncio.run(
        make_harness(
            seed_kb=False, risk_detector=RulesRiskDetector.from_file(RAIZ / "config" / "risk_lexicon.yaml")
        )
    )
    r = _cliente(sin).post("/v1/analyze", json=CRISIS, headers=LLAVE)
    d = r.json()
    assert d["es_crisis"] is True and not any(ch.isdigit() for ch in d["respuesta"]) and d["fuentes"] == []


# ----------------------------------------------------------------------------- resiliencia
def test_ningun_500_cuando_fallan_encoder_y_llm():
    caido = asyncio.run(
        make_harness(encoder=StubEncoder(fail=True), llm=FakeLLM(fail_classify=True, fail_generate=True))
    )
    r = _cliente(caido).post("/v1/analyze", json=CUERPO, headers=LLAVE)
    assert r.status_code == 200 and r.json()["degradado"] is True and r.json()["respuesta"]


def test_ningun_500_cuando_falla_solo_el_llm():
    h2 = asyncio.run(make_harness(llm=FakeLLM(fail_classify=True, fail_generate=True)))
    r = _cliente(h2).post("/v1/analyze", json=CUERPO, headers=LLAVE)
    assert r.status_code == 200 and r.json()["degradado"] is True


def test_error_inesperado_responde_formato_uniforme_con_trace_id(h):
    async def boom(_: object) -> None:
        raise RuntimeError("fallo imprevisto")

    h.use_case.execute = boom  # type: ignore[method-assign]
    r = _cliente(h).post("/v1/analyze", json=CUERPO, headers={**LLAVE, "X-Trace-Id": TRACE})
    assert (
        r.status_code == 500
        and r.json()["error"]["codigo"] == "error_interno"
        and r.headers["X-Trace-Id"] == TRACE
    )


# ----------------------------------------------------------------------------- privacidad
def test_los_logs_de_los_routers_y_del_detector_no_contienen_el_texto(client, caplog):
    secreto = "kiero morirme xq ya no aguanto mas la tesis de mi hermana Lucía"
    with caplog.at_level(logging.DEBUG):
        client.post("/v1/analyze", json={**CRISIS, "texto": secreto}, headers=LLAVE)
        client.post("/v1/analyze", json={**CUERPO, "texto": secreto + " ok"}, headers=LLAVE)
    assert caplog.records
    volcado = " ".join(str(r.__dict__) + r.getMessage() for r in caplog.records)
    assert "morirme" not in volcado and "Lucía" not in volcado and "tesis" not in volcado


# ----------------------------------------------------------------------------- /v1/strategies
def test_strategies_requiere_llave(client):
    assert client.get("/v1/strategies").status_code == 401
    assert client.get("/v1/strategies", headers={"X-API-Key": "llave-admin"}).status_code == 401


def test_strategies_forma_correcta(client):
    r = client.get("/v1/strategies", headers=LLAVE)
    assert r.status_code == 200
    estrategias = r.json()["estrategias"]
    assert estrategias and all(
        set(e) == {"id", "nombre", "tipo", "emociones_objetivo", "descripcion"} for e in estrategias
    )
    assert all(isinstance(e["emociones_objetivo"], list) for e in estrategias)


# ----------------------------------------------------------------------------- mapeo de estados
@pytest.mark.parametrize(
    ("error", "estado"),
    [
        (EmptyMessage("x"), 422),
        (AnalysisNotFound("x"), 404),
        (DuplicateFeedback("x"), 409),
        (FeedbackNotApplicable("x"), 409),
        (IngestError("x"), 422),
        (InvalidStrategy("x"), 422),
        (DomainError("x"), 422),
    ],
)
def test_estado_http_por_clase_de_error(error, estado):
    assert domain_status(error) == estado


def test_create_app_acepta_un_contenedor_inyectado(h):
    base = build_container(_settings())
    app = create_app(container=dataclasses.replace(base, analyze=h.use_case))
    assert app.state.container.analyze is h.use_case
