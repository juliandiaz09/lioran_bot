import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from chatbot.adapters.inbound.api.dependencies import require_admin_key, require_service_key
from chatbot.adapters.inbound.api.main import create_app
from chatbot.config.settings import AppEnv, Settings
from chatbot.domain.errors import DomainError

TRACE = "123e4567-e89b-42d3-a456-426614174000"


@pytest.fixture
def app() -> FastAPI:
    settings = Settings(
        _env_file=None,
        app_env=AppEnv.TEST,
        service_api_key=SecretStr("llave-n8n"),
        admin_api_key=SecretStr("llave-admin"),
    )
    app = create_app(settings)

    class Body(BaseModel):
        n: int

    @app.post("/_t/echo", dependencies=[Depends(require_service_key)])
    async def echo(b: Body) -> dict[str, int]:
        return {"n": b.n}

    @app.get("/_t/admin", dependencies=[Depends(require_admin_key)])
    async def admin() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/_t/domain")
    async def domain() -> None:
        raise DomainError("algo del dominio")

    @app.get("/_t/boom")
    async def boom() -> None:
        raise RuntimeError("secreto interno")

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["checks"]["dominio"] == "ok"


def test_siempre_devuelve_trace_id(client):
    assert client.get("/health").headers["X-Trace-Id"]


def test_respeta_trace_id_valido_y_reemplaza_invalido(client):
    assert client.get("/health", headers={"X-Trace-Id": TRACE}).headers["X-Trace-Id"] == TRACE
    nuevo = client.get("/health", headers={"X-Trace-Id": "no-es-uuid"}).headers["X-Trace-Id"]
    assert nuevo != "no-es-uuid" and len(nuevo) == 36


def test_sin_llave_401_con_formato_uniforme(client):
    r = client.post("/_t/echo", json={"n": 1})
    assert r.status_code == 401
    assert r.json() == {"error": {"codigo": "no_autorizado", "mensaje": "Llave de API ausente o inválida"}}


def test_llave_incorrecta_401_y_correcta_200(client):
    assert client.post("/_t/echo", json={"n": 1}, headers={"X-API-Key": "mala"}).status_code == 401
    ok = client.post("/_t/echo", json={"n": 1}, headers={"X-API-Key": "llave-n8n"})
    assert ok.status_code == 200 and ok.json() == {"n": 1}


def test_llave_admin_distinta_de_la_de_servicio(client):
    assert client.get("/_t/admin", headers={"X-API-Key": "llave-n8n"}).status_code == 401
    assert client.get("/_t/admin", headers={"X-API-Key": "llave-admin"}).status_code == 200


def test_llave_vacia_en_config_nunca_autoriza():
    s = Settings(_env_file=None, app_env=AppEnv.TEST)
    c = TestClient(create_app(s), raise_server_exceptions=False)
    assert c.get("/ready", headers={"X-API-Key": ""}).status_code == 200  # ruta pública
    app = create_app(s)

    @app.get("/_t/x", dependencies=[Depends(require_service_key)])
    async def x() -> None: ...

    assert TestClient(app).get("/_t/x", headers={"X-API-Key": ""}).status_code == 401


def test_validacion_de_entrada_formato_uniforme(client):
    r = client.post("/_t/echo", json={"n": "abc"}, headers={"X-API-Key": "llave-n8n"})
    assert r.status_code == 422
    assert r.json()["error"]["codigo"] == "validacion_fallida"
    assert "n" in r.json()["error"]["mensaje"]


def test_404_y_405_uniformes(client):
    assert client.get("/no-existe").json()["error"]["codigo"] == "no_encontrado"
    assert client.post("/health").json()["error"]["codigo"] == "metodo_no_permitido"


def test_error_de_dominio_422(client):
    r = client.get("/_t/domain")
    assert r.status_code == 422 and r.json()["error"] == {
        "codigo": "error_dominio",
        "mensaje": "algo del dominio",
    }


def test_error_inesperado_500_sin_filtrar_detalles(client):
    r = client.get("/_t/boom")
    assert r.status_code == 500
    assert r.json()["error"]["codigo"] == "error_interno"
    assert "secreto" not in r.text
    assert r.headers["X-Trace-Id"]
