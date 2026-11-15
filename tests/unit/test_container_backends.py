"""Fase 5: el proveedor se elige SOLO por configuración; `AnalyzeAndRespond` no cambia.

Las combinaciones reales se prueban con dobles inyectados (BD falsa, `MockTransport`, modelo falso).
"""

import itertools
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import chatbot.config.container as container_mod
from chatbot.adapters.inbound.api.main import create_app
from chatbot.adapters.outbound.classifier.encoder_hf import EncoderModelNotFound, HFEncoder
from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.llm.llm_openai_compatible import OpenAICompatibleLLM
from chatbot.adapters.outbound.memory.crisis import InMemoryCrisisEventRepository
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase
from chatbot.adapters.outbound.memory.seed import seed_documents
from chatbot.adapters.outbound.persistence.crisis_event_repo import CrisisEventRepository
from chatbot.adapters.outbound.rag.embedder import HashEmbedder, SentenceTransformerEmbedder
from chatbot.adapters.outbound.rag.pgvector_retriever import PgVectorRetriever
from chatbot.application.dto import AnalyzeCommand, IngestCommand
from chatbot.config.container import build_container
from chatbot.config.settings import AppEnv, Settings
from doubles.fake_sql import FakeDatabase

SECRETO = "tengo mucho miedo por el parcial secreto-xyz-777"
ID2LABEL = {0: "others", 1: "joy", 2: "sadness", 3: "anger", 4: "surprise", 5: "disgust", 6: "fear"}
PROBS = {"alegria": 0, "tristeza": 0.1, "miedo": 0.7, "ira": 0.05, "asco": 0, "sorpresa": 0, "neutral": 0.15}
GOOD_JSON = {"probabilidades": PROBS, "riesgo": "NONE", "justificacion_breve": "ok"}
TECNICA = {
    "fuente": "tec-respiracion-4-6",
    "titulo": "Respiración 4-6",
    "tipo": "tecnica",
    "distancia": 0.1,
    "texto": "Inhala por la nariz contando hasta 4 y exhala contando hasta 6.",
}
RESPUESTA = "Entiendo que estés nervioso. Prueba respirar con calma. ¿Te sirvió esta sugerencia?"


class FakeST:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, textos, **_kw):
        return [[0.01] * 384 for _ in textos]


def llm_handler(request: httpx.Request) -> httpx.Response:
    prompt = json.loads(request.content)["messages"][0]["content"]
    content = json.dumps(GOOD_JSON) if "Eres un clasificador" in prompt else RESPUESTA
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def real_doubles(monkeypatch):
    """Sustituye en el contenedor los adaptadores reales por versiones con dobles inyectados."""
    db = FakeDatabase({"FROM rag_chunks\nWHERE": [TECNICA]})

    def make_encoder(**kw):
        return HFEncoder(infer=lambda _t: [0.0, 0.2, 0.5, 0.0, 0.0, 0.0, 3.0], id2label=ID2LABEL, **kw)

    def make_llm(**kw):
        client = httpx.AsyncClient(transport=httpx.MockTransport(llm_handler))
        return OpenAICompatibleLLM(client=client, **kw)

    def make_embedder(name, dim, **kw):
        return SentenceTransformerEmbedder(name, dim, model_factory=lambda _n: FakeST(), **kw)

    monkeypatch.setattr(container_mod, "Database", lambda *a, **k: db)
    monkeypatch.setattr(container_mod, "HFEncoder", make_encoder)
    monkeypatch.setattr(container_mod, "OpenAICompatibleLLM", make_llm)
    monkeypatch.setattr(container_mod, "SentenceTransformerEmbedder", make_embedder)
    return db


def settings(**kw):
    return Settings(**{"_env_file": None, "app_env": AppEnv.TEST, **kw})


def config_for(encoder, llm, persistence, retriever):
    kw = {
        "encoder_backend": encoder,
        "llm_backend": llm,
        "persistence_backend": persistence,
        "retriever_backend": retriever,
    }
    if llm == "openai_compatible":
        kw |= {"llm_base_url": "https://llm.test/v1", "llm_model": "m", "llm_api_key": "sk-x"}
    if persistence == "postgres" or retriever == "pgvector":
        kw["database_url"] = "postgresql+asyncpg://u:p@h/db"
    return settings(**kw)


REPOS = [("memory", "memory"), ("postgres", "memory"), ("postgres", "pgvector")]
COMBINACIONES = [
    (enc, llm, pers, ret)
    for enc, llm, (pers, ret) in itertools.product(["stub", "hf"], ["fake", "openai_compatible"], REPOS)
]
REALES = (HFEncoder, OpenAICompatibleLLM, PgVectorRetriever, SentenceTransformerEmbedder)
FALSOS = (StubEncoder, FakeLLM, InMemoryKnowledgeBase, HashEmbedder)


@pytest.mark.parametrize("encoder, llm, persistence, retriever", COMBINACIONES)
async def test_cada_combinacion_de_backends_funciona_sin_cambiar_el_caso_de_uso(
    real_doubles, encoder, llm, persistence, retriever
):
    c = build_container(config_for(encoder, llm, persistence, retriever))
    await c.warmup()
    if retriever == "memory":
        await c.ingest_documents.execute(IngestCommand(tuple(seed_documents())))

    r = await c.analyze.execute(AnalyzeCommand("c1", "u-1", SECRETO, "tr-1"))

    assert r.es_crisis is False and r.estrategia is not None and r.fuentes
    assert r.respuesta.rstrip().endswith("?") and r.degradado is False
    assert set(r.modelos_usados) == {"encoder", "llm"}
    crisis = await c.analyze.execute(AnalyzeCommand("c1", "u-1", "ya no quiero vivir", "tr-2"))
    assert crisis.es_crisis is True
    await c.close()


@pytest.mark.parametrize(
    "encoder, llm, persistence, retriever, tipos",
    [
        ("stub", "fake", "memory", "memory", FALSOS),
        ("hf", "openai_compatible", "postgres", "pgvector", REALES),
    ],
)
def test_las_fabricas_eligen_el_adaptador_segun_la_configuracion(
    real_doubles, encoder, llm, persistence, retriever, tipos
):
    c = build_container(config_for(encoder, llm, persistence, retriever))
    t_enc, t_llm, _t_ret, t_emb = tipos
    assert isinstance(c.components["encoder"], t_enc) and isinstance(c.components["llm"], t_llm)
    assert isinstance(c.components["embedder"], t_emb)
    esperado = InMemoryCrisisEventRepository if persistence == "memory" else CrisisEventRepository
    assert isinstance(c.crisis_events, esperado)
    assert (c.database is None) == (persistence == "memory")


def test_ya_no_lanza_not_implemented_con_backends_reales(real_doubles):
    build_container(config_for("hf", "openai_compatible", "postgres", "pgvector"))


def test_el_catalogo_siempre_sale_del_yaml_incluso_con_memory():
    assert type(build_container(settings()).catalog).__name__ == "YamlStrategyCatalog"


def test_sin_catalogo_valido_no_arranca(tmp_path):
    with pytest.raises(Exception, match="No se encontró el catálogo"):
        build_container(settings(strategies_file=str(tmp_path / "no.yaml")))


def test_sin_prompts_validos_el_llm_real_no_arranca(real_doubles, tmp_path):
    cfg = settings(
        llm_backend="openai_compatible", llm_base_url="http://x", llm_model="m", prompts_dir=str(tmp_path)
    )
    with pytest.raises(Exception, match="No se encontró el prompt"):
        build_container(cfg)


# ------------------------------------------------------------------ privacidad
async def test_store_user_text_false_no_deja_el_texto_en_la_bd(real_doubles):
    cfg = settings(persistence_backend="postgres", database_url="postgresql://x", store_user_text=False)
    c = build_container(cfg)
    await c.ingest_documents.execute(IngestCommand(tuple(seed_documents())))
    await c.analyze.execute(AnalyzeCommand("c1", "u-1", SECRETO, "tr-1"))
    guardado = json.dumps([p for _, _, p in real_doubles.committed], default=str)
    assert SECRETO not in guardado and "[omitido]" in guardado


@pytest.mark.parametrize("persistence, retriever", [("memory", "memory"), ("postgres", "pgvector")])
async def test_ningun_log_del_flujo_completo_contiene_el_texto_del_usuario(
    real_doubles, caplog, persistence, retriever
):
    caplog.set_level(logging.DEBUG)
    c = build_container(config_for("hf", "openai_compatible", persistence, retriever))
    await c.warmup()
    if retriever == "memory":
        await c.ingest_documents.execute(IngestCommand(tuple(seed_documents())))
    await c.analyze.execute(AnalyzeCommand("c1", "u-1", SECRETO, "tr-1"))
    await c.analyze.execute(AnalyzeCommand("c1", "u-1", "ya no quiero vivir", "tr-2"))
    volcado = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert caplog.records and "secreto-xyz-777" not in volcado and "sk-x" not in volcado


# ------------------------------------------------------------------ /ready y lifespan
def test_ready_con_adaptadores_falsos_reporta_no_aplica_y_200():
    r = TestClient(create_app(settings())).get("/ready")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ready"
    for k in ("encoder", "embedder", "base_de_datos", "extension_vector"):
        assert body["checks"][k] == "no_aplica"
    assert body["checks"]["dominio"] == "ok" and body["checks"]["catalogo_estrategias"] == "ok"


def test_ready_real_es_503_hasta_que_el_warmup_carga_el_encoder(real_doubles):
    c = build_container(config_for("hf", "openai_compatible", "postgres", "pgvector"))
    app = create_app(container=c)
    sin_lifespan = TestClient(app).get("/ready")  # sin `with`: no corre el warmup
    assert sin_lifespan.status_code == 503
    assert sin_lifespan.json()["checks"]["encoder"] == "no_listo"
    assert sin_lifespan.json()["checks"]["embedder"] == "no_listo"
    with TestClient(app) as client:  # el lifespan hace el warmup
        r = client.get("/ready")
        checks = r.json()["checks"]
        assert r.status_code == 200 and r.json()["status"] == "ready"
        assert {checks[k] for k in ("encoder", "embedder", "base_de_datos", "extension_vector")} == {"ok"}


def test_ready_503_con_detalle_por_componente_si_la_bd_falla(real_doubles):
    c = build_container(config_for("stub", "fake", "postgres", "memory"))
    real_doubles.fail_on = "ping"
    r = TestClient(create_app(container=c)).get("/ready")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"
    assert r.json()["checks"]["base_de_datos"] == "error: ConnectionError"
    assert r.json()["checks"]["dominio"] == "ok"  # el resto del detalle sigue visible


def test_ready_503_si_falta_la_extension_vector(real_doubles):
    c = build_container(config_for("stub", "fake", "postgres", "pgvector"))
    real_doubles.vector = False
    r = TestClient(create_app(container=c)).get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["extension_vector"] == "error: LookupError"


def test_ready_no_filtra_el_mensaje_de_error_de_la_bd(real_doubles):
    c = build_container(config_for("stub", "fake", "postgres", "memory"))

    async def boom():
        raise ConnectionError("password=SUPERSECRETA host=interno")

    real_doubles.ping = boom
    assert "SUPERSECRETA" not in TestClient(create_app(container=c)).get("/ready").text


def test_lifespan_cierra_el_pool_de_la_bd_y_el_cliente_del_llm(real_doubles):
    c = build_container(config_for("hf", "openai_compatible", "postgres", "pgvector"))
    cierres = []
    original = c.components["llm"].aclose

    async def espia():
        cierres.append("llm")
        await original()

    c.components["llm"].aclose = espia
    with TestClient(create_app(container=c)):
        assert not real_doubles.closed and cierres == []
    assert real_doubles.closed and cierres == ["llm"]


def test_lifespan_failfast_si_el_warmup_falla_el_servicio_no_arranca(tmp_path):
    c = build_container(settings(encoder_backend="hf", encoder_model_path=str(tmp_path / "no-existe")))
    with pytest.raises(EncoderModelNotFound, match="No se encontró el modelo del encoder"):  # noqa: SIM117
        with TestClient(create_app(container=c)):
            pass


async def test_close_es_robusto_si_un_componente_falla(real_doubles):
    c = build_container(config_for("stub", "openai_compatible", "postgres", "memory"))

    async def boom():
        raise RuntimeError("x")

    c.components["llm"].aclose = boom
    await c.close()
    assert real_doubles.closed  # igual cerró la BD
