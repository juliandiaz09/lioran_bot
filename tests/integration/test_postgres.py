"""Pruebas contra Postgres + pgvector reales (repos, retriever, migrador, contenedor de punta a punta)."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import chatbot.config.container as container_mod
from chatbot.adapters.inbound.api.main import create_app
from chatbot.adapters.outbound.memory.seed import seed_documents
from chatbot.adapters.outbound.persistence.crisis_event_repo import CrisisEventRepository
from chatbot.adapters.outbound.persistence.supabase_conversation_repo import SupabaseConversationRepository
from chatbot.adapters.outbound.persistence.supabase_feedback_repo import SupabaseFeedbackRepository
from chatbot.adapters.outbound.rag.pgvector_retriever import PgVectorRetriever
from chatbot.application.dto import (
    AnalysisRecord,
    AnalyzeCommand,
    CatalogDocument,
    CrisisEvent,
    DocumentType,
    FeedbackCommand,
    IngestCommand,
    RetrievalQuery,
    SourceRef,
)
from chatbot.config.container import build_container
from chatbot.config.settings import AppEnv, Settings
from chatbot.domain.models import (
    BanditContext,
    Certainty,
    Emotion,
    EmotionDistribution,
    Feedback,
    Message,
    RiskLevel,
    Role,
    TimeOfDay,
)
from chatbot.scripts.migrate import Migrator
from integration.conftest import ToyEmbedder

pytestmark = pytest.mark.integration
E = Emotion
NOW = datetime(2026, 11, 15, 10, 0, tzinfo=UTC)
SECRETO = "texto-privado-integracion-555"


def turn(n=1, texto=SECRETO, user_message_id=None, riesgo=RiskLevel.LOW, estrategia="resp", conv="c1"):
    user = Message(f"mu{n}", conv, texto, Role.USER, NOW)
    bot = Message(f"mb{n}", conv, "Respuesta ¿Te sirvió?", Role.ASSISTANT, NOW)
    ctx = BanditContext(E.MIEDO, Certainty.ALTA, riesgo, TimeOfDay.MANANA, ("a",))
    rec = AnalysisRecord(
        f"t{n}", conv, "user-ref-1", user_message_id or user.id, bot.id,
        EmotionDistribution({E.MIEDO: 0.75, E.NEUTRAL: 0.25}), ("ansiedad",), 0.2, riesgo, "reglas",
        estrategia, ctx, False, False, (SourceRef("d1", "Doc 1"),), NOW,
    )  # fmt: skip
    return user, bot, rec


async def count(db, table, where="true", params=None):
    async with db.transaction() as tx:
        row = await tx.fetch_one(f"SELECT count(*) AS n FROM {table} WHERE {where}", params)
    return row["n"]


# ------------------------------------------------------------------ BD / esquema
async def test_ping_y_extension_vector(db):
    await db.ping()
    assert await db.has_vector_extension() is True


async def test_el_esquema_tiene_rls_en_todas_las_tablas_y_vector_384(db):
    async with db.transaction() as tx:
        rows = await tx.fetch_all(
            "SELECT relname, relrowsecurity FROM pg_class "
            "WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace"
        )
        dim = await tx.fetch_one(
            "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
            "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
        )
    assert rows and all(r["relrowsecurity"] for r in rows)
    assert dim["t"] == "vector(384)"


# ------------------------------------------------------------------ repos
async def test_record_turn_history_y_get_analysis_ida_y_vuelta(db):
    repo = SupabaseConversationRepository(db)
    user, bot, rec = turn(1)
    await repo.record_turn(user, bot, rec)
    await repo.record_turn(*turn(2, riesgo=RiskLevel.MEDIUM, estrategia="grounding"))

    h = await repo.history("c1", 5)
    assert [m.id for m in h.messages] == [
        "mu1",
        "mb1",
        "mu2",
        "mb2",
    ]  # orden estable aunque el timestamp coincida
    assert h.messages[0].rol is Role.USER and h.messages[0].texto == SECRETO
    assert h.risk_levels == (RiskLevel.LOW, RiskLevel.MEDIUM) and h.strategy_ids == ("resp", "grounding")
    assert (await repo.get_analysis("t1")) == rec
    assert await repo.get_analysis("no-existe") is None


async def test_history_respeta_el_limite_y_el_aislamiento_entre_conversaciones(db):
    repo = SupabaseConversationRepository(db)
    for n in (1, 2, 3):
        await repo.record_turn(*turn(n))
    await repo.record_turn(*turn(9, conv="otra"))
    h = await repo.history("c1", 2)
    assert [m.id for m in h.messages] == ["mu2", "mb2", "mu3", "mb3"] and len(h.risk_levels) == 2
    assert (await repo.history("nadie", 5)).messages == ()


async def test_record_turn_es_atomico_si_falla_a_mitad_no_queda_nada(db):
    repo = SupabaseConversationRepository(db)
    user, bot, rec = turn(1, user_message_id="mensaje-inexistente")  # el análisis viola la FK al final
    with pytest.raises(Exception, match="(?i)foreign key|violates"):
        await repo.record_turn(user, bot, rec)
    assert [await count(db, t) for t in ("conversations", "messages", "analyses")] == [0, 0, 0]


async def test_store_user_text_false_guarda_omitido_y_la_longitud_real(db):
    await SupabaseConversationRepository(db, store_user_text=False).record_turn(*turn(1))
    async with db.transaction() as tx:
        rows = await tx.fetch_all("SELECT rol, texto, longitud FROM messages ORDER BY seq")
    assert rows[0] == {"rol": "user", "texto": "[omitido]", "longitud": len(SECRETO)}
    assert rows[1]["rol"] == "assistant" and rows[1]["texto"] == "Respuesta ¿Te sirvió?"
    assert await count(db, "messages", "texto LIKE :p", {"p": f"%{SECRETO}%"}) == 0


async def test_feedback_un_solo_registro_por_trace_id(db):
    await SupabaseConversationRepository(db).record_turn(*turn(1))
    repo = SupabaseFeedbackRepository(db)
    assert not await repo.exists("t1")
    await repo.save("t1", Feedback("mb1", "resp", True, "gracias"))
    assert await repo.exists("t1")
    with pytest.raises(Exception, match="(?i)duplicate|unique|violates"):
        await repo.save("t1", Feedback("mb1", "resp", False))
    async with db.transaction() as tx:
        row = await tx.fetch_one("SELECT util_bool, util_escala, comentario, reward FROM feedback")
    assert row == {"util_bool": True, "util_escala": None, "comentario": "gracias", "reward": 1.0}


async def test_feedback_con_escala_y_sin_texto(db):
    await SupabaseConversationRepository(db).record_turn(*turn(1))
    await SupabaseFeedbackRepository(db, store_user_text=False).save(
        "t1", Feedback("mb1", "resp", 5, SECRETO)
    )
    async with db.transaction() as tx:
        row = await tx.fetch_one("SELECT util_bool, util_escala, comentario, reward FROM feedback")
    assert row == {"util_bool": None, "util_escala": 5, "comentario": None, "reward": 1.0}


async def test_crisis_events_inserta_sin_texto_y_es_idempotente(db, caplog):
    caplog.set_level(logging.WARNING)
    repo = CrisisEventRepository(db)
    ev = CrisisEvent("t9", "c1", "user-ref-1", ("ideacion_suicida",), "reglas", NOW)
    await repo.register(ev)
    await repo.register(ev)  # mismo trace_id: no duplica ni falla
    async with db.transaction() as tx:
        rows = await tx.fetch_all("SELECT * FROM crisis_events")
    assert len(rows) == 1 and rows[0]["categorias"] == ["ideacion_suicida"] and rows[0]["atendido"] is False
    assert not any(c in rows[0] for c in ("texto", "mensaje"))
    assert any(r.getMessage() == "evento_crisis" for r in caplog.records)


# ------------------------------------------------------------------ retriever pgvector
def doc(fuente, titulo, texto, tipo, emociones=(), estrategia=None, version="v1"):
    return CatalogDocument(titulo, texto, tipo, fuente, version, frozenset(emociones), estrategia)


async def poblar(db):
    r = PgVectorRetriever(db, ToyEmbedder())
    tec, psi, inst = DocumentType.TECNICA, DocumentType.PSICOEDUCACION, DocumentType.INSTITUCIONAL
    docs = [
        doc("tec-resp", "Respiración", "respira lento", tec, {E.MIEDO, E.IRA}, "respiracion"),
        doc("tec-ancla", "Anclaje", "ancla tus sentidos", tec, {E.MIEDO}, "grounding"),
        doc("tec-reen", "Reencuadre", "piensa otra vez", tec, {E.TRISTEZA}, "reencuadre"),
        doc("psi-miedo", "Qué es el miedo", "respira: el miedo es normal", psi, {E.MIEDO}),
        doc("psi-tris", "Qué es la tristeza", "la tristeza pasa", psi, {E.TRISTEZA}),
        doc("inst-bienestar", "Bienestar", "contacto de Bienestar", inst),
    ]
    for d in docs:
        emb = await ToyEmbedder().embed([d.texto])
        await r.save(d, [d.texto], emb)
    return r


async def test_retrieve_filtra_por_estrategia_emocion_y_tipo(db):
    r = await poblar(db)
    out = await r.retrieve(RetrievalQuery("respira", E.MIEDO, "respiracion", 10))
    assert {c.doc_id for c in out} == {
        "tec-resp",
        "psi-miedo",
    }  # técnica SOLO de su estrategia + psicoeducación del miedo
    assert all(c.tipo != "institucional" for c in out)
    out = await r.retrieve(RetrievalQuery("respira", E.MIEDO, None, 10))
    assert {c.doc_id for c in out} == {
        "tec-resp",
        "tec-ancla",
        "psi-miedo",
    }  # sin estrategia: técnicas por emoción
    assert {c.doc_id for c in await r.retrieve(RetrievalQuery("x", E.TRISTEZA, None, 10))} == {
        "tec-reen",
        "psi-tris",
    }
    assert await r.retrieve(RetrievalQuery("x", E.ALEGRIA, None, 10)) == []


async def test_retrieve_ordena_por_similitud_coseno_con_score_y_top_k(db):
    r = await poblar(db)
    out = await r.retrieve(RetrievalQuery("respira", E.MIEDO, None, 10))
    assert out[0].score > out[-1].score
    assert {out[0].doc_id, out[1].doc_id} == {"tec-resp", "psi-miedo"} and round(out[0].score, 3) == 1.0
    assert len(await r.retrieve(RetrievalQuery("respira", E.MIEDO, None, 1))) == 1


async def test_retrieve_incluye_institucional_solo_si_se_pide(db):
    r = await poblar(db)
    sin = await r.retrieve(RetrievalQuery("respira", E.MIEDO, None, 5))
    con = await r.retrieve(RetrievalQuery("respira", E.MIEDO, None, 5, incluir_institucional=True))
    assert "inst-bienestar" not in {c.doc_id for c in sin}
    assert con[-1].doc_id == "inst-bienestar" and len(con) == len(sin) + 1
    assert {"fuente", "titulo", "texto", "tipo"} <= set(con[-1].__dataclass_fields__)


async def test_institutional_info_no_calcula_embeddings(db):
    await poblar(db)
    emb = ToyEmbedder(fail=True)
    out = await PgVectorRetriever(db, emb).institutional_info()
    assert [c.doc_id for c in out] == ["inst-bienestar"] and emb.calls == 0


async def test_idempotencia_y_reemplazo_de_version(db):
    r = PgVectorRetriever(db, ToyEmbedder())
    d1 = doc("tec-resp", "Respiración", "respira lento", DocumentType.TECNICA, {E.MIEDO}, "respiracion", "v1")
    await r.save(d1, ["respira lento", "respira profundo"], await ToyEmbedder().embed(["a", "b"]))
    assert await r.exists("tec-resp", "v1") and not await r.exists("tec-resp", "v2")
    assert await count(db, "rag_chunks", "fuente = 'tec-resp'") == 2

    d2 = doc("tec-resp", "Respiración", "respira nuevo", DocumentType.TECNICA, {E.MIEDO}, "respiracion", "v2")
    await r.save(d2, ["respira nuevo"], await ToyEmbedder().embed(["a"]))
    assert not await r.exists("tec-resp", "v1") and await r.exists("tec-resp", "v2")
    assert await count(db, "rag_chunks", "fuente = 'tec-resp'") == 1  # la v1 se borró


async def test_save_es_atomico_si_un_fragmento_falla_no_queda_nada_a_medias(db):
    r = PgVectorRetriever(db, ToyEmbedder())
    d1 = doc("tec-x", "X", "respira", DocumentType.TECNICA, {E.MIEDO}, "x", "v1")
    await r.save(d1, ["uno"], await ToyEmbedder().embed(["a"]))
    d2 = doc("tec-x", "X", "respira", DocumentType.TECNICA, {E.MIEDO}, "x", "v2")
    # dos fragmentos con el mismo `indice` no es posible; forzamos fallo con un tipo inválido en SQL:
    object.__setattr__(d2, "tipo", type("T", (), {"value": "inventado"})())
    with pytest.raises(Exception, match="(?i)check|violates"):
        await r.save(d2, ["dos"], await ToyEmbedder().embed(["a"]))
    assert await r.exists("tec-x", "v1") and not await r.exists(
        "tec-x", "v2"
    )  # la v1 sigue intacta (rollback del DELETE)


# ------------------------------------------------------------------ migrador
async def test_migrador_es_atomico_e_idempotente_sobre_la_bd_real(db, tmp_path):
    (tmp_path / "901_ok.sql").write_text("CREATE TABLE zz_it_ok (id int);", encoding="utf-8")
    (tmp_path / "902_rota.sql").write_text(
        "CREATE TABLE zz_it_rota (id int); SELECT * FROM tabla_que_no_existe;", encoding="utf-8"
    )
    try:
        with pytest.raises(Exception, match="(?i)tabla_que_no_existe|does not exist"):
            await Migrator(db, tmp_path).run()
        async with db.transaction() as tx:
            ok = await tx.fetch_one("SELECT to_regclass('zz_it_ok') AS t")
            rota = await tx.fetch_one("SELECT to_regclass('zz_it_rota') AS t")
        assert ok["t"] is not None and rota["t"] is None  # la 902 se revirtió completa
        assert await count(db, "schema_migrations", "name = '901_ok.sql'") == 1
        assert await count(db, "schema_migrations", "name = '902_rota.sql'") == 0
        (tmp_path / "902_rota.sql").unlink()
        assert await Migrator(db, tmp_path).run() == []  # idempotente: la 901 no se repite
    finally:
        async with db.transaction() as tx:
            await tx.execute("DROP TABLE IF EXISTS zz_it_ok, zz_it_rota")
            await tx.execute("DELETE FROM schema_migrations WHERE name LIKE '90%'")


# ------------------------------------------------------------------ contenedor de punta a punta
@pytest.fixture
def container(database_url, monkeypatch):
    monkeypatch.setattr(container_mod, "SentenceTransformerEmbedder", lambda *a, **k: ToyEmbedder())
    return Settings(
        _env_file=None,
        app_env=AppEnv.TEST,
        persistence_backend="postgres",
        retriever_backend="pgvector",
        database_url=database_url,
        store_user_text=False,
    )


async def test_punta_a_punta_con_postgres_y_pgvector_reales(db, container):
    c = build_container(container)
    try:
        await c.ingest_documents.execute(IngestCommand(tuple(seed_documents())))
        again = await c.ingest_documents.execute(IngestCommand(tuple(seed_documents())))
        assert again.ingested == () and len(again.skipped) == len(seed_documents())  # idempotente

        r = await c.analyze.execute(
            AnalyzeCommand("c1", "u-1", SECRETO + " tengo miedo parcial respira", "tr-1")
        )
        assert not r.es_crisis and r.estrategia is not None and r.fuentes
        fb = await c.register_feedback.execute(FeedbackCommand("tr-1", True))
        assert fb.strategy_id == r.estrategia.id and await count(db, "feedback") == 1

        crisis = await c.analyze.execute(AnalyzeCommand("c1", "u-1", "ya no quiero vivir", "tr-2"))
        assert crisis.es_crisis and await count(db, "crisis_events") == 1
        assert (
            await count(db, "messages", "texto LIKE :p", {"p": "%" + SECRETO + "%"}) == 0
        )  # STORE_USER_TEXT=false
        ok, checks = await c.check_readiness()
        assert ok and checks["base_de_datos"] == "ok" and checks["extension_vector"] == "ok"
    finally:
        await c.close()


# ------------------------------------------------------------------ encoder real (opcional)
async def test_encoder_real_cargado_desde_models_encoder():
    model_dir = Path(__file__).resolve().parents[2] / "models" / "encoder"
    if not (model_dir / "config.json").is_file():
        pytest.skip("No hay modelo en models/encoder")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from chatbot.adapters.outbound.classifier.encoder_hf import HFEncoder

    enc = HFEncoder(model_dir, None)
    await enc.warmup()
    dist = await enc.classify("Tengo mucho miedo por el examen de mañana")
    assert enc.is_ready() and abs(sum(dist.probabilities.values()) - 1.0) < 1e-6
    assert json.dumps(dist.to_dict())


def test_ready_http_real_200_con_postgres_y_pgvector(database_url, container):
    """Sincrónica a propósito: el TestClient corre la app en su propio event loop, y el pool de asyncpg
    debe crearse en ese mismo loop."""
    import asyncio

    from chatbot.adapters.outbound.persistence.database import Database

    async def preparar():
        database = Database(database_url, pool_size=1)
        try:
            await Migrator(database, Path(__file__).resolve().parents[2] / "migrations").run()
        finally:
            await database.close()

    asyncio.run(preparar())
    with TestClient(create_app(container=build_container(container))) as client:
        r = client.get("/ready")
    assert r.status_code == 200, r.json()
    assert r.json()["checks"]["base_de_datos"] == "ok" and r.json()["checks"]["extension_vector"] == "ok"
