"""Repos SQL con un doble de BD: armado de consultas, mapeo de filas, atomicidad y privacidad."""

import json
import logging
from datetime import UTC, datetime

import pytest

from chatbot.adapters.outbound.persistence.crisis_event_repo import CrisisEventRepository
from chatbot.adapters.outbound.persistence.supabase_conversation_repo import (
    OMITTED_TEXT,
    SupabaseConversationRepository,
)
from chatbot.adapters.outbound.persistence.supabase_feedback_repo import SupabaseFeedbackRepository
from chatbot.application.dto import AnalysisRecord, CrisisEvent, SourceRef
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
from doubles.fake_sql import FakeDatabase

NOW = datetime(2026, 11, 15, 10, 0, tzinfo=UTC)
SECRETO = "texto-privado-del-estudiante-987"


def turn(texto=SECRETO, dist=True):
    user = Message("m-u", "c1", texto, Role.USER, NOW)
    bot = Message("m-b", "c1", "Respuesta del bot ¿Te sirvió?", Role.ASSISTANT, NOW)
    d = EmotionDistribution.certain(Emotion.MIEDO) if dist else None
    ctx = BanditContext(Emotion.MIEDO, Certainty.ALTA, RiskLevel.LOW, TimeOfDay.MANANA, ("a",))
    rec = AnalysisRecord(
        "t1", "c1", "user-ref-1", "m-u", "m-b", d, ("ansiedad",), 0.25, RiskLevel.LOW, "reglas",
        "resp", ctx if dist else None, False, False, (SourceRef("d1", "Doc 1"),), NOW,
    )  # fmt: skip
    return user, bot, rec


# ------------------------------------------------------------------ record_turn
async def test_record_turn_inserta_conversacion_dos_mensajes_y_analisis_en_una_transaccion():
    db = FakeDatabase()
    await SupabaseConversationRepository(db).record_turn(*turn())
    assert db.transactions == 1
    kinds = [s.split()[2] for _, s, _ in db.committed]  # INSERT INTO <tabla>
    assert kinds == ["conversations", "messages", "messages", "analyses"]
    (_, user_p), (_, bot_p) = db.executed("INTO messages")
    assert user_p["rol"] == "user" and user_p["texto"] == SECRETO and user_p["longitud"] == len(SECRETO)
    assert bot_p["rol"] == "assistant"
    (_, a) = db.executed("INTO analyses")[0]
    assert a["riesgo"] == 1 and json.loads(a["distribucion"])["miedo"] == 1.0
    assert json.loads(a["bandit_context"])["franja"] == "manana"
    assert json.loads(a["fuentes"]) == [{"doc_id": "d1", "titulo": "Doc 1"}]
    assert a["created_at"].tzinfo is not None


async def test_record_turn_atomico_si_falla_a_mitad_no_queda_nada():
    db = FakeDatabase(fail_on="INTO analyses")
    with pytest.raises(RuntimeError):
        await SupabaseConversationRepository(db).record_turn(*turn())
    assert db.committed == []


async def test_analisis_sin_distribucion_guarda_null():
    db = FakeDatabase()
    await SupabaseConversationRepository(db).record_turn(*turn(dist=False))
    (_, a) = db.executed("INTO analyses")[0]
    assert a["distribucion"] is None and a["bandit_context"] is None


async def test_fecha_ingenua_se_interpreta_como_utc():
    user, bot, rec = turn()
    naive = Message("m-u", "c1", "hola", Role.USER, datetime(2026, 1, 1, 8, 0))
    db = FakeDatabase()
    await SupabaseConversationRepository(db).record_turn(naive, bot, rec)
    assert db.executed("INTO messages")[0][1]["created_at"].tzinfo is UTC


# ------------------------------------------------------------------ STORE_USER_TEXT
async def test_store_user_text_false_omite_solo_el_texto_del_usuario():
    db = FakeDatabase()
    await SupabaseConversationRepository(db, store_user_text=False).record_turn(*turn())
    (_, user_p), (_, bot_p) = db.executed("INTO messages")
    assert user_p["texto"] == OMITTED_TEXT == "[omitido]" and user_p["longitud"] == len(SECRETO)
    assert bot_p["texto"] == "Respuesta del bot ¿Te sirvió?"  # el asistente sí se guarda
    assert SECRETO not in json.dumps([p for _, _, p in db.committed], default=str)


# ------------------------------------------------------------------ history / get_analysis
async def test_history_ordena_cronologicamente_y_mapea_filas():
    db = FakeDatabase(
        {
            "FROM messages": [  # la consulta devuelve de más nuevo a más viejo
                {"id": "b", "conversation_id": "c1", "texto": "resp", "rol": "assistant", "created_at": NOW},
                {"id": "a", "conversation_id": "c1", "texto": "hola", "rol": "user", "created_at": NOW},
            ],
            "FROM analyses": [
                {"riesgo": 2, "estrategia_id": "s2"},
                {"riesgo": 0, "estrategia_id": None},
                {"riesgo": 1, "estrategia_id": "s1"},
            ],
        }
    )
    h = await SupabaseConversationRepository(db).history("c1", 5)
    assert [m.id for m in h.messages] == ["a", "b"] and h.messages[0].rol is Role.USER
    assert h.risk_levels == (RiskLevel.LOW, RiskLevel.NONE, RiskLevel.MEDIUM)
    assert h.strategy_ids == ("s1", "s2")
    _, params = db.executed("FROM messages")[0]
    assert params == {"conversation_id": "c1", "n": 10}


async def test_history_con_limite_menor_que_uno_no_consulta():
    db = FakeDatabase()
    h = await SupabaseConversationRepository(db).history("c1", 0)
    assert h.messages == () and db.transactions == 0


async def test_get_analysis_reconstruye_el_registro_completo():
    user, bot, rec = turn()
    row = {
        "trace_id": "t1", "conversation_id": "c1", "user_ref": "user-ref-1", "user_message_id": "m-u",
        "assistant_message_id": "m-b", "distribucion": {"miedo": 1.0}, "emociones_secundarias": ["ansiedad"],
        "desacuerdo_modelos": 0.25, "riesgo": 1, "riesgo_fuente": "reglas", "estrategia_id": "resp",
        "bandit_context": rec.bandit_context.to_dict(), "es_crisis": False, "degradado": False,
        "fuentes": [{"doc_id": "d1", "titulo": "Doc 1"}], "created_at": NOW,
    }  # fmt: skip
    got = await SupabaseConversationRepository(FakeDatabase({"FROM analyses": [row]})).get_analysis("t1")
    assert got == rec


async def test_get_analysis_acepta_jsonb_como_texto_y_devuelve_none_si_no_existe():
    row = {
        "trace_id": "t1", "conversation_id": "c1", "user_ref": "u", "user_message_id": "a",
        "assistant_message_id": "b", "distribucion": None, "emociones_secundarias": "[]",
        "desacuerdo_modelos": 0, "riesgo": 3, "riesgo_fuente": "reglas", "estrategia_id": None,
        "bandit_context": None, "es_crisis": True, "degradado": False, "fuentes": "[]", "created_at": NOW,
    }  # fmt: skip
    repo = SupabaseConversationRepository(FakeDatabase({"FROM analyses": [row]}))
    got = await repo.get_analysis("t1")
    assert got.riesgo is RiskLevel.HIGH and got.distribucion is None and got.fuentes == ()
    assert await SupabaseConversationRepository(FakeDatabase()).get_analysis("no") is None


# ------------------------------------------------------------------ feedback
@pytest.mark.parametrize(
    "util, bool_v, escala",
    [(True, True, None), (False, False, None), (4, None, 4), (1, None, 1)],
)
async def test_feedback_guarda_bool_o_escala(util, bool_v, escala):
    db = FakeDatabase()
    await SupabaseFeedbackRepository(db).save("t1", Feedback("m1", "s1", util, "buen consejo"))
    (_, p) = db.executed("INTO feedback")[0]
    assert (p["util_bool"], p["util_escala"]) == (bool_v, escala)
    assert p["trace_id"] == "t1" and p["comentario"] == "buen consejo" and 0 <= p["reward"] <= 1


async def test_feedback_sin_guardar_texto_omite_el_comentario():
    db = FakeDatabase()
    await SupabaseFeedbackRepository(db, store_user_text=False).save(
        "t1", Feedback("m1", "s1", True, SECRETO)
    )
    assert db.executed("INTO feedback")[0][1]["comentario"] is None


async def test_feedback_exists():
    assert await SupabaseFeedbackRepository(FakeDatabase({"FROM feedback": [{"ok": 1}]})).exists("t1")
    assert not await SupabaseFeedbackRepository(FakeDatabase()).exists("t1")


# ------------------------------------------------------------------ crisis
def evento():
    return CrisisEvent("t9", "c1", "user-ref-1", ("ideacion_suicida",), "reglas", NOW)


async def test_crisis_inserta_sin_texto_y_mantiene_el_log_warning(caplog):
    caplog.set_level(logging.WARNING)
    db = FakeDatabase()
    await CrisisEventRepository(db).register(evento())
    (_, p) = db.executed("INTO crisis_events")[0]
    assert json.loads(p["categorias"]) == ["ideacion_suicida"] and "texto" not in p
    assert any(r.getMessage() == "evento_crisis" and r.trace_id == "t9" for r in caplog.records)


async def test_crisis_deja_el_log_aunque_la_bd_falle(caplog):
    caplog.set_level(logging.WARNING)
    with pytest.raises(RuntimeError):
        await CrisisEventRepository(FakeDatabase(fail_on="INTO crisis_events")).register(evento())
    assert any(r.getMessage() == "evento_crisis" for r in caplog.records)


# ------------------------------------------------------------------ privacidad en logs
async def test_los_repos_no_registran_el_texto_del_usuario(caplog):
    caplog.set_level(logging.DEBUG)
    db = FakeDatabase()
    repo = SupabaseConversationRepository(db)
    await repo.record_turn(*turn())
    await repo.history("c1", 5)
    await SupabaseFeedbackRepository(db).save("t1", Feedback("m1", "s1", True, SECRETO))
    await CrisisEventRepository(db).register(evento())
    assert SECRETO not in "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
