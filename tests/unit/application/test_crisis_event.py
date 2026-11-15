"""Registro del evento de crisis para el seguimiento humano (sin texto del usuario)."""

import logging

from chatbot.adapters.outbound.memory.crisis import InMemoryCrisisEventRepository
from chatbot.adapters.outbound.risk.rules_risk_detector import RulesRiskDetector
from chatbot.application.dto import LLMClassification
from chatbot.domain.models import RiskLevel
from helpers import MIEDO, RAIZ, TEXTO_NORMAL, command, make_harness

SECRETO = "quiero morirme, soy una carga para todos"
DETECTOR = RulesRiskDetector.from_file(RAIZ / "config" / "risk_lexicon.yaml")


async def test_crisis_por_reglas_registra_un_evento_con_categorias_y_sin_texto():
    h = await make_harness(risk_detector=DETECTOR)
    r = await h.use_case.execute(command(SECRETO, trace="t-crisis", conv="c-9"))

    assert r.es_crisis and len(h.crisis_events.events) == 1
    ev = h.crisis_events.events[0]
    assert (ev.trace_id, ev.conversation_id, ev.user_ref) == ("t-crisis", "c-9", "u-seudonimo")
    assert "ideacion_suicida" in ev.categorias and ev.fuente_riesgo == "reglas"
    assert SECRETO not in str(ev) and "morirme" not in str(ev)


async def test_crisis_elevada_por_el_llm_tambien_registra_evento():
    from chatbot.adapters.outbound.llm.llm_fake import FakeLLM

    h = await make_harness(llm=FakeLLM(LLMClassification(MIEDO, RiskLevel.HIGH, "indirecta")))
    r = await h.use_case.execute(command(TEXTO_NORMAL))
    assert r.es_crisis and len(h.crisis_events.events) == 1
    assert h.crisis_events.events[0].fuente_riesgo == "llm" and h.crisis_events.events[0].categorias == ()


async def test_sin_crisis_no_hay_evento():
    h = await make_harness(risk_detector=DETECTOR)
    r = await h.use_case.execute(command(TEXTO_NORMAL))
    assert not r.es_crisis and h.crisis_events.events == []


async def test_si_falla_el_registro_igual_se_responde_con_la_plantilla_de_crisis():
    h = await make_harness(risk_detector=DETECTOR)
    h.crisis_events.fail = True
    r = await h.use_case.execute(command(SECRETO))
    assert r.es_crisis and "Bienestar Universitario" in r.respuesta and h.crisis_events.events == []


async def test_el_evento_se_loguea_como_warning_sin_el_texto_del_usuario(caplog):
    h = await make_harness(risk_detector=DETECTOR)
    with caplog.at_level(logging.DEBUG):
        await h.use_case.execute(command(SECRETO))
    avisos = [r for r in caplog.records if r.getMessage() == "evento_crisis"]
    assert len(avisos) == 1 and avisos[0].levelno == logging.WARNING
    assert all(SECRETO not in str(r.__dict__) and "morirme" not in str(r.__dict__) for r in caplog.records)


def test_el_repositorio_en_memoria_acumula_eventos():
    assert InMemoryCrisisEventRepository().events == []
