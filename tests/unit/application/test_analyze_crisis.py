import time

from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase
from chatbot.application.dto import LLMClassification
from chatbot.application.templates import CRISIS_NO_CONTACTS
from chatbot.domain.models import RiskLevel
from helpers import MIEDO, TEXTO_CRISIS, TEXTO_NORMAL, command, make_harness


async def test_riesgo_alto_por_reglas_cortocircuita_sin_llm_ni_encoder():
    h = await make_harness()
    r = await h.use_case.execute(command(TEXTO_CRISIS))

    assert r.es_crisis is True and r.riesgo is RiskLevel.HIGH
    assert h.llm.generate_calls == [] and h.llm.classify_calls == [] and h.encoder.calls == []
    assert "Bienestar Universitario" in r.respuesta  # contacto salido del catálogo, no del LLM
    assert r.estrategia is None and r.emocion_dominante is None
    assert {f.doc_id for f in r.fuentes} == {"inst-bienestar"}


async def test_crisis_se_registra_para_seguimiento_humano():
    h = await make_harness()
    await h.use_case.execute(command(TEXTO_CRISIS))
    rec = h.repo.analyses["trace-1"]
    assert rec.es_crisis is True and rec.riesgo is RiskLevel.HIGH and rec.estrategia_id is None
    assert rec.riesgo_fuente == "reglas"


async def test_crisis_responde_rapido_aunque_el_llm_este_lento():
    h = await make_harness(llm=FakeLLM(delay_s=5.0))
    t0 = time.perf_counter()
    r = await h.use_case.execute(command(TEXTO_CRISIS))
    assert r.es_crisis and time.perf_counter() - t0 < 1.0


async def test_crisis_sin_catalogo_no_inventa_telefonos():
    h = await make_harness(seed_kb=False)
    r = await h.use_case.execute(command(TEXTO_CRISIS))
    assert r.es_crisis and CRISIS_NO_CONTACTS in r.respuesta
    assert not any(ch.isdigit() for ch in r.respuesta)


async def test_crisis_con_catalogo_caido_sigue_respondiendo():
    h = await make_harness(kb=InMemoryKnowledgeBase(fail_retrieve=False))

    async def boom():
        raise RuntimeError("caído")

    h.kb.institutional_info = boom  # type: ignore[method-assign]
    r = await h.use_case.execute(command(TEXTO_CRISIS))
    assert r.es_crisis and CRISIS_NO_CONTACTS in r.respuesta


async def test_el_llm_puede_elevar_a_crisis():
    llm = FakeLLM(LLMClassification(MIEDO, RiskLevel.HIGH, "ideación indirecta"))
    h = await make_harness(llm=llm)
    r = await h.use_case.execute(command(TEXTO_NORMAL))
    assert r.es_crisis is True and r.riesgo is RiskLevel.HIGH
    assert llm.generate_calls == []  # nunca se llama al generador en crisis
    assert r.distribucion is not None  # aquí sí hubo clasificación
    assert h.repo.analyses["trace-1"].riesgo_fuente == "llm"


async def test_el_historial_eleva_el_riesgo_del_turno_siguiente():
    h = await make_harness()
    await h.use_case.execute(command(TEXTO_CRISIS, trace="t1"))
    r = await h.use_case.execute(command(TEXTO_NORMAL, trace="t2"))
    assert r.riesgo is RiskLevel.MEDIUM and r.es_crisis is False
    assert h.repo.analyses["t2"].riesgo_fuente == "historial"
