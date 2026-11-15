import time

from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase
from chatbot.application.dto import AnalyzeConfig, LLMClassification
from chatbot.domain.models import RiskLevel
from helpers import MIEDO, TEMPLATES, TEXTO_CRISIS, TEXTO_NORMAL, command, make_harness


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
    assert r.es_crisis and TEMPLATES.crisis_sin_contactos in r.respuesta
    assert not any(ch.isdigit() for ch in r.respuesta)


async def test_crisis_con_catalogo_caido_sigue_respondiendo():
    h = await make_harness(kb=InMemoryKnowledgeBase(fail_retrieve=False))

    async def boom():
        raise RuntimeError("caído")

    h.kb.institutional_info = boom  # type: ignore[method-assign]
    r = await h.use_case.execute(command(TEXTO_CRISIS))
    assert r.es_crisis and TEMPLATES.crisis_sin_contactos in r.respuesta


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


async def test_riesgo_al_final_de_un_mensaje_largo_activa_la_crisis():
    # La frase HIGH queda después del carácter 50: el recorte no debe ocultarla al detector.
    h = await make_harness(config=AnalyzeConfig(max_text_length=50))
    texto = "hoy fue un dia largo en la universidad " * 3 + TEXTO_CRISIS
    assert texto.index(TEXTO_CRISIS) > 50

    r = await h.use_case.execute(command(texto))

    assert r.es_crisis is True and r.riesgo is RiskLevel.HIGH
    assert h.llm.generate_calls == [] and h.llm.classify_calls == [] and h.encoder.calls == []
    assert h.repo.analyses["trace-1"].riesgo_fuente == "reglas"


async def test_riesgo_cortado_por_el_limite_de_longitud_activa_la_crisis():
    # El límite cae a mitad de la frase HIGH: el detector debe verla entera.
    h = await make_harness(config=AnalyzeConfig(max_text_length=50))
    texto = "x" * 40 + " " + TEXTO_CRISIS
    assert texto.index(TEXTO_CRISIS) < 50 < texto.index(TEXTO_CRISIS) + len(TEXTO_CRISIS)

    r = await h.use_case.execute(command(texto))

    assert r.es_crisis is True and r.riesgo is RiskLevel.HIGH


async def test_espacios_de_relleno_no_ocultan_el_riesgo_al_final():
    # Los espacios se normalizan antes de detectar, pero el texto completo llega al detector.
    h = await make_harness(config=AnalyzeConfig(max_text_length=50))
    texto = "estoy cansado" + " \n\t " * 500 + "a" * 60 + " " + TEXTO_CRISIS

    r = await h.use_case.execute(command(texto))

    assert r.es_crisis is True and r.riesgo is RiskLevel.HIGH


async def test_el_encoder_y_el_llm_reciben_como_maximo_max_text_length():
    h = await make_harness(config=AnalyzeConfig(max_text_length=50))
    texto = (TEXTO_NORMAL + " ") * 10

    r = await h.use_case.execute(command(texto))

    assert r.es_crisis is False
    assert h.encoder.calls and all(len(t) <= 50 for t in h.encoder.calls)
    assert h.llm.classify_calls and all(len(t) <= 50 for t in h.llm.classify_calls)
    assert h.llm.generate_calls
    assert all(len(ctx.texto_usuario) <= 50 for ctx in h.llm.generate_calls)
