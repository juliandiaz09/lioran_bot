import time

from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase
from chatbot.adapters.outbound.memory.repositories import InMemoryConversationRepository
from chatbot.application.dto import AnalyzeConfig, LLMClassification
from chatbot.application.templates import SAFE_TEMPLATE
from chatbot.domain.models import Emotion, RiskLevel
from helpers import MIEDO, command, make_harness


async def test_cae_el_llm_de_clasificacion_usa_el_encoder():
    h = await make_harness(llm=FakeLLM(fail_classify=True))
    r = await h.use_case.execute(command())
    assert r.modelos_usados == ("encoder",) and r.degradado is True
    assert r.emocion_dominante is Emotion.MIEDO and r.respuesta != SAFE_TEMPLATE


async def test_cae_el_encoder_usa_solo_el_llm():
    h = await make_harness(encoder=StubEncoder(fail=True))
    r = await h.use_case.execute(command())
    assert r.modelos_usados == ("llm",) and r.degradado is True
    assert r.emocion_dominante is Emotion.MIEDO


async def test_cae_todo_responde_con_plantilla_segura_sin_error():
    h = await make_harness(
        encoder=StubEncoder(fail=True), llm=FakeLLM(fail_classify=True, fail_generate=True)
    )
    r = await h.use_case.execute(command())
    assert r.respuesta == SAFE_TEMPLATE and r.degradado is True
    assert r.emocion_dominante is None and r.estrategia is None and r.fuentes == ()
    assert h.repo.analyses["trace-1"].degradado is True  # igualmente persistido


async def test_cae_el_llm_generador_usa_plantilla_segura_pero_conserva_clasificacion():
    h = await make_harness(llm=FakeLLM(LLMClassification(MIEDO, RiskLevel.NONE), fail_generate=True))
    r = await h.use_case.execute(command())
    assert r.respuesta == SAFE_TEMPLATE and r.emocion_dominante is Emotion.MIEDO
    assert r.estrategia is not None and r.fuentes == ()


async def test_timeout_de_un_clasificador_se_trata_como_caida():
    h = await make_harness(
        encoder=StubEncoder(fixed=MIEDO, delay_s=0.5),
        config=AnalyzeConfig(classification_timeout_s=0.05),
    )
    r = await h.use_case.execute(command())
    assert r.modelos_usados == ("llm",)


async def test_encoder_y_llm_se_ejecutan_en_paralelo():
    h = await make_harness(
        encoder=StubEncoder(fixed=MIEDO, delay_s=0.2),
        llm=FakeLLM(LLMClassification(MIEDO, RiskLevel.NONE), delay_s=0.2),
    )
    t0 = time.perf_counter()
    await h.use_case.execute(command())
    # 2 x 0.2 s secuencial = 0.4 s (+ generación 0.2 s del LLM); en paralelo la clasificación cuesta 0.2 s
    assert time.perf_counter() - t0 < 0.55


async def test_rag_caido_usa_plantilla_segura():
    h = await make_harness(kb=InMemoryKnowledgeBase(fail_retrieve=True))
    r = await h.use_case.execute(command())
    assert r.respuesta == SAFE_TEMPLATE and r.fuentes == ()


async def test_rag_sin_resultados_usa_plantilla_segura():
    h = await make_harness(seed_kb=False)
    r = await h.use_case.execute(command())
    assert r.respuesta == SAFE_TEMPLATE
    assert h.llm.generate_calls == []  # sin contexto recuperado no se genera


async def test_bd_caida_no_rompe_la_respuesta():
    h = await make_harness(repo=InMemoryConversationRepository(fail=True))
    r = await h.use_case.execute(command())
    assert r.respuesta != SAFE_TEMPLATE and r.emocion_dominante is Emotion.MIEDO
