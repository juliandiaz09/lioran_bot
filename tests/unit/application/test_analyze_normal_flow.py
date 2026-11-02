import logging

import pytest

from chatbot.application.dto import AnalyzeConfig
from chatbot.application.errors import EmptyMessage
from chatbot.application.templates import SAFE_TEMPLATE
from chatbot.domain.models import Emotion, RiskLevel
from helpers import MIEDO, TEXTO_NORMAL, command, make_harness


async def test_flujo_normal_completo():
    h = await make_harness()
    r = await h.use_case.execute(command())

    assert r.trace_id == "trace-1"
    assert r.emocion_dominante is Emotion.MIEDO
    assert r.distribucion is not None and r.distribucion.top() is Emotion.MIEDO
    assert r.riesgo is RiskLevel.NONE and r.es_crisis is False
    assert r.estrategia is not None and r.estrategia.id == "respiracion_4_6"
    assert r.respuesta.rstrip().endswith("?")
    assert r.respuesta != SAFE_TEMPLATE
    assert {f.doc_id for f in r.fuentes} == {"tec-respiracion-4-6"}
    assert set(r.modelos_usados) == {"encoder", "llm"} and r.degradado is False
    assert r.desacuerdo_modelos == pytest.approx(0.0)


async def test_persiste_mensajes_y_analisis():
    h = await make_harness()
    await h.use_case.execute(command())
    assert [m.rol.value for m in h.repo.messages["conv-1"]] == ["user", "assistant"]
    rec = h.repo.analyses["trace-1"]
    assert rec.estrategia_id == "respiracion_4_6" and rec.user_ref == "u-seudonimo"
    assert rec.bandit_context is not None and rec.bandit_context.cell_key.startswith("miedo|")
    assert rec.assistant_message_id == h.repo.messages["conv-1"][1].id


async def test_deriva_emociones_secundarias():
    h = await make_harness()
    r = await h.use_case.execute(command("Estoy estresado por el parcial y tengo nervios"))
    assert "ansiedad" in r.emociones_secundarias and "estres" in r.emociones_secundarias


async def test_normaliza_solo_espacios_y_conserva_emojis_y_errores():
    h = await make_harness()
    await h.use_case.execute(command("  tengo   miedoo \n\n por el   parcial 😟  "))
    assert h.encoder.calls == ["tengo miedoo por el parcial 😟"]


async def test_limita_la_longitud():
    h = await make_harness(config=AnalyzeConfig(max_text_length=20))
    await h.use_case.execute(command("a" * 500))
    assert len(h.encoder.calls[0]) == 20


@pytest.mark.parametrize("texto", ["", "   ", "\n\t"])
async def test_mensaje_vacio(texto):
    h = await make_harness()
    with pytest.raises(EmptyMessage):
        await h.use_case.execute(command(texto))


async def test_el_generador_recibe_solo_contexto_recuperado_y_estrategia():
    h = await make_harness()
    await h.use_case.execute(command())
    ctx = h.llm.generate_calls[0]
    assert ctx.estrategia is not None and ctx.estrategia.id == "respiracion_4_6"
    assert ctx.fragmentos and all(f.tipo != "institucional" for f in ctx.fragmentos)
    assert ctx.correcciones == ()


async def test_riesgo_medio_incluye_informacion_institucional():
    h = await make_harness()
    r = await h.use_case.execute(command("no puedo más con esto, tengo miedo"))
    assert r.riesgo is RiskLevel.MEDIUM and r.es_crisis is False
    assert "inst-bienestar" in {f.doc_id for f in r.fuentes}


async def test_desacuerdo_entre_modelos():
    from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
    from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
    from chatbot.application.dto import LLMClassification
    from chatbot.domain.models import EmotionDistribution

    h = await make_harness(
        encoder=StubEncoder(fixed=EmotionDistribution.certain(Emotion.MIEDO)),
        llm=FakeLLM(LLMClassification(EmotionDistribution.certain(Emotion.IRA), RiskLevel.NONE)),
    )
    r = await h.use_case.execute(command())
    assert r.desacuerdo_modelos > 0.9


async def test_los_logs_no_contienen_el_texto_del_usuario(caplog):
    h = await make_harness()
    secreto = "zorro_secreto_8841"
    with caplog.at_level(logging.DEBUG):
        await h.use_case.execute(command(f"{secreto} tengo miedo"))
    assert caplog.records
    assert all(secreto not in str(r.__dict__) for r in caplog.records)


async def test_el_estado_inicial_del_arnes_usa_la_distribucion_esperada():
    assert MIEDO.top() is Emotion.MIEDO and TEXTO_NORMAL
