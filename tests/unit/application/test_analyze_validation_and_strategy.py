from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.selector import RuleBasedSelector
from chatbot.application.dto import AnalyzeConfig, LLMClassification
from chatbot.domain.models import RiskLevel
from helpers import MIEDO, TEMPLATES, command, make_harness

MALA = "Creo que tienes depresión; toma un antidepresivo."
BUENA = "Respira lento y con calma. ¿Te sirvió esta sugerencia?"


def _llm(*outputs: str) -> FakeLLM:
    return FakeLLM(LLMClassification(MIEDO, RiskLevel.NONE), generate_outputs=outputs)


async def test_validador_rechaza_y_regenera_una_vez():
    llm = _llm(MALA, BUENA)
    h = await make_harness(llm=llm)
    r = await h.use_case.execute(command())
    assert r.respuesta == BUENA
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[0].correcciones == ()
    assert llm.generate_calls[1].correcciones  # el validador explica qué corregir


async def test_si_falla_dos_veces_usa_plantilla_segura():
    llm = _llm(MALA, MALA)
    h = await make_harness(llm=llm)
    r = await h.use_case.execute(command())
    assert r.respuesta == TEMPLATES.segura and len(llm.generate_calls) == 2 and r.fuentes == ()


async def test_sin_regeneraciones_configuradas_solo_un_intento():
    llm = _llm(MALA, BUENA)
    h = await make_harness(llm=llm, config=AnalyzeConfig(max_regenerations=0))
    r = await h.use_case.execute(command())
    assert r.respuesta == TEMPLATES.segura and len(llm.generate_calls) == 1


async def test_rechaza_telefonos_inventados():
    llm = _llm("Llama al 3001234567 ya. ¿Te ayudó?", BUENA)
    h = await make_harness(llm=llm)
    r = await h.use_case.execute(command())
    assert r.respuesta == BUENA


async def test_el_bandido_recibe_el_contexto_y_su_eleccion_se_respeta():
    class Spy(RuleBasedSelector):
        async def select(self, context, candidates):
            self.seen = (context, [c.id for c in candidates])
            return type(await super().select(context, candidates))(candidates[-1])

    spy = Spy()
    h = await make_harness(selector=spy, hour=21)
    r = await h.use_case.execute(command())
    ctx, ids = spy.seen
    assert ctx.emocion.value == "miedo" and ctx.franja.value == "noche" and ctx.riesgo is RiskLevel.NONE
    assert ids == ["respiracion_4_6", "grounding_5_sentidos", "reencuadre_cognitivo", "planificacion_bloques"]
    assert r.estrategia is not None and r.estrategia.id == "planificacion_bloques"


async def test_pasa_las_estrategias_previas_y_evita_repetir():
    h = await make_harness()
    r1 = await h.use_case.execute(command(trace="t1"))
    r2 = await h.use_case.execute(command(trace="t2"))
    assert r1.estrategia is not None and r2.estrategia is not None
    assert r2.estrategia.id != r1.estrategia.id
    assert h.repo.analyses["t2"].bandit_context.estrategias_previas == (r1.estrategia.id,)


async def test_si_el_bandido_falla_degrada_a_la_primera_candidata():
    h = await make_harness(selector=RuleBasedSelector(fail=True))
    r = await h.use_case.execute(command())
    assert r.estrategia is not None and r.estrategia.id == "respiracion_4_6"
    assert r.respuesta != TEMPLATES.segura
