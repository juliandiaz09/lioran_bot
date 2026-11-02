"""Flujo completo con el contenedor real y los adaptadores falsos: ingesta -> analizar -> feedback."""

from pydantic import SecretStr

from chatbot.adapters.outbound.memory.seed import seed_documents
from chatbot.application.dto import AnalyzeCommand, FeedbackCommand, IngestCommand
from chatbot.config.container import build_container
from chatbot.config.settings import AppEnv, Settings


async def test_flujo_completo_con_adaptadores_falsos():
    c = build_container(Settings(_env_file=None, app_env=AppEnv.TEST, service_api_key=SecretStr("k")))
    await c.ingest_documents.execute(IngestCommand(tuple(seed_documents())))
    assert len(await c.list_strategies.execute()) == 5

    r = await c.analyze.execute(
        AnalyzeCommand("c1", "u-1", "Tengo mucho miedo, mañana es el parcial", "tr-1")
    )
    assert r.es_crisis is False and r.estrategia is not None and r.fuentes
    assert r.respuesta.rstrip().endswith("?")

    fb = await c.register_feedback.execute(FeedbackCommand("tr-1", True))
    assert fb.strategy_id == r.estrategia.id

    crisis = await c.analyze.execute(AnalyzeCommand("c1", "u-1", "ya no quiero vivir", "tr-2"))
    assert crisis.es_crisis is True
