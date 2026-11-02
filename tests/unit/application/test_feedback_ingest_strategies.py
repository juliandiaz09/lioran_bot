import pytest

from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.memory.knowledge import InMemoryKnowledgeBase, InMemoryStrategyCatalog
from chatbot.adapters.outbound.memory.seed import seed_documents, seed_strategies
from chatbot.adapters.outbound.rag.chunker import ParagraphChunker
from chatbot.adapters.outbound.rag.embedder import HashEmbedder
from chatbot.application.dto import CatalogDocument, DocumentType, FeedbackCommand, IngestCommand
from chatbot.application.errors import (
    AnalysisNotFound,
    DuplicateFeedback,
    FeedbackNotApplicable,
    IngestError,
)
from chatbot.application.use_cases import IngestDocuments, ListStrategies
from chatbot.domain.errors import InvalidFeedback
from chatbot.domain.models import Emotion
from helpers import TEXTO_CRISIS, command, make_harness


# ------------------------------------------------------------------ feedback
async def test_feedback_guarda_y_actualiza_al_bandido():
    h = await make_harness()
    r = await h.use_case.execute(command())
    res = await h.register_feedback.execute(FeedbackCommand("trace-1", True, "me sirvió"))
    assert res.strategy_id == r.estrategia.id and res.reward == 1.0
    assert h.feedback_repo.items["trace-1"].comentario == "me sirvió"
    cell, strategy, reward = h.selector.updates[0]
    assert cell.startswith("miedo|") and strategy == r.estrategia.id and reward == 1.0


async def test_feedback_con_escala():
    h = await make_harness()
    await h.use_case.execute(command())
    res = await h.register_feedback.execute(FeedbackCommand("trace-1", 4))
    assert res.reward == pytest.approx(0.75)


async def test_feedback_trace_desconocido():
    h = await make_harness()
    with pytest.raises(AnalysisNotFound):
        await h.register_feedback.execute(FeedbackCommand("no-existe", True))


async def test_feedback_duplicado():
    h = await make_harness()
    await h.use_case.execute(command())
    await h.register_feedback.execute(FeedbackCommand("trace-1", True))
    with pytest.raises(DuplicateFeedback):
        await h.register_feedback.execute(FeedbackCommand("trace-1", False))
    assert len(h.selector.updates) == 1  # no se recompensa dos veces


async def test_feedback_no_aplica_a_crisis():
    h = await make_harness()
    await h.use_case.execute(command(TEXTO_CRISIS))
    with pytest.raises(FeedbackNotApplicable):
        await h.register_feedback.execute(FeedbackCommand("trace-1", True))


async def test_feedback_invalido():
    h = await make_harness()
    await h.use_case.execute(command())
    with pytest.raises(InvalidFeedback):
        await h.register_feedback.execute(FeedbackCommand("trace-1", 9))


async def test_si_el_bandido_falla_el_feedback_queda_guardado():
    h = await make_harness()
    await h.use_case.execute(command())

    async def boom(*a, **k):
        raise RuntimeError("caído")

    h.selector.update = boom  # type: ignore[method-assign]
    res = await h.register_feedback.execute(FeedbackCommand("trace-1", True))
    assert res.reward == 1.0 and "trace-1" in h.feedback_repo.items


# ------------------------------------------------------------------ ingesta
def _ingestor(kb: InMemoryKnowledgeBase) -> IngestDocuments:
    return IngestDocuments(chunker=ParagraphChunker(), embedder=HashEmbedder(), store=kb)


async def test_ingesta_guarda_y_es_recuperable():
    kb = InMemoryKnowledgeBase()
    res = await _ingestor(kb).execute(IngestCommand(tuple(seed_documents())))
    assert len(res.ingested) == 6 and res.skipped == () and res.total_chunks >= 6
    assert await kb.exists("tec-respiracion-4-6", "v1")


async def test_ingesta_idempotente_por_fuente_y_version():
    kb = InMemoryKnowledgeBase()
    ing = _ingestor(kb)
    cmd = IngestCommand(tuple(seed_documents()))
    await ing.execute(cmd)
    again = await ing.execute(cmd)
    assert again.ingested == () and len(again.skipped) == 6 and again.total_chunks == 0


async def test_nueva_version_reemplaza_a_la_anterior():
    kb = InMemoryKnowledgeBase()
    ing = _ingestor(kb)
    doc = seed_documents()[0]
    await ing.execute(IngestCommand((doc,)))
    nuevo = CatalogDocument(
        doc.titulo,
        "Texto nuevo de la técnica.",
        doc.tipo,
        doc.fuente,
        "v2",
        doc.emociones_objetivo,
        doc.estrategia_id,
    )
    res = await ing.execute(IngestCommand((nuevo,)))
    assert res.ingested == (f"{doc.fuente}@v2",)
    assert await kb.exists(doc.fuente, "v2") and not await kb.exists(doc.fuente, "v1")


async def test_ingesta_divide_documentos_largos_en_varios_fragmentos():
    kb = InMemoryKnowledgeBase()
    largo = CatalogDocument(
        "Largo",
        "\n\n".join(["párrafo " * 60] * 5),
        DocumentType.PSICOEDUCACION,
        "psico-1",
        "v1",
        frozenset({Emotion.TRISTEZA}),
    )
    res = await _ingestor(kb).execute(IngestCommand((largo,)))
    assert res.total_chunks > 1


@pytest.mark.parametrize("campo", ["titulo", "texto", "fuente", "version"])
async def test_ingesta_rechaza_campos_vacios_sin_escribir_nada(campo):
    kb = InMemoryKnowledgeBase()
    base = dict(titulo="t", texto="x", tipo=DocumentType.TECNICA, fuente="f", version="v1")
    malo = CatalogDocument(**{**base, campo: "  "})
    ok = seed_documents()[0]
    with pytest.raises(IngestError):
        await _ingestor(kb).execute(IngestCommand((ok, malo)))
    assert not await kb.exists(ok.fuente, ok.version)  # validación previa: nada quedó guardado


async def test_ingesta_sin_documentos():
    with pytest.raises(IngestError):
        await _ingestor(InMemoryKnowledgeBase()).execute(IngestCommand(()))


async def test_ingesta_embeddings_inconsistentes():
    class Mala:
        async def embed(self, textos):
            return []

    ing = IngestDocuments(chunker=ParagraphChunker(), embedder=Mala(), store=InMemoryKnowledgeBase())
    with pytest.raises(IngestError):
        await ing.execute(IngestCommand((seed_documents()[0],)))


async def test_ingesta_documento_sin_fragmentos():
    class Vacio:
        def split(self, texto):
            return []

    ing = IngestDocuments(chunker=Vacio(), embedder=HashEmbedder(), store=InMemoryKnowledgeBase())
    with pytest.raises(IngestError):
        await ing.execute(IngestCommand((seed_documents()[0],)))


# ------------------------------------------------------------------ estrategias
async def test_listar_estrategias():
    res = await ListStrategies(catalog=InMemoryStrategyCatalog(seed_strategies())).execute()
    assert {s.id for s in res} >= {"respiracion_4_6", "grounding_5_sentidos"}


async def test_catalogo_filtra_por_emocion():
    cat = InMemoryStrategyCatalog(seed_strategies())
    assert all(s.applies_to(Emotion.IRA) for s in await cat.for_emotion(Emotion.IRA))
    assert StubEncoder and FakeLLM  # los falsos son importables desde adapters
