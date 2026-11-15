import asyncio
import logging
import sys

import pytest

from chatbot.adapters.outbound.rag.embedder import EmbedderError, HashEmbedder, SentenceTransformerEmbedder


class FakeST:
    def __init__(self, dim=4, out_dim=None):
        self.dim, self.out_dim = dim, out_dim or dim
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, textos, **kw):
        self.encode_calls.append((list(textos), kw))
        return [[float(len(t))] * self.out_dim for t in textos]


def emb(model=None, dim=4, **kw):
    return SentenceTransformerEmbedder("m", dim, model_factory=lambda _n: model or FakeST(dim), **kw)


async def test_embed_devuelve_una_lista_de_floats_por_texto():
    out = await emb().embed(["hola", "buenos dias"])
    assert out == [[4.0] * 4, [11.0] * 4]


async def test_pide_vectores_normalizados():
    model = FakeST()
    await emb(model).embed(["x"])
    assert model.encode_calls[0][1]["normalize_embeddings"] is True


async def test_lista_vacia_no_carga_el_modelo():
    e = emb()
    assert await e.embed([]) == [] and not e.is_ready()


async def test_dimension_distinta_a_embedding_dim_falla_en_warmup():
    e = emb(FakeST(dim=768), dim=384)
    with pytest.raises(EmbedderError, match="768.*EMBEDDING_DIM=384"):
        await e.warmup()
    assert not e.is_ready()


async def test_vector_de_dimension_inesperada_falla_al_embeber():
    with pytest.raises(EmbedderError, match="dimensión 3"):
        await emb(FakeST(dim=4, out_dim=3)).embed(["x"])


async def test_is_ready_y_carga_unica_concurrente():
    cargas = 0

    def factory(_name):
        nonlocal cargas
        cargas += 1
        return FakeST()

    e = SentenceTransformerEmbedder("m", 4, model_factory=factory)
    assert not e.is_ready()
    await asyncio.gather(*(e.embed([f"t{i}"]) for i in range(8)))
    assert e.is_ready() and cargas == 1


async def test_sin_los_extras_indica_como_instalarlos(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbedderError, match=r"\.\[ml\]"):
        await SentenceTransformerEmbedder().warmup()


async def test_no_registra_el_texto(caplog):
    caplog.set_level(logging.DEBUG)
    await emb().embed(["texto-privado-xyz"])
    assert "texto-privado-xyz" not in "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)


def test_dimension_esperada_invalida():
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(expected_dim=0)


async def test_hash_embedder_se_conserva():
    assert len((await HashEmbedder(16).embed(["hola mundo"]))[0]) == 16
