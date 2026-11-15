import pytest

from chatbot.adapters.outbound.rag.pgvector_retriever import (
    PgVectorRetriever,
    RetrieverError,
    build_retrieval_query,
    vector_literal,
)
from chatbot.application.dto import CatalogDocument, DocumentType, RetrievalQuery
from chatbot.domain.models import Emotion
from doubles.fake_sql import FakeDatabase

E = Emotion


class Embedder:
    def __init__(self, dim=3, fail=False):
        self.dim, self.fail, self.calls = dim, fail, []

    async def embed(self, textos):
        self.calls.append(list(textos))
        if self.fail:
            raise AssertionError("no debía calcular embeddings")
        return [[0.1] * self.dim for _ in textos]


def q(**kw):
    base = {"texto": "tengo miedo", "emocion": E.MIEDO, "estrategia_id": None, "top_k": 4}
    return RetrievalQuery(**{**base, **kw})


# ------------------------------------------------------------------ armado de la consulta
def test_con_estrategia_las_tecnicas_se_filtran_por_estrategia_y_la_psicoeducacion_por_emocion():
    sql, params = build_retrieval_query(q(estrategia_id="resp"), "[0.1]")
    assert "tipo = 'tecnica' AND estrategia_id = :estrategia" in sql
    assert "tipo = 'psicoeducacion' AND emociones_objetivo @> CAST(:emocion AS text[])" in sql
    assert "tipo = 'tecnica' AND emociones_objetivo" not in sql
    assert params["estrategia"] == "resp" and params["emocion"] == ["miedo"] and params["k"] == 4


def test_sin_estrategia_las_tecnicas_se_filtran_por_emocion():
    sql, params = build_retrieval_query(q(), "[0.1]")
    assert "tipo = 'tecnica' AND emociones_objetivo @>" in sql and "estrategia" not in params


def test_ordena_por_distancia_coseno_ascendente_y_limita_top_k():
    sql, params = build_retrieval_query(q(top_k=7), "[0.1]")
    assert "embedding <=> CAST(:q AS vector)" in sql
    assert "ORDER BY distancia ASC" in sql and sql.rstrip().endswith("LIMIT :k") and params["k"] == 7


def test_la_consulta_nunca_incluye_institucional_y_usa_solo_parametros():
    sql, _ = build_retrieval_query(q(incluir_institucional=True, estrategia_id="x"), "[0.1]")
    assert "institucional" not in sql
    assert "tengo miedo" not in sql and "miedo'" not in sql  # nada de valores incrustados


def test_sin_emocion_ni_estrategia_no_hay_consulta():
    assert build_retrieval_query(q(emocion=None), "[0.1]") is None


def test_solo_estrategia_sin_emocion():
    sql, params = build_retrieval_query(q(emocion=None, estrategia_id="r"), "[0.1]")
    assert "psicoeducacion" not in sql and "emocion" not in params


# ------------------------------------------------------------------ vector_literal
def test_vector_literal_formato_pgvector():
    assert vector_literal([0.5, -1.0, 2], 3) == "[0.5,-1.0,2.0]"


@pytest.mark.parametrize("vec", [[0.1, 0.2], [0.1, float("nan"), 0.2], [float("inf"), 0, 0]])
def test_vector_literal_rechaza_dimension_o_valores_invalidos(vec):
    with pytest.raises(RetrieverError):
        vector_literal(vec, 3)


# ------------------------------------------------------------------ retrieve
ROWS = [
    {"fuente": "tec-1", "titulo": "T1", "texto": "Respira", "tipo": "tecnica", "distancia": 0.2},
    {"fuente": "psi-1", "titulo": "P1", "texto": "Es normal", "tipo": "psicoeducacion", "distancia": 0.5},
]


async def test_retrieve_mapea_filas_a_chunks_con_score_uno_menos_distancia():
    db = FakeDatabase({"FROM rag_chunks\nWHERE": ROWS})
    out = await PgVectorRetriever(db, Embedder(), embedding_dim=3).retrieve(q(estrategia_id="resp"))
    assert [(c.doc_id, c.fuente, c.tipo) for c in out] == [
        ("tec-1", "tec-1", "tecnica"),
        ("psi-1", "psi-1", "psicoeducacion"),
    ]
    assert [round(c.score, 3) for c in out] == [0.8, 0.5]


async def test_retrieve_agrega_institucional_solo_si_se_pide():
    inst = {"fuente": "inst", "titulo": "Bienestar", "texto": "Contacto", "tipo": "institucional"}
    db = FakeDatabase({"FROM rag_chunks\nWHERE": ROWS, "tipo = 'institucional'": [inst]})
    r = PgVectorRetriever(db, Embedder(), embedding_dim=3)
    assert [c.tipo for c in await r.retrieve(q())] == ["tecnica", "psicoeducacion"]
    con = await r.retrieve(q(incluir_institucional=True))
    assert con[-1].tipo == "institucional" and con[-1].score == 1.0


async def test_retrieve_valida_la_dimension_del_embedding():
    with pytest.raises(RetrieverError):
        await PgVectorRetriever(FakeDatabase(), Embedder(dim=5), embedding_dim=3).retrieve(q())


async def test_retrieve_sin_filtros_no_toca_la_bd():
    db = FakeDatabase()
    assert await PgVectorRetriever(db, Embedder(), embedding_dim=3).retrieve(q(emocion=None)) == []
    assert db.transactions == 0


async def test_institutional_info_no_calcula_embeddings():
    inst = {"fuente": "inst", "titulo": "B", "texto": "Contacto", "tipo": "institucional"}
    db = FakeDatabase({"tipo = 'institucional'": [inst]})
    emb = Embedder(fail=True)
    out = await PgVectorRetriever(db, emb, embedding_dim=3).institutional_info()
    assert [c.texto for c in out] == ["Contacto"] and emb.calls == []
    sql = db.executed("institucional")[0][0]
    assert "embedding" not in sql


# ------------------------------------------------------------------ DocumentStorePort
def doc(**kw):
    base = {
        "titulo": "Resp", "texto": "x", "tipo": DocumentType.TECNICA, "fuente": "tec-1", "version": "v2",
        "emociones_objetivo": frozenset({E.MIEDO, E.IRA}), "estrategia_id": "resp",
    }  # fmt: skip
    return CatalogDocument(**{**base, **kw})


async def test_exists_consulta_la_pareja_fuente_version():
    assert await PgVectorRetriever(FakeDatabase({"FROM rag_chunks": [{"ok": 1}]}), Embedder()).exists(
        "a", "v1"
    )
    db = FakeDatabase()
    assert not await PgVectorRetriever(db, Embedder()).exists("a", "v1")
    assert db.executed("fuente = :fuente AND version = :version")[0][1] == {"fuente": "a", "version": "v1"}


async def test_save_borra_versiones_anteriores_e_inserta_en_una_transaccion():
    db = FakeDatabase()
    await PgVectorRetriever(db, Embedder(), embedding_dim=2).save(doc(), ["a", "b"], [[0.1, 0.2], [0.3, 0.4]])
    assert db.transactions == 1
    assert [s.split()[0] for _, s, _ in db.committed] == ["DELETE", "INSERT"]
    assert db.committed[0][2] == {"fuente": "tec-1"}
    rows = db.committed[1][2]
    assert [r["indice"] for r in rows] == [0, 1] and rows[0]["emociones"] == ["ira", "miedo"]
    assert rows[1]["embedding"] == "[0.3,0.4]" and rows[0]["version"] == "v2" and rows[0]["tipo"] == "tecnica"


async def test_save_si_falla_el_insert_no_queda_nada_confirmado():
    db = FakeDatabase(fail_on="INSERT INTO rag_chunks")
    with pytest.raises(RuntimeError):
        await PgVectorRetriever(db, Embedder(), embedding_dim=2).save(doc(), ["a"], [[0.1, 0.2]])
    assert db.committed == []


@pytest.mark.parametrize("frag, emb", [([], []), (["a", "b"], [[0.1, 0.2]]), (["a"], [[0.1]])])
async def test_save_valida_antes_de_abrir_la_transaccion(frag, emb):
    db = FakeDatabase()
    with pytest.raises(RetrieverError):
        await PgVectorRetriever(db, Embedder(), embedding_dim=2).save(doc(), frag, emb)
    assert db.transactions == 0
