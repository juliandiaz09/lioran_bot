"""Retriever + almacén de documentos sobre Postgres con pgvector.

Una sola clase cumple `RetrieverPort` y `DocumentStorePort` (lo ingerido es lo que se recupera).

- `retrieve`: búsqueda HÍBRIDA = filtros por metadatos + distancia coseno (`<=>`), con `top_k`.
- `institutional_info`: SOLO filtro `tipo='institucional'`, SIN calcular embeddings (ruta de crisis).
- `save`: una transacción (borra versiones anteriores de la `fuente` e inserta los fragmentos nuevos).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from chatbot.adapters.outbound.persistence.database import SqlDatabase
from chatbot.application.dto import CatalogDocument, RetrievalQuery, RetrievedChunk
from chatbot.application.ports.outbound import EmbedderPort


class RetrieverError(Exception):
    """Embedding con dimensión inválida o datos de ingesta inconsistentes."""


def vector_literal(vector: Sequence[float], expected_dim: int) -> str:
    """Formato de texto de pgvector (`[0.1,0.2,...]`). Valida dimensión y valores finitos."""
    if len(vector) != expected_dim:
        raise RetrieverError(f"Embedding de dimensión {len(vector)}; se esperaba {expected_dim}")
    if not all(math.isfinite(x) for x in vector):
        raise RetrieverError("El embedding contiene valores no finitos")
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def build_retrieval_query(query: RetrievalQuery, embedding: str) -> tuple[str, dict[str, Any]] | None:
    """SQL y parámetros de `retrieve`; `None` si ningún filtro aplica (sin estrategia ni emoción).

    - Con `estrategia_id`: las TÉCNICAS se filtran por esa estrategia; sin ella, por emoción objetivo.
    - La PSICOEDUCACIÓN se filtra por emoción objetivo.
    - La información institucional NO entra aquí (se agrega aparte si `incluir_institucional`).
    """
    params: dict[str, Any] = {"q": embedding, "k": query.top_k}
    branches: list[str] = []
    if query.estrategia_id is not None:
        branches.append("(tipo = 'tecnica' AND estrategia_id = :estrategia)")
        params["estrategia"] = query.estrategia_id
    if query.emocion is not None:
        params["emocion"] = [query.emocion.value]
        if query.estrategia_id is None:
            branches.append("(tipo = 'tecnica' AND emociones_objetivo @> CAST(:emocion AS text[]))")
        branches.append("(tipo = 'psicoeducacion' AND emociones_objetivo @> CAST(:emocion AS text[]))")
    if not branches:
        return None
    sql = (
        "SELECT fuente, titulo, texto, tipo, embedding <=> CAST(:q AS vector) AS distancia\n"
        "FROM rag_chunks\n"
        f"WHERE {' OR '.join(branches)}\n"
        "ORDER BY distancia ASC, id ASC\n"
        "LIMIT :k"
    )
    return sql, params


_INSTITUTIONAL = """
SELECT fuente, titulo, texto, tipo
FROM rag_chunks WHERE tipo = 'institucional'
ORDER BY fuente, indice
"""

_INSERT_CHUNK = """
INSERT INTO rag_chunks (
    fuente, version, titulo, tipo, estrategia_id, emociones_objetivo, indice, texto, embedding
)
VALUES (:fuente, :version, :titulo, :tipo, :estrategia_id, CAST(:emociones AS text[]), :indice, :texto,
        CAST(:embedding AS vector))
"""


class PgVectorRetriever:
    def __init__(self, database: SqlDatabase, embedder: EmbedderPort, *, embedding_dim: int = 384) -> None:
        self._db = database
        self._embedder = embedder
        self._dim = embedding_dim

    # ------------------------------------------------------------------------- RetrieverPort
    async def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        if query.estrategia_id is not None or query.emocion is not None:
            embedding = (await self._embedder.embed([query.texto]))[0]
            built = build_retrieval_query(query, vector_literal(embedding, self._dim))
            if built is not None:
                sql, params = built
                async with self._db.transaction() as tx:
                    rows = await tx.fetch_all(sql, params)
                chunks = [
                    RetrievedChunk(
                        doc_id=r["fuente"],
                        titulo=r["titulo"],
                        texto=r["texto"],
                        tipo=r["tipo"],
                        fuente=r["fuente"],
                        score=1.0 - float(r["distancia"]),
                    )
                    for r in rows
                ]
        if query.incluir_institucional:
            chunks.extend(await self.institutional_info())
        return chunks

    async def institutional_info(self) -> Sequence[RetrievedChunk]:
        async with self._db.transaction() as tx:
            rows = await tx.fetch_all(_INSTITUTIONAL)
        return [
            RetrievedChunk(r["fuente"], r["titulo"], r["texto"], r["tipo"], r["fuente"], 1.0) for r in rows
        ]

    # ------------------------------------------------------------------------- DocumentStorePort
    async def exists(self, fuente: str, version: str) -> bool:
        async with self._db.transaction() as tx:
            row = await tx.fetch_one(
                "SELECT 1 AS ok FROM rag_chunks WHERE fuente = :fuente AND version = :version LIMIT 1",
                {"fuente": fuente, "version": version},
            )
        return row is not None

    async def save(
        self,
        documento: CatalogDocument,
        fragmentos: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not fragmentos or len(fragmentos) != len(embeddings):
            raise RetrieverError("Se requiere al menos un fragmento y un embedding por fragmento")
        emociones = sorted(e.value for e in documento.emociones_objetivo)
        rows = [
            {
                "fuente": documento.fuente,
                "version": documento.version,
                "titulo": documento.titulo,
                "tipo": documento.tipo.value,
                "estrategia_id": documento.estrategia_id,
                "emociones": emociones,
                "indice": i,
                "texto": texto,
                "embedding": vector_literal(embedding, self._dim),
            }
            for i, (texto, embedding) in enumerate(zip(fragmentos, embeddings, strict=True))
        ]  # se valida TODO antes de abrir la transacción
        async with self._db.transaction() as tx:
            await tx.execute("DELETE FROM rag_chunks WHERE fuente = :fuente", {"fuente": documento.fuente})
            await tx.execute(_INSERT_CHUNK, rows)
