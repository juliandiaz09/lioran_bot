"""Ingesta de documentos del catálogo al RAG. Idempotente por (fuente, version)."""

from __future__ import annotations

import logging

from chatbot.application.dto import CatalogDocument, IngestCommand, IngestResult
from chatbot.application.errors import IngestError
from chatbot.application.ports.outbound import ChunkerPort, DocumentStorePort, EmbedderPort

logger = logging.getLogger(__name__)


class IngestDocuments:
    def __init__(self, *, chunker: ChunkerPort, embedder: EmbedderPort, store: DocumentStorePort):
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    async def execute(self, command: IngestCommand) -> IngestResult:
        if not command.documents:
            raise IngestError("No se recibieron documentos")
        for doc in command.documents:  # validar todo antes de escribir nada
            self._validate(doc)

        ingested: list[str] = []
        skipped: list[str] = []
        total_chunks = 0
        for doc in command.documents:
            key = f"{doc.fuente}@{doc.version}"
            if await self._store.exists(doc.fuente, doc.version):
                skipped.append(key)
                continue
            chunks = self._chunker.split(doc.texto)
            if not chunks:
                raise IngestError(f"El documento {key} no produjo fragmentos")
            embeddings = await self._embedder.embed(chunks)
            if len(embeddings) != len(chunks):
                raise IngestError(f"Embeddings inconsistentes para {key}")
            await self._store.save(doc, chunks, embeddings)
            ingested.append(key)
            total_chunks += len(chunks)
        logger.info(
            "ingesta_completada",
            extra={
                "ingeridos": len(ingested),
                "omitidos": len(skipped),
                "fragmentos": total_chunks,
            },
        )
        return IngestResult(tuple(ingested), tuple(skipped), total_chunks)

    @staticmethod
    def _validate(doc: CatalogDocument) -> None:
        for campo, valor in (
            ("titulo", doc.titulo),
            ("texto", doc.texto),
            ("fuente", doc.fuente),
            ("version", doc.version),
        ):
            if not valor.strip():
                raise IngestError(f"El campo '{campo}' no puede estar vacío")
