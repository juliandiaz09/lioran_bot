from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chatbot.application.dto import CatalogDocument, RetrievalQuery, RetrievedChunk


class RetrieverPort(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievedChunk]:
        """Búsqueda híbrida (vectorial + filtros por metadatos) con la fuente de cada fragmento."""
        ...

    async def institutional_info(self) -> Sequence[RetrievedChunk]:
        """Información institucional (contactos). Solo filtro por metadatos: rápida, sin embeddings."""
        ...


class ChunkerPort(Protocol):
    def split(self, texto: str) -> list[str]: ...


class EmbedderPort(Protocol):
    async def embed(self, textos: Sequence[str]) -> list[list[float]]: ...


class DocumentStorePort(Protocol):
    async def exists(self, fuente: str, version: str) -> bool: ...

    async def save(
        self,
        documento: CatalogDocument,
        fragmentos: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Guarda los fragmentos; reemplaza las versiones anteriores de la misma `fuente`."""
        ...
