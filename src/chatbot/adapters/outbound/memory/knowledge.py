"""Catálogo de estrategias y base de conocimiento (RAG) en memoria.

`InMemoryKnowledgeBase` implementa a la vez `DocumentStorePort` y `RetrieverPort`, de modo que
lo ingerido es lo que se recupera. La versión real usa pgvector (Fase 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chatbot.application.dto import (
    CatalogDocument,
    DocumentType,
    RetrievalQuery,
    RetrievedChunk,
)
from chatbot.domain.models import Emotion, Strategy
from chatbot.domain.services.secondary_emotions import normalize_text


class InMemoryStrategyCatalog:
    def __init__(self, strategies: Sequence[Strategy]) -> None:
        self._strategies = list(strategies)

    async def list_all(self) -> Sequence[Strategy]:
        return list(self._strategies)

    async def for_emotion(self, emocion: Emotion) -> Sequence[Strategy]:
        return [s for s in self._strategies if s.applies_to(emocion)]


@dataclass(frozen=True, slots=True)
class _StoredChunk:
    document: CatalogDocument
    texto: str
    embedding: tuple[float, ...]


class InMemoryKnowledgeBase:
    def __init__(self, *, fail_retrieve: bool = False) -> None:
        self.fail_retrieve = fail_retrieve
        self._chunks: dict[str, list[_StoredChunk]] = {}  # fuente -> fragmentos
        self._versions: dict[str, str] = {}

    # --- DocumentStorePort
    async def exists(self, fuente: str, version: str) -> bool:
        return self._versions.get(fuente) == version

    async def save(
        self,
        documento: CatalogDocument,
        fragmentos: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self._chunks[documento.fuente] = [
            _StoredChunk(documento, t, tuple(e)) for t, e in zip(fragmentos, embeddings, strict=True)
        ]
        self._versions[documento.fuente] = documento.version

    # --- RetrieverPort
    async def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievedChunk]:
        if self.fail_retrieve:
            raise RuntimeError("RAG caído (simulado)")
        words = set(normalize_text(query.texto).split())
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunks in self._chunks.values():
            for ch in chunks:
                d = ch.document
                if d.tipo is DocumentType.INSTITUCIONAL:
                    continue
                by_strategy = query.estrategia_id is not None and d.estrategia_id == query.estrategia_id
                by_emotion = query.emocion is not None and query.emocion in d.emociones_objetivo
                if d.tipo is DocumentType.TECNICA and query.estrategia_id is not None:
                    keep = by_strategy  # con estrategia elegida, solo su técnica
                else:
                    keep = by_strategy or by_emotion
                if not keep:
                    continue
                overlap = len(words & set(normalize_text(ch.texto).split())) / (len(words) or 1)
                score = (2.0 if by_strategy else 0.0) + (1.0 if by_emotion else 0.0) + overlap
                scored.append((score, self._to_chunk(ch, score)))
        scored.sort(key=lambda s: -s[0])
        result = [c for _, c in scored[: query.top_k]]
        if query.incluir_institucional:
            result.extend(await self.institutional_info())
        return result

    async def institutional_info(self) -> Sequence[RetrievedChunk]:
        return [
            self._to_chunk(ch, 1.0)
            for chunks in self._chunks.values()
            for ch in chunks
            if ch.document.tipo is DocumentType.INSTITUCIONAL
        ]

    @staticmethod
    def _to_chunk(ch: _StoredChunk, score: float) -> RetrievedChunk:
        d = ch.document
        return RetrievedChunk(d.fuente, d.titulo, ch.texto, d.tipo.value, d.fuente, score)
