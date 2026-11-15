"""Embedders.

- `HashEmbedder`: bolsa de palabras con hashing (determinista, sin modelo). Desarrollo y pruebas.
- `SentenceTransformerEmbedder`: modelo real multilingüe (sentence-transformers). Importa la librería de
  forma perezosa: sin el extra `ml` el módulo se importa igual.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from chatbot.domain.services.secondary_emotions import normalize_text

logger = logging.getLogger(__name__)


class EmbedderError(Exception):
    """Error de carga o de dimensión del embedder."""


class HashEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    async def embed(self, textos: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in textos]

    def _one(self, texto: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in normalize_text(texto).split():
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """`EmbedderPort` real. Verifica en `warmup()` que la dimensión del modelo sea `EMBEDDING_DIM`.

    `model_factory` permite inyectar un modelo falso en las pruebas: recibe el nombre del modelo y devuelve un
    objeto con `get_sentence_embedding_dimension()` y `encode(textos, ...)`.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        expected_dim: int = 384,
        *,
        device: str = "cpu",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if expected_dim <= 0:
            raise ValueError("expected_dim debe ser > 0")
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._device = device
        self._factory = model_factory
        self._model: Any = None
        self._lock = asyncio.Lock()

    def is_ready(self) -> bool:
        return self._model is not None

    async def warmup(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load)

    async def embed(self, textos: Sequence[str]) -> list[list[float]]:
        if not textos:
            return []
        if self._model is None:
            await self.warmup()
        started = time.perf_counter()
        vectors = await asyncio.to_thread(self._encode, list(textos))
        logger.info(
            "embeddings_generados",
            extra={"cantidad": len(vectors), "duracion_ms": round((time.perf_counter() - started) * 1000)},
        )
        return vectors

    # ------------------------------------------------------------------------- interno
    def _load(self) -> None:
        if self._factory is not None:
            model = self._factory(self._model_name)
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(
                    "RETRIEVER_BACKEND=pgvector requiere los extras de ML: pip install '.[ml]'"
                ) from exc
            model = SentenceTransformer(self._model_name, device=self._device)
        real_dim = int(model.get_sentence_embedding_dimension())
        if real_dim != self._expected_dim:
            raise EmbedderError(
                f"El modelo '{self._model_name}' produce vectores de dimensión {real_dim}, "
                f"pero EMBEDDING_DIM={self._expected_dim}"
            )
        self._model = model
        logger.info("embedder_cargado", extra={"modelo": self._model_name, "dimension": real_dim})

    def _encode(self, textos: list[str]) -> list[list[float]]:
        raw = self._model.encode(textos, normalize_embeddings=True, show_progress_bar=False)
        vectors = [[float(x) for x in row] for row in raw]
        for v in vectors:
            if len(v) != self._expected_dim:
                raise EmbedderError(
                    f"Se obtuvo un vector de dimensión {len(v)}; se esperaba {self._expected_dim}"
                )
        return vectors
