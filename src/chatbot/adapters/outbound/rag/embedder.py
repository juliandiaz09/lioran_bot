"""Embedder de desarrollo: bolsa de palabras con hashing (determinista, sin modelo).

El embedder real (sentence-transformers multilingüe) se añade aquí en la Fase 5.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from chatbot.domain.services.secondary_emotions import normalize_text


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
