"""Divide texto en fragmentos por párrafos, respetando un tamaño máximo."""

from __future__ import annotations


class ParagraphChunker:
    def __init__(self, max_chars: int = 600) -> None:
        if max_chars < 50:
            raise ValueError("max_chars debe ser >= 50")
        self._max = max_chars

    def split(self, texto: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for paragraph in (p.strip() for p in texto.split("\n\n")):
            if not paragraph:
                continue
            for piece in self._split_long(paragraph):
                if current and len(current) + len(piece) + 2 > self._max:
                    chunks.append(current)
                    current = piece
                else:
                    current = f"{current}\n\n{piece}" if current else piece
        if current:
            chunks.append(current)
        return chunks

    def _split_long(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self._max:
            return [paragraph]
        pieces, current = [], ""
        for word in paragraph.split():
            if current and len(current) + len(word) + 1 > self._max:
                pieces.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            pieces.append(current)
        return pieces
