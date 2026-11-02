"""Retroalimentación del usuario sobre una estrategia sugerida."""

from __future__ import annotations

from dataclasses import dataclass

from chatbot.domain.errors import InvalidFeedback

MAX_COMMENT_LEN = 1000


@dataclass(frozen=True, slots=True)
class Feedback:
    """`util` es un bool (útil / no útil) o una escala entera 1-5."""

    message_id: str
    strategy_id: str
    util: bool | int
    comentario: str | None = None

    def __post_init__(self) -> None:
        if not self.message_id or not self.strategy_id:
            raise InvalidFeedback("El feedback requiere message_id y strategy_id")
        if not isinstance(self.util, bool) and not (isinstance(self.util, int) and 1 <= self.util <= 5):
            raise InvalidFeedback("util debe ser bool o un entero entre 1 y 5")
        if self.comentario is not None and len(self.comentario) > MAX_COMMENT_LEN:
            raise InvalidFeedback(f"El comentario excede {MAX_COMMENT_LEN} caracteres")

    @property
    def reward(self) -> float:
        """Recompensa en [0, 1] para el bandido: bool -> 0/1; escala 1-5 -> (n-1)/4."""
        if isinstance(self.util, bool):
            return 1.0 if self.util else 0.0
        return (self.util - 1) / 4.0
