from __future__ import annotations

from datetime import datetime
from typing import Protocol


class ClockPort(Protocol):
    def now(self) -> datetime:
        """Hora actual con zona horaria local (la franja horaria del bandido depende de ella)."""
        ...


class IdGeneratorPort(Protocol):
    def new_id(self) -> str: ...
