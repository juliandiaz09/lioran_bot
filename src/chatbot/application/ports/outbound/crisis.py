"""Puerto del registro de eventos de crisis (insumo del seguimiento humano)."""

from __future__ import annotations

from typing import Protocol

from chatbot.application.dto import CrisisEvent


class CrisisEventPort(Protocol):
    async def register(self, event: CrisisEvent) -> None:
        """Registra que un caso quedó marcado para seguimiento humano. Sin texto del usuario."""
        ...
