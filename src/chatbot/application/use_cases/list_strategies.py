from __future__ import annotations

from collections.abc import Sequence

from chatbot.application.ports.outbound import StrategyCatalogPort
from chatbot.domain.models import Strategy


class ListStrategies:
    def __init__(self, *, catalog: StrategyCatalogPort) -> None:
        self._catalog = catalog

    async def execute(self) -> Sequence[Strategy]:
        return await self._catalog.list_all()
