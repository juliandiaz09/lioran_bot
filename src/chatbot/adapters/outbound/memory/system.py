from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

_BOGOTA = timezone(timedelta(hours=-5), "America/Bogota")  # Colombia no usa horario de verano


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(_BOGOTA)


class UuidGenerator:
    def new_id(self) -> str:
        return str(uuid.uuid4())
