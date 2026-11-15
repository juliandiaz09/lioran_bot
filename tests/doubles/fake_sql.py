"""Doble de `SqlDatabase`: registra SQL y parámetros, devuelve filas programadas y simula rollback."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any


class FakeConnection:
    def __init__(self, db: FakeDatabase) -> None:
        self.db = db

    def _log(self, kind: str, sql: str, params: Any) -> None:
        if self.db.fail_on is not None and self.db.fail_on in sql:
            raise RuntimeError("fallo simulado de BD")
        self.db.current.append((kind, sql, params))

    async def fetch_all(self, sql, params=None):
        self._log("fetch_all", sql, params)
        return [dict(r) for r in self.db.rows_for(sql)]

    async def fetch_one(self, sql, params=None):
        self._log("fetch_one", sql, params)
        rows = self.db.rows_for(sql)
        return dict(rows[0]) if rows else None

    async def execute(self, sql, params=None):
        self._log("execute", sql, params)

    async def execute_script(self, sql):
        self._log("script", sql, None)


class FakeDatabase:
    def __init__(self, rows: dict[str, list[dict]] | None = None, fail_on: str | None = None) -> None:
        self.rows = rows or {}  # fragmento de SQL -> filas
        self.fail_on = fail_on
        self.vector = True
        self.committed: list[tuple[str, str, Any]] = []  # solo lo que sobrevivió a la transacción
        self.current: list[tuple[str, str, Any]] = []
        self.transactions = 0
        self.closed = False

    def rows_for(self, sql: str) -> list[dict]:
        for fragment, rows in self.rows.items():
            if fragment in sql:
                return rows
        return []

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        self.current = []
        try:
            yield FakeConnection(self)
        except BaseException:
            self.current = []  # rollback: nada de lo ejecutado queda confirmado
            raise
        else:
            self.committed.extend(self.current)

    async def ping(self):
        if self.fail_on == "ping":
            raise ConnectionError("sin BD")

    async def has_vector_extension(self) -> bool:
        return self.vector

    async def close(self):
        self.closed = True

    # utilidades de prueba
    def executed(self, fragment: str) -> list[tuple[str, Any]]:
        return [(s, p) for _, s, p in self.committed if fragment in s]
