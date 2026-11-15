"""Acceso a Postgres (Supabase): SQLAlchemy 2 asíncrono + asyncpg.

Los repositorios dependen del protocolo mínimo `SqlDatabase`/`SqlConnection`, no de SQLAlchemy: así se
prueban con un doble sin BD y sin los extras `db`. Solo `Database` importa SQLAlchemy, y lo hace de forma
PEREZOSA (dentro de las funciones), para que el módulo se importe aunque el extra no esté instalado.

Privacidad: `echo=False` y `hide_parameters=True` (SQLAlchemy no imprime parámetros, ni en errores).

Nota Supabase: con el *pooler* en modo transacción (puerto 6543) asyncpg no puede usar sentencias
preparadas. Se detecta el puerto 6543 y se pasa `statement_cache_size=0` (ver README).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

logger = logging.getLogger(__name__)

SUPABASE_TRANSACTION_POOLER_PORT = 6543
Params = Mapping[str, Any]


class SqlConnection(Protocol):
    """Conexión dentro de una transacción. SQL con parámetros con nombre (`:nombre`)."""

    async def fetch_all(self, sql: str, params: Params | None = None) -> list[dict[str, Any]]: ...

    async def fetch_one(self, sql: str, params: Params | None = None) -> dict[str, Any] | None: ...

    async def execute(self, sql: str, params: Params | Sequence[Params] | None = None) -> None:
        """Con una lista de mapas ejecuta la sentencia una vez por cada mapa."""
        ...

    async def execute_script(self, sql: str) -> None:
        """Varias sentencias separadas por `;` (sin parámetros). Para migraciones."""
        ...


class SqlDatabase(Protocol):
    def transaction(self) -> AbstractAsyncContextManager[SqlConnection]:
        """Todo lo ejecutado dentro del bloque confirma junto o se revierte junto."""
        ...

    async def ping(self) -> None:
        """Lanza una excepción si la BD no responde."""
        ...

    async def has_vector_extension(self) -> bool: ...

    async def close(self) -> None: ...


# ------------------------------------------------------------------------- utilidades puras (sin SQLAlchemy)
def normalize_database_url(url: str) -> str:
    """Acepta `postgresql://` y `postgres://` (formato por defecto de Supabase) y lo lleva a asyncpg."""
    url = url.strip()
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    return url


def uses_transaction_pooler(url: str) -> bool:
    try:
        return urlsplit(url).port == SUPABASE_TRANSACTION_POOLER_PORT
    except ValueError:
        return False


def engine_options(url: str, pool_size: int) -> tuple[str, dict[str, Any]]:
    """URL final y argumentos de `create_async_engine`. Pura: se prueba sin SQLAlchemy."""
    url = normalize_database_url(url)
    options: dict[str, Any] = {
        "pool_size": pool_size,
        "max_overflow": pool_size,
        "pool_pre_ping": True,
        "echo": False,
        "hide_parameters": True,
    }
    if uses_transaction_pooler(url):
        options["connect_args"] = {
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
        }
        url += ("&" if "?" in url else "?") + "prepared_statement_cache_size=0"
    return url, options


# --------------------------------------------------------------------------- implementación real
class _SaConnection:
    def __init__(self, connection: Any) -> None:
        self._conn = connection

    async def fetch_all(self, sql: str, params: Params | None = None) -> list[dict[str, Any]]:
        from sqlalchemy import text

        result = await self._conn.execute(text(sql), dict(params or {}))
        return [dict(row) for row in result.mappings().all()]

    async def fetch_one(self, sql: str, params: Params | None = None) -> dict[str, Any] | None:
        rows = await self.fetch_all(sql, params)
        return rows[0] if rows else None

    async def execute(self, sql: str, params: Params | Sequence[Params] | None = None) -> None:
        from sqlalchemy import text

        if params is None:
            await self._conn.execute(text(sql))
        elif isinstance(params, Mapping):
            await self._conn.execute(text(sql), dict(params))
        else:
            batch = [dict(p) for p in params]
            if batch:
                await self._conn.execute(text(sql), batch)

    async def execute_script(self, sql: str) -> None:
        # asyncpg no admite varias sentencias en una sentencia preparada: se usa su protocolo simple.
        # Se abre antes la transacción de SQLAlchemy para que el script quede dentro de ella.
        await self._conn.exec_driver_sql("SELECT 1")
        raw = await self._conn.get_raw_connection()
        await raw.driver_connection.execute(sql)


class Database:
    """Motor asíncrono + fábrica de transacciones. `engine` se puede inyectar (pruebas)."""

    def __init__(self, url: str, *, pool_size: int = 5, engine: Any = None) -> None:
        if engine is None:
            try:
                from sqlalchemy.ext.asyncio import create_async_engine
            except ImportError as exc:
                raise RuntimeError(
                    "PERSISTENCE_BACKEND=postgres requiere los extras de BD: pip install '.[db]'"
                ) from exc
            final_url, options = engine_options(url, pool_size)
            engine = create_async_engine(final_url, **options)
        self._engine = engine

    @property
    def engine(self) -> Any:
        return self._engine

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[SqlConnection]:
        async with self._engine.begin() as connection:
            yield _SaConnection(connection)

    async def ping(self) -> None:
        async with self.transaction() as tx:
            await tx.fetch_one("SELECT 1 AS ok")

    async def has_vector_extension(self) -> bool:
        async with self.transaction() as tx:
            row = await tx.fetch_one("SELECT 1 AS ok FROM pg_extension WHERE extname = 'vector'")
        return row is not None

    async def close(self) -> None:
        await self._engine.dispose()
