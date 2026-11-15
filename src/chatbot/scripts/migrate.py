"""Migrador de esquema: `python -m chatbot.scripts.migrate [--dir migrations]`.

Aplica en orden los archivos `NNN_nombre.sql` que falten y los registra en `schema_migrations`.
Idempotente: volver a ejecutarlo no hace nada. Cada migración corre en UNA transacción (o todo o nada) y
bajo un bloqueo consultivo, de modo que dos migradores simultáneos no se pisan.

El marcador `{{EMBEDDING_DIM}}` de los SQL se sustituye con `EMBEDDING_DIM` de la configuración.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

from chatbot.adapters.outbound.persistence.database import Database, SqlDatabase
from chatbot.config.settings import get_settings

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^(\d{3,})_[\w\-]+\.sql$")
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
#: Clave del bloqueo consultivo de Postgres que serializa a los migradores.
ADVISORY_LOCK_KEY = 7_272_015

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY;
"""


class MigrationError(Exception):
    """Migración mal nombrada, con marcadores sin resolver o carpeta ausente."""


class Migrator:
    def __init__(
        self, database: SqlDatabase, migrations_dir: str | Path, *, embedding_dim: int = 384
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim debe ser > 0")
        self._db = database
        self._dir = Path(migrations_dir)
        self._dim = embedding_dim

    def discover(self) -> list[tuple[str, str]]:
        """(nombre, sql ya renderizado) de cada migración, en orden numérico."""
        if not self._dir.is_dir():
            raise MigrationError(f"No existe la carpeta de migraciones: {self._dir}")
        found: list[tuple[int, str, Path]] = []
        for path in self._dir.glob("*.sql"):
            match = _NAME.match(path.name)
            if match is None:
                raise MigrationError(f"Nombre de migración inválido (se espera NNN_nombre.sql): {path.name}")
            found.append((int(match.group(1)), path.name, path))
        numbers = [n for n, _, _ in found]
        if len(set(numbers)) != len(numbers):
            raise MigrationError("Hay dos migraciones con el mismo número")
        return [
            (name, self._render(name, path.read_text(encoding="utf-8"))) for _, name, path in sorted(found)
        ]

    def _render(self, name: str, sql: str) -> str:
        values = {"EMBEDDING_DIM": str(self._dim)}

        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise MigrationError(f"{name}: marcador desconocido {{{{{key}}}}}")
            return values[key]

        return _PLACEHOLDER.sub(substitute, sql)

    async def run(self) -> list[str]:
        """Aplica las migraciones pendientes y devuelve sus nombres."""
        pending = self.discover()
        async with self._db.transaction() as tx:
            await tx.execute("SELECT pg_advisory_xact_lock(:k)", {"k": ADVISORY_LOCK_KEY})
            await tx.execute_script(_BOOTSTRAP)
        applied: list[str] = []
        for name, sql in pending:
            async with self._db.transaction() as tx:
                await tx.execute("SELECT pg_advisory_xact_lock(:k)", {"k": ADVISORY_LOCK_KEY})
                done = await tx.fetch_one(
                    "SELECT 1 AS ok FROM schema_migrations WHERE name = :n", {"n": name}
                )
                if done is not None:
                    continue
                await tx.execute_script(sql)
                await tx.execute("INSERT INTO schema_migrations (name) VALUES (:n)", {"n": name})
            applied.append(name)
            logger.info("migracion_aplicada", extra={"migracion": name})
        return applied


async def _run(directory: str) -> int:
    settings = get_settings()
    if not settings.database_url.strip():
        print("DATABASE_URL está vacío: no hay a qué base de datos migrar.", file=sys.stderr)
        return 2
    database = Database(settings.database_url, pool_size=settings.db_pool_size)
    try:
        applied = await Migrator(database, directory, embedding_dim=settings.embedding_dim).run()
    except MigrationError as exc:
        print(f"Error de migración: {exc}", file=sys.stderr)
        return 1
    finally:
        await database.close()
    print("Migraciones aplicadas: " + ", ".join(applied) if applied else "Sin migraciones pendientes.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aplica las migraciones SQL pendientes.")
    parser.add_argument("--dir", default="migrations", help="carpeta con los archivos NNN_nombre.sql")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.dir))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
