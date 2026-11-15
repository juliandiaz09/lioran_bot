"""Integración: BD real. Se saltan solas si falta TEST_DATABASE_URL o los extras `db`.

TEST_DATABASE_URL debe apuntar a una base DESECHABLE de pruebas (ej. postgresql+asyncpg://u:p@localhost/chatbot_test).
Las pruebas aplican las migraciones (idempotente) y vacían las tablas del esquema antes de cada prueba.
"""

import os
from pathlib import Path

import pytest

from chatbot.adapters.outbound.persistence.database import Database
from chatbot.scripts.migrate import Migrator

ROOT = Path(__file__).resolve().parents[2]
TABLES = "feedback, crisis_events, analyses, messages, conversations, bandit_state, rag_chunks"


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip("TEST_DATABASE_URL no está definido")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("asyncpg")
    return url


@pytest.fixture
async def db(database_url):
    database = Database(database_url, pool_size=2)
    try:
        await database.ping()
    except Exception as exc:  # BD inalcanzable -> no es un fallo del código
        await database.close()
        pytest.skip(f"No se pudo conectar a TEST_DATABASE_URL: {type(exc).__name__}")
    await Migrator(database, ROOT / "migrations").run()
    async with database.transaction() as tx:
        await tx.execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
    yield database
    await database.close()


class ToyEmbedder:
    """Embedder determinista de 384 dims: cada palabra clave activa su propio eje (orden predecible)."""

    AXES = {"respira": 0, "piensa": 1, "ancla": 2, "contacto": 3}

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    async def embed(self, textos):
        self.calls += 1
        if self.fail:
            raise AssertionError("no debía calcular embeddings")
        out = []
        for t in textos:
            v = [0.0] * 384
            for word, axis in self.AXES.items():
                if word in t.lower():
                    v[axis] = 1.0
            if not any(v):
                v[383] = 1.0
            out.append(v)
        return out
