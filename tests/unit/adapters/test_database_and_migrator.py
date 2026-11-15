import pytest

from chatbot.adapters.outbound.persistence.database import (
    Database,
    engine_options,
    normalize_database_url,
    uses_transaction_pooler,
)
from chatbot.scripts.migrate import ADVISORY_LOCK_KEY, MigrationError, Migrator
from doubles.fake_sql import FakeDatabase


# ------------------------------------------------------------------ database.py (funciones puras)
@pytest.mark.parametrize(
    "url, esperado",
    [
        ("postgresql://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgres://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        ("postgresql+asyncpg://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
    ],
)
def test_normaliza_la_url_a_asyncpg(url, esperado):
    assert normalize_database_url(url) == esperado


@pytest.mark.parametrize(
    "url, pooler",
    [
        ("postgresql+asyncpg://u:p@aws-0.pooler.supabase.com:6543/postgres", True),
        ("postgresql+asyncpg://u:p@db.x.supabase.co:5432/postgres", False),
        ("postgresql+asyncpg://u:p@h/db", False),
        ("no es una url", False),
    ],
)
def test_detecta_el_pooler_en_modo_transaccion_por_el_puerto_6543(url, pooler):
    assert uses_transaction_pooler(url) is pooler


def test_pooler_desactiva_las_sentencias_preparadas():
    url, opts = engine_options("postgresql://u:p@h:6543/postgres", 7)
    assert url.startswith("postgresql+asyncpg://") and url.endswith("?prepared_statement_cache_size=0")
    assert opts["connect_args"]["statement_cache_size"] == 0
    assert opts["connect_args"]["prepared_statement_name_func"]().startswith("__asyncpg_")
    assert opts["pool_size"] == 7


def test_conexion_directa_no_toca_connect_args_y_nunca_imprime_parametros():
    url, opts = engine_options("postgresql+asyncpg://u:p@h:5432/db?ssl=require", 5)
    assert "connect_args" not in opts and url.endswith("?ssl=require")
    assert opts["echo"] is False and opts["hide_parameters"] is True and opts["pool_pre_ping"] is True


def test_el_pooler_con_parametros_previos_usa_ampersand():
    url, _ = engine_options("postgresql+asyncpg://u:p@h:6543/db?ssl=require", 5)
    assert url.endswith("?ssl=require&prepared_statement_cache_size=0")


async def test_database_con_motor_inyectado_hace_ping_y_cierra():
    class Result:
        def mappings(self):
            return self

        def all(self):
            return [{"ok": 1}]

    class Conn:
        async def execute(self, *_a, **_k):
            return Result()

    class Begin:
        async def __aenter__(self):
            return Conn()

        async def __aexit__(self, *a):
            return False

    class Engine:
        disposed = False

        def begin(self):
            return Begin()

        async def dispose(self):
            self.disposed = True

    pytest.importorskip("sqlalchemy")  # `text()` lo aporta SQLAlchemy
    engine = Engine()
    db = Database("postgresql://x", engine=engine)
    await db.ping()
    assert await db.has_vector_extension()
    await db.close()
    assert engine.disposed


def test_sin_los_extras_db_el_error_indica_como_instalarlos(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sqlalchemy.ext.asyncio", None)
    with pytest.raises(RuntimeError, match=r"\.\[db\]"):
        Database("postgresql://x")


# ------------------------------------------------------------------ migrador
def write(dirpath, name, sql="SELECT 1;"):
    (dirpath / name).write_text(sql, encoding="utf-8")


def scripts(db):
    return [(s, p) for k, s, p in db.committed if k == "script"]


async def test_aplica_en_orden_numerico_y_registra_cada_una(tmp_path):
    write(tmp_path, "002_b.sql", "-- dos")
    write(tmp_path, "001_a.sql", "-- uno")
    write(tmp_path, "010_c.sql", "-- diez")
    db = FakeDatabase()
    applied = await Migrator(db, tmp_path).run()
    assert applied == ["001_a.sql", "002_b.sql", "010_c.sql"]
    assert [s for s, _ in scripts(db)][1:] == ["-- uno", "-- dos", "-- diez"]  # [0] es el bootstrap
    registros = [p["n"] for _, s, p in db.committed if "INSERT INTO schema_migrations" in s]
    assert registros == applied


async def test_idempotente_no_reaplica_lo_ya_registrado(tmp_path):
    write(tmp_path, "001_a.sql", "-- uno")
    write(tmp_path, "002_b.sql", "-- dos")
    db = FakeDatabase({"FROM schema_migrations WHERE name": [{"ok": 1}]})  # todo "ya aplicado"
    assert await Migrator(db, tmp_path).run() == []
    assert not any("INSERT INTO schema_migrations" in s for _, s, _ in db.committed)


async def test_crea_schema_migrations_con_rls_y_toma_el_bloqueo(tmp_path):
    write(tmp_path, "001_a.sql")
    db = FakeDatabase()
    await Migrator(db, tmp_path).run()
    bootstrap = scripts(db)[0][0]
    assert (
        "CREATE TABLE IF NOT EXISTS schema_migrations" in bootstrap
        and "ENABLE ROW LEVEL SECURITY" in bootstrap
    )
    locks = [p for _, s, p in db.committed if "pg_advisory_xact_lock" in s]
    assert locks and all(p == {"k": ADVISORY_LOCK_KEY} for p in locks)


async def test_cada_migracion_es_una_transaccion_y_si_falla_no_se_registra(tmp_path):
    write(tmp_path, "001_a.sql", "-- uno")
    write(tmp_path, "002_b.sql", "ROMPE")
    db = FakeDatabase(fail_on="ROMPE")  # la 2ª migración falla al ejecutar su script
    with pytest.raises(RuntimeError):
        await Migrator(db, tmp_path).run()
    registros = [p["n"] for _, s, p in db.committed if "INSERT INTO schema_migrations" in s]
    assert registros == ["001_a.sql"]  # la 001 quedó; la 002 se revirtió completa


async def test_sustituye_embedding_dim(tmp_path):
    write(tmp_path, "001_a.sql", "vector({{EMBEDDING_DIM}}) y vector({{ EMBEDDING_DIM }})")
    db = FakeDatabase()
    await Migrator(db, tmp_path, embedding_dim=768).run()
    assert scripts(db)[1][0] == "vector(768) y vector(768)"


def test_el_001_real_usa_el_marcador_y_habilita_rls_en_todas_las_tablas():
    from pathlib import Path

    sql = (Path(__file__).resolve().parents[3] / "migrations" / "001_init.sql").read_text(encoding="utf-8")
    assert "vector({{EMBEDDING_DIM}})" in sql and "USING hnsw" in sql and "vector_cosine_ops" in sql
    tablas = [ln.split()[2] for ln in sql.splitlines() if ln.startswith("CREATE TABLE")]
    assert sorted(tablas) == sorted(
        ["conversations", "messages", "analyses", "feedback", "crisis_events", "bandit_state", "rag_chunks"]
    )
    for t in tablas:
        assert f"ALTER TABLE {t}" in sql.replace("  ", " ") or f"ALTER TABLE {t} " in sql
    assert sql.count("ENABLE ROW LEVEL SECURITY") == len(tablas)


@pytest.mark.parametrize(
    "nombre, sql, fragmento",
    [
        ("malo.sql", "x", "Nombre de migración inválido"),
        ("1_a.sql", "x", "Nombre de migración inválido"),
        ("001_a.sql", "{{OTRA}}", "marcador desconocido"),
    ],
)
async def test_errores_de_migracion(tmp_path, nombre, sql, fragmento):
    write(tmp_path, nombre, sql)
    with pytest.raises(MigrationError, match=fragmento):
        await Migrator(FakeDatabase(), tmp_path).run()


async def test_numeros_repetidos_y_carpeta_ausente(tmp_path):
    write(tmp_path, "001_a.sql")
    write(tmp_path, "001_b.sql")
    with pytest.raises(MigrationError, match="mismo número"):
        await Migrator(FakeDatabase(), tmp_path).run()
    with pytest.raises(MigrationError, match="No existe la carpeta"):
        await Migrator(FakeDatabase(), tmp_path / "nada").run()


def test_embedding_dim_invalido():
    with pytest.raises(ValueError):
        Migrator(FakeDatabase(), ".", embedding_dim=0)
