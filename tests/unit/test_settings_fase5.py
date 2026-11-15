"""Fase 5: validación de combinaciones incoherentes de adaptadores (sección 3.3 de la guía)."""

import pytest
from pydantic import ValidationError

from chatbot.config.settings import Settings


def settings(**kw):
    return Settings(_env_file=None, **kw)


def test_valores_por_defecto_de_la_fase_5():
    s = settings()
    assert (s.persistence_backend, s.retriever_backend) == ("memory", "memory")
    assert s.store_user_text is True and s.llm_timeout_s == 20.0 and s.llm_max_retries == 2
    assert (s.encoder_device, s.encoder_model_path) == ("cpu", "models/encoder")
    assert (s.embedding_dim, s.db_pool_size) == (384, 5)
    assert (s.prompts_dir, s.strategies_file, s.test_database_url) == (
        "prompts",
        "config/strategies.yaml",
        "",
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"persistence_backend": "postgres"},  # 1. postgres sin DATABASE_URL
        {"persistence_backend": "postgres", "database_url": "   "},
        {"retriever_backend": "pgvector", "persistence_backend": "postgres"},  # 1. pgvector sin URL
        {"llm_backend": "openai_compatible", "llm_model": "m"},  # 2. falta base_url
        {"llm_backend": "openai_compatible", "llm_base_url": "http://x"},  # 2. falta modelo
        {"encoder_backend": "hf", "encoder_model_path": ""},  # 3. hf sin ruta
        {"retriever_backend": "pgvector", "database_url": "postgresql+asyncpg://x"},  # 4. pgvector+memory
        {"embedding_dim": 0},  # 5.
        {"embedding_dim": -3},
        {"llm_timeout_s": 0},
        {"llm_max_retries": -1},
        {"db_pool_size": 0},
    ],
)
def test_combinaciones_incoherentes_fallan_al_arrancar(kw):
    with pytest.raises(ValidationError):
        settings(**kw)


def test_pgvector_con_memoria_da_un_mensaje_claro():
    with pytest.raises(ValidationError, match="pgvector"):
        settings(retriever_backend="pgvector", database_url="postgresql+asyncpg://x")


@pytest.mark.parametrize(
    "kw",
    [
        {"persistence_backend": "postgres", "database_url": "postgresql+asyncpg://u:p@h/db"},
        {
            "persistence_backend": "postgres",
            "retriever_backend": "pgvector",
            "database_url": "postgresql+asyncpg://u:p@h/db",
        },
        {"llm_backend": "openai_compatible", "llm_base_url": "http://x/v1", "llm_model": "m"},
        {"encoder_backend": "hf"},
    ],
)
def test_combinaciones_validas(kw):
    settings(**kw)


def test_no_filtra_la_llave_del_llm_en_repr():
    assert "sk-secreta" not in repr(settings(llm_api_key="sk-secreta"))


def test_lee_las_variables_nuevas_del_entorno(monkeypatch):
    monkeypatch.setenv("STORE_USER_TEXT", "false")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    s = Settings(_env_file=None)
    assert (s.store_user_text, s.embedding_dim, s.llm_max_retries) == (False, 768, 0)
