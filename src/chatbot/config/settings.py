"""Configuración por entorno (pydantic-settings). Sin secretos en el repositorio."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    app_env: AppEnv = AppEnv.DEV
    log_level: str = "INFO"
    log_user_text: bool = False

    # Seguridad servicio a servicio
    service_api_key: SecretStr = SecretStr("")
    admin_api_key: SecretStr = SecretStr("")

    # Fusión tardía
    fusion_weight_encoder: float = Field(0.5, ge=0)
    fusion_weight_llm: float = Field(0.5, ge=0)

    # Riesgo, bandido, RAG
    risk_history_window: int = Field(5, ge=1)
    bandit_min_observations: int = Field(20, ge=0)
    bandit_seed: int | None = None
    rag_top_k: int = Field(4, ge=1)

    # Tiempos límite (segundos) y límites del caso de uso principal
    classification_timeout_s: float = Field(6.0, gt=0)
    generation_timeout_s: float = Field(8.0, gt=0)
    persistence_timeout_s: float = Field(3.0, gt=0)
    crisis_lookup_timeout_s: float = Field(1.0, gt=0)
    max_message_length: int = Field(2000, ge=50)
    max_regenerations: int = Field(1, ge=0)

    # Configuración declarativa
    secondary_emotions_file: str | None = None
    risk_lexicon_file: str = "config/risk_lexicon.yaml"
    templates_dir: str = "config/templates"

    # Adaptadores (Fase 3+)
    encoder_backend: Literal["stub", "hf"] = "stub"
    #: Solo informativo (logs). El modelo se carga SIEMPRE desde `encoder_model_path`, sin red.
    encoder_model_name: str = "pysentimiento/robertuito-emotion-analysis"
    encoder_temperature_file: str = "config/encoder_temperature.json"
    llm_backend: Literal["fake", "openai_compatible"] = "fake"
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    database_url: str = ""
    supabase_url: str = ""
    supabase_service_key: SecretStr = SecretStr("")
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Adaptadores reales (Fase 5)
    persistence_backend: Literal["memory", "postgres"] = "memory"
    retriever_backend: Literal["memory", "pgvector"] = "memory"
    store_user_text: bool = True
    llm_timeout_s: float = Field(20.0, gt=0)
    llm_max_retries: int = Field(2, ge=0)
    encoder_device: str = "cpu"
    encoder_model_path: str = "models/encoder"
    embedding_dim: int = Field(384, gt=0)
    db_pool_size: int = Field(5, ge=1)
    prompts_dir: str = "prompts"
    strategies_file: str = "config/strategies.yaml"
    test_database_url: str = ""

    @model_validator(mode="after")
    def _check_fusion_weights(self) -> Settings:
        if self.fusion_weight_encoder + self.fusion_weight_llm <= 0:
            raise ValueError("La suma de los pesos de fusión debe ser mayor que 0")
        return self

    @model_validator(mode="after")
    def _check_backend_combinations(self) -> Settings:
        """Combinaciones incoherentes de adaptadores: el servicio no arranca (fail-fast)."""
        if (
            self.persistence_backend == "postgres" or self.retriever_backend == "pgvector"
        ) and not self.database_url.strip():
            raise ValueError(
                "PERSISTENCE_BACKEND=postgres o RETRIEVER_BACKEND=pgvector requieren DATABASE_URL"
            )
        if self.llm_backend == "openai_compatible" and not (
            self.llm_base_url.strip() and self.llm_model.strip()
        ):
            raise ValueError("LLM_BACKEND=openai_compatible requiere LLM_BASE_URL y LLM_MODEL")
        if self.encoder_backend == "hf" and not self.encoder_model_path.strip():
            raise ValueError("ENCODER_BACKEND=hf requiere ENCODER_MODEL_PATH")
        if self.retriever_backend == "pgvector" and self.persistence_backend == "memory":
            # Decisión de la Fase 5: combinación no soportada, se rechaza por simplicidad.
            raise ValueError("RETRIEVER_BACKEND=pgvector no se admite con PERSISTENCE_BACKEND=memory")
        return self

    @model_validator(mode="after")
    def _require_keys_in_prod(self) -> Settings:
        if self.app_env is AppEnv.PROD and not (
            self.service_api_key.get_secret_value() and self.admin_api_key.get_secret_value()
        ):
            raise ValueError("En prod SERVICE_API_KEY y ADMIN_API_KEY son obligatorias")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
