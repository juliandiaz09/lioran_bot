import json

import pytest
from pydantic import ValidationError

from chatbot.application.dto import ENCODER_MODEL as ENCODER
from chatbot.application.dto import LLM_MODEL as LLM
from chatbot.config.container import build_container
from chatbot.config.settings import AppEnv, Settings


def settings(**kw):
    return Settings(_env_file=None, **kw)


def test_valores_por_defecto():
    s = settings()
    assert s.app_env is AppEnv.DEV and s.log_user_text is False and s.rag_top_k == 4


def test_lee_variables_de_entorno(monkeypatch):
    monkeypatch.setenv("FUSION_WEIGHT_ENCODER", "0.7")
    monkeypatch.setenv("FUSION_WEIGHT_LLM", "0.3")
    monkeypatch.setenv("BANDIT_SEED", "42")
    s = Settings(_env_file=None)
    assert (s.fusion_weight_encoder, s.fusion_weight_llm, s.bandit_seed) == (0.7, 0.3, 42)


def test_pesos_de_fusion_en_cero_son_invalidos():
    with pytest.raises(ValidationError):
        settings(fusion_weight_encoder=0, fusion_weight_llm=0)


def test_prod_exige_llaves():
    with pytest.raises(ValidationError):
        settings(app_env="prod")
    settings(app_env="prod", service_api_key="a", admin_api_key="b")


def test_no_filtra_secretos_en_repr():
    assert "super-secreta" not in repr(settings(service_api_key="super-secreta"))


def test_contenedor_compone_servicios_con_la_configuracion():
    c = build_container(settings(fusion_weight_encoder=0.8, fusion_weight_llm=0.2, risk_history_window=3))
    assert c.fusion_service.weights == {ENCODER: 0.8, LLM: 0.2}
    assert c.readiness()["dominio"] == "ok"


def test_contenedor_carga_reglas_secundarias_desde_archivo(tmp_path):
    f = tmp_path / "reglas.json"
    f.write_text(json.dumps([{"name": "culpa", "min_probs": {"tristeza": 0.3}}]))
    c = build_container(settings(secondary_emotions_file=str(f)))
    from chatbot.domain.models import Emotion, EmotionDistribution

    assert c.secondary_mapper.derive(EmotionDistribution.certain(Emotion.TRISTEZA)) == ["culpa"]
