import pytest

from chatbot.domain.errors import InvalidFusionConfig
from chatbot.domain.models import Emotion, EmotionDistribution
from chatbot.domain.services import FusionService

ENC, LLM = "encoder", "llm"
MIEDO = EmotionDistribution.certain(Emotion.MIEDO)
IRA = EmotionDistribution.certain(Emotion.IRA)


def test_promedio_ponderado():
    r = FusionService({ENC: 0.75, LLM: 0.25}).fuse({ENC: MIEDO, LLM: IRA})
    assert r.distribution[Emotion.MIEDO] == pytest.approx(0.75)
    assert r.distribution[Emotion.IRA] == pytest.approx(0.25)
    assert r.distribution.top() is Emotion.MIEDO


def test_pesos_se_normalizan():
    a = FusionService({ENC: 1, LLM: 1}).fuse({ENC: MIEDO, LLM: IRA})
    b = FusionService({ENC: 5, LLM: 5}).fuse({ENC: MIEDO, LLM: IRA})
    assert a.distribution.to_dict() == pytest.approx(b.distribution.to_dict())


def test_desacuerdo_maximo_y_minimo():
    svc = FusionService({ENC: 0.5, LLM: 0.5})
    assert svc.fuse({ENC: MIEDO, LLM: MIEDO}).disagreement == pytest.approx(0.0)
    assert svc.fuse({ENC: MIEDO, LLM: IRA}).disagreement == pytest.approx(1.0)


def test_desacuerdo_intermedio(miedo_alto):
    r = FusionService({ENC: 0.5, LLM: 0.5}).fuse({ENC: miedo_alto, LLM: EmotionDistribution.uniform()})
    assert 0.0 < r.disagreement < 1.0


def test_un_solo_modelo_degrada_sin_fallar():
    r = FusionService({ENC: 0.5, LLM: 0.5}).fuse({ENC: MIEDO})
    assert r.distribution.top() is Emotion.MIEDO
    assert r.disagreement == 0.0
    assert r.models_used == (ENC,)


def test_solo_llm_disponible():
    r = FusionService({ENC: 0.5, LLM: 0.5}).fuse({LLM: IRA})
    assert r.distribution.top() is Emotion.IRA


def test_peso_cero_no_aporta():
    r = FusionService({ENC: 1.0, LLM: 0.0}).fuse({ENC: MIEDO, LLM: IRA})
    assert r.distribution[Emotion.IRA] == pytest.approx(0.0)
    assert r.disagreement == 0.0


@pytest.mark.parametrize("pesos", [{}, {ENC: -1, LLM: 1}, {ENC: 0, LLM: 0}, {ENC: float("nan")}])
def test_configuracion_invalida(pesos):
    with pytest.raises(InvalidFusionConfig):
        FusionService(pesos)


def test_modelo_sin_peso_o_sin_entradas():
    svc = FusionService({ENC: 1.0})
    with pytest.raises(InvalidFusionConfig):
        svc.fuse({"otro": MIEDO})
    with pytest.raises(InvalidFusionConfig):
        svc.fuse({})


def test_modelos_disponibles_con_peso_total_cero():
    with pytest.raises(InvalidFusionConfig):
        FusionService({ENC: 1.0, LLM: 0.0}).fuse({LLM: IRA})


def test_expone_pesos():
    assert FusionService({ENC: 1.0}).weights == {ENC: 1.0}
