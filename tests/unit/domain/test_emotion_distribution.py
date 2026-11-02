import math

import pytest

from chatbot.domain.errors import InvalidEmotionDistribution
from chatbot.domain.models import EMOTION_ORDER, Emotion, EmotionDistribution


def test_valida_suma_uno(miedo_alto):
    assert sum(miedo_alto.probabilities.values()) == pytest.approx(1.0)


def test_rechaza_suma_distinta_de_uno():
    with pytest.raises(InvalidEmotionDistribution):
        EmotionDistribution({Emotion.MIEDO: 0.5, Emotion.IRA: 0.2})


def test_acepta_suma_dentro_de_tolerancia():
    EmotionDistribution({Emotion.MIEDO: 0.5004, Emotion.IRA: 0.5})


@pytest.mark.parametrize("valor", [-0.1, float("nan"), 1.5])
def test_rechaza_probabilidades_invalidas(valor):
    with pytest.raises(InvalidEmotionDistribution):
        EmotionDistribution({Emotion.MIEDO: valor, Emotion.IRA: 1.0})


def test_completa_emociones_ausentes_con_cero():
    d = EmotionDistribution({Emotion.ALEGRIA: 1.0})
    assert set(d.probabilities) == set(EMOTION_ORDER)
    assert d[Emotion.IRA] == 0.0


def test_es_inmutable(miedo_alto):
    with pytest.raises(TypeError):
        miedo_alto.probabilities[Emotion.MIEDO] = 0.0  # type: ignore[index]


def test_top_y_top_k(miedo_alto):
    assert miedo_alto.top() is Emotion.MIEDO
    assert miedo_alto.top_k(3)[0] is Emotion.MIEDO
    assert miedo_alto.top_k(2) == [Emotion.MIEDO, Emotion.TRISTEZA]  # empate 0.10 -> orden fijo


def test_top_k_invalido(miedo_alto):
    with pytest.raises(ValueError):
        miedo_alto.top_k(0)


def test_empate_es_determinista():
    d = EmotionDistribution.uniform()
    assert d.top() is EMOTION_ORDER[0]


def test_entropia_certeza_y_uniforme():
    assert EmotionDistribution.certain(Emotion.IRA).entropy() == 0.0
    uniforme = EmotionDistribution.uniform()
    assert uniforme.entropy() == pytest.approx(math.log2(7))
    assert uniforme.normalized_entropy() == pytest.approx(1.0)


def test_entropia_crece_con_la_incertidumbre(miedo_alto):
    assert 0 < miedo_alto.entropy() < EmotionDistribution.uniform().entropy()


def test_from_scores_normaliza():
    d = EmotionDistribution.from_scores({Emotion.IRA: 3.0, Emotion.MIEDO: 1.0})
    assert d[Emotion.IRA] == pytest.approx(0.75)


@pytest.mark.parametrize("scores", [{}, {Emotion.IRA: 0.0}, {Emotion.IRA: -1.0, Emotion.MIEDO: 2}])
def test_from_scores_invalido(scores):
    with pytest.raises(InvalidEmotionDistribution):
        EmotionDistribution.from_scores(scores)


def test_to_dict_usa_valores_en_espanol(miedo_alto):
    d = miedo_alto.to_dict()
    assert d["miedo"] == pytest.approx(0.70)
    assert list(d) == ["alegria", "tristeza", "miedo", "ira", "asco", "sorpresa", "neutral"]
