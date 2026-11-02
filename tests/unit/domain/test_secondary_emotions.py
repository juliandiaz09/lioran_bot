import pytest

from chatbot.domain.errors import InvalidRule
from chatbot.domain.models import Emotion, EmotionDistribution
from chatbot.domain.services import SecondaryEmotionMapper, SecondaryRule
from chatbot.domain.services.secondary_emotions import normalize_text


def dist(**kw: float) -> EmotionDistribution:
    rest = 1.0 - sum(kw.values())
    scores = {Emotion(k): v for k, v in kw.items()}
    scores[Emotion.NEUTRAL] = scores.get(Emotion.NEUTRAL, 0) + rest
    return EmotionDistribution(scores)


MAPPER = SecondaryEmotionMapper()


def test_ansiedad_por_miedo_alto(miedo_alto):
    assert "ansiedad" in MAPPER.derive(miedo_alto)


def test_palabra_clave_relaja_umbral():
    d = dist(miedo=0.25)
    assert "ansiedad" not in MAPPER.derive(d, "me tiene mal")
    assert "ansiedad" in MAPPER.derive(d, "tengo muchos NERVIOS")


def test_estres_requiere_palabra_clave():
    d = dist(miedo=0.30)
    assert "estres" not in MAPPER.derive(d, "me siento mal")
    assert "estres" in MAPPER.derive(d, "estoy estresado por los parciales")


def test_soledad_requiere_palabra_clave():
    d = dist(tristeza=0.6)
    assert "soledad" not in MAPPER.derive(d, "todo me da igual")
    assert "soledad" in MAPPER.derive(d, "me siento muy sola, nadie me escribe")


def test_frustracion():
    assert "frustracion" in MAPPER.derive(dist(ira=0.5), "no me sale nada")
    assert "frustracion" in MAPPER.derive(dist(ira=0.25), "otra vez no me sale")


def test_ignora_tildes_y_mayusculas():
    assert normalize_text("ESTRÉS Y ÁNIMO") == "estres y animo"
    assert "estres" in MAPPER.derive(dist(miedo=0.3), "ESTRÉS total")


def test_palabra_completa_no_subcadena():
    # "solo" no debe activarse dentro de "solomillo"
    assert "soledad" not in MAPPER.derive(dist(tristeza=0.6), "comí solomillo")


def test_sin_senales_no_deriva_nada():
    assert MAPPER.derive(EmotionDistribution.certain(Emotion.ALEGRIA), "qué buen día") == []


def test_ordena_por_intensidad_y_limita_resultados():
    d = dist(miedo=0.5, ira=0.4, tristeza=0.1)
    texto = "estoy estresado, no me sale, otra vez"
    todas = SecondaryEmotionMapper(max_results=5).derive(d, texto)
    assert todas[0] == "ansiedad"
    assert len(SecondaryEmotionMapper(max_results=1).derive(d, texto)) == 1


def test_reglas_desde_diccionarios():
    m = SecondaryEmotionMapper.from_dicts(
        [
            {
                "name": "culpa",
                "min_probs": {"tristeza": 0.3},
                "keywords": ["culpa"],
                "require_keyword": True,
            }
        ]
    )
    assert m.derive(dist(tristeza=0.5), "siento culpa") == ["culpa"]
    assert m.derive(dist(tristeza=0.5), "siento algo") == []


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"name": "x", "min_probs": {"inexistente": 0.3}},
        {"name": "x", "min_probs": {"miedo": 2.0}},
        {"name": "x", "min_probs": {"miedo": 0.3}, "keyword_threshold_factor": 0},
        {"name": "x", "min_probs": {"miedo": 0.3}, "require_keyword": True},
        {"name": " ", "min_probs": {"miedo": 0.3}},
        {"name": "x", "min_probs": {}},
    ],
)
def test_reglas_invalidas(data):
    with pytest.raises(InvalidRule):
        SecondaryRule.from_dict(data)


def test_max_results_invalido():
    with pytest.raises(ValueError):
        SecondaryEmotionMapper(max_results=0)
