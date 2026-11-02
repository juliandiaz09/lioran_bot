import pytest

from chatbot.domain.models import (
    BanditContext,
    Certainty,
    Emotion,
    EmotionDistribution,
    RiskLevel,
    TimeOfDay,
)


@pytest.mark.parametrize(
    "hour,franja",
    [
        (0, "madrugada"),
        (5, "madrugada"),
        (6, "manana"),
        (11, "manana"),
        (12, "tarde"),
        (17, "tarde"),
        (18, "noche"),
        (23, "noche"),
    ],
)
def test_franja_horaria(hour, franja):
    assert TimeOfDay.from_hour(hour).value == franja


@pytest.mark.parametrize("h", [-1, 24])
def test_hora_invalida(h):
    with pytest.raises(ValueError):
        TimeOfDay.from_hour(h)


@pytest.mark.parametrize("e,c", [(0.0, Certainty.ALTA), (0.5, Certainty.MEDIA), (0.9, Certainty.BAJA)])
def test_certeza(e, c):
    assert Certainty.from_entropy(e) is c


def test_build_y_clave_de_celda():
    ctx = BanditContext.build(EmotionDistribution.certain(Emotion.MIEDO), RiskLevel.LOW, 21, ["a"])
    assert ctx.cell_key == "miedo|alta|LOW|noche"
    assert ctx.estrategias_previas == ("a",)


def test_estrategias_previas_no_cambian_la_celda():
    d = EmotionDistribution.certain(Emotion.IRA)
    a = BanditContext.build(d, RiskLevel.NONE, 10, [])
    b = BanditContext.build(d, RiskLevel.NONE, 10, ["x", "y"])
    assert a.cell_key == b.cell_key


def test_serializacion_ida_y_vuelta():
    ctx = BanditContext.build(EmotionDistribution.uniform(), RiskLevel.MEDIUM, 3, ["z"])
    assert BanditContext.from_dict(ctx.to_dict()) == ctx
