from datetime import UTC, datetime

import pytest

from chatbot.domain.errors import InvalidConversation, InvalidFeedback, InvalidStrategy
from chatbot.domain.models import (
    Conversation,
    Emotion,
    Feedback,
    Message,
    Role,
    Strategy,
    StrategyType,
)

NOW = datetime(2026, 10, 26, tzinfo=UTC)


def msg(i: str = "m1", conv: str = "c1", texto: str = "hola") -> Message:
    return Message(id=i, conversation_id=conv, texto=texto, rol=Role.USER, timestamp=NOW)


def test_estrategia_valida_y_applies_to():
    s = Strategy("resp", "Respiración 4-6", StrategyType.RESPIRACION, frozenset({Emotion.MIEDO}))
    assert s.applies_to(Emotion.MIEDO) and not s.applies_to(Emotion.IRA)


@pytest.mark.parametrize(
    "sid,nombre,emos", [("", "x", {Emotion.IRA}), ("a", " ", {Emotion.IRA}), ("a", "x", set())]
)
def test_estrategia_invalida(sid, nombre, emos):
    with pytest.raises(InvalidStrategy):
        Strategy(sid, nombre, StrategyType.GROUNDING, frozenset(emos))


def test_conversacion_agrega_y_recorta_mensajes():
    c = Conversation(id="c1", user_ref="u-abc123")
    for i in range(5):
        c.add_message(msg(f"m{i}"))
    assert [m.id for m in c.recent(2)] == ["m3", "m4"]


def test_conversacion_rechaza_mensaje_ajeno_y_user_ref_invalido():
    c = Conversation(id="c1", user_ref="u-abc123")
    with pytest.raises(InvalidConversation):
        c.add_message(msg(conv="otra"))
    for bad in ("", "  ", "x" * 200):
        with pytest.raises(InvalidConversation):
            Conversation(id="c1", user_ref=bad)
    with pytest.raises(InvalidConversation):
        Conversation(id="", user_ref="u")
    with pytest.raises(ValueError):
        c.recent(0)


def test_mensaje_invalido():
    with pytest.raises(InvalidConversation):
        msg(texto="   ")
    with pytest.raises(InvalidConversation):
        msg(i="")


@pytest.mark.parametrize("util,reward", [(True, 1.0), (False, 0.0), (1, 0.0), (3, 0.5), (5, 1.0)])
def test_feedback_recompensa(util, reward):
    assert Feedback("m", "s", util).reward == pytest.approx(reward)


@pytest.mark.parametrize("util", [0, 6, -1, 2.5, "si", None])
def test_feedback_util_invalido(util):
    with pytest.raises(InvalidFeedback):
        Feedback("m", "s", util)  # type: ignore[arg-type]


def test_feedback_otros_invalidos():
    with pytest.raises(InvalidFeedback):
        Feedback("", "s", True)
    with pytest.raises(InvalidFeedback):
        Feedback("m", "s", True, "x" * 1001)
