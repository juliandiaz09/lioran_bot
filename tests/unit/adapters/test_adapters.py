import pytest

from chatbot.adapters.outbound.classifier.encoder_stub import StubEncoder
from chatbot.adapters.outbound.llm.llm_fake import FakeLLM
from chatbot.adapters.outbound.rag.chunker import ParagraphChunker
from chatbot.adapters.outbound.rag.embedder import HashEmbedder
from chatbot.adapters.outbound.validation.response_validator import RuleBasedResponseValidator
from chatbot.application.dto import RetrievedChunk
from chatbot.domain.models import Emotion, RiskLevel
from doubles.basic_risk_detector import BasicRiskDetector


@pytest.mark.parametrize(
    "texto,nivel",
    [
        ("quiero quitarme la vida", RiskLevel.HIGH),
        ("ya no quiero vivir", RiskLevel.HIGH),
        ("pienso en hacerme daño", RiskLevel.HIGH),
        ("no puedo más", RiskLevel.MEDIUM),
        ("hoy fue un buen día", RiskLevel.NONE),
    ],
)
def test_detector_basico(texto, nivel):
    assert BasicRiskDetector().detect(texto).level is nivel


async def test_encoder_stub_por_palabras_clave_y_determinista():
    enc = StubEncoder()
    a = await enc.classify("tengo miedo y nervios")
    b = await enc.classify("tengo miedo y nervios")
    assert a.top() is Emotion.MIEDO and a.to_dict() == b.to_dict()
    assert (await enc.classify("hola")).top() is Emotion.NEUTRAL


async def test_llm_fake_genera_y_registra():
    llm = FakeLLM()
    cls = await llm.classify_emotion("estoy triste", [])
    assert cls.distribucion.top() is Emotion.TRISTEZA and llm.classify_calls == ["estoy triste"]


def test_chunker_respeta_tamano_y_descarta_vacios():
    ch = ParagraphChunker(max_chars=100)
    assert ch.split("") == [] and ch.split("\n\n  \n\n") == []
    pedazos = ch.split("palabra " * 80)
    assert len(pedazos) > 1 and all(len(p) <= 100 for p in pedazos)
    with pytest.raises(ValueError):
        ParagraphChunker(max_chars=10)


async def test_embedder_normalizado_y_determinista():
    e = HashEmbedder(dim=16)
    v1, v2 = await e.embed(["hola mundo", "hola mundo"])
    assert v1 == v2 and sum(x * x for x in v1) == pytest.approx(1.0)


def test_validador_acepta_respuesta_correcta():
    v = RuleBasedResponseValidator().validate("Respira hondo. ¿Te sirvió?", [], RiskLevel.NONE)
    assert v.ok


@pytest.mark.parametrize(
    "texto",
    ["", "Respira hondo.", "Tienes depresión. ¿Ok?", "Toma ansiolíticos. ¿Ok?", "x" * 1300 + "?"],
)
def test_validador_rechaza(texto):
    assert not RuleBasedResponseValidator().validate(texto, [], RiskLevel.NONE).ok


def test_validador_permite_telefonos_que_estan_en_el_contexto():
    chunk = RetrievedChunk("d", "t", "Llama al 601 234 5678 en Bienestar.", "institucional", "d")
    v = RuleBasedResponseValidator()
    assert v.validate("Llama al 601 234 5678. ¿Te ayuda?", [chunk], RiskLevel.MEDIUM).ok
    assert not v.validate("Llama al 601 999 0000. ¿Te ayuda?", [chunk], RiskLevel.MEDIUM).ok
