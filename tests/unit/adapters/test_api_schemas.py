"""Esquemas Pydantic de la API: validación estricta de entrada y forma exacta de salida."""

import json

import pytest
from pydantic import ValidationError

from chatbot.adapters.inbound.api.schemas.analyze import (
    MAX_TEXT_INPUT_LENGTH,
    AnalyzeRequest,
    AnalyzeResponse,
)
from chatbot.adapters.inbound.api.schemas.strategies import StrategyListOut, StrategyOut
from chatbot.application.dto import AnalysisResult, SourceRef, StrategyRef
from chatbot.config.settings import Settings
from chatbot.domain.models import Emotion, EmotionDistribution, RiskLevel, Strategy, StrategyType


def _cuerpo_valido() -> dict[str, object]:
    return {"conversation_id": "conv-1", "user_ref": "u-seudonimo", "texto": "hoy me siento nervioso"}


# --------------------------------------------------------------------------- AnalyzeRequest
def test_request_valido_se_acepta():
    req = AnalyzeRequest.model_validate(_cuerpo_valido())
    assert (req.conversation_id, req.user_ref, req.texto) == (
        "conv-1",
        "u-seudonimo",
        "hoy me siento nervioso",
    )


def test_request_valido_desde_json_se_acepta():
    req = AnalyzeRequest.model_validate_json(json.dumps(_cuerpo_valido()))
    assert req.texto == "hoy me siento nervioso"


def test_request_rechaza_campos_extra():
    cuerpo = _cuerpo_valido() | {"nombre": "Ana"}
    with pytest.raises(ValidationError) as exc:
        AnalyzeRequest.model_validate(cuerpo)
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("campo", ["conversation_id", "user_ref", "texto"])
def test_request_rechaza_campos_faltantes(campo):
    cuerpo = _cuerpo_valido()
    del cuerpo[campo]
    with pytest.raises(ValidationError) as exc:
        AnalyzeRequest.model_validate(cuerpo)
    assert exc.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("campo", ["conversation_id", "user_ref", "texto"])
@pytest.mark.parametrize("valor", [123, 1.5, True, None, ["a"], {"a": 1}])
def test_request_rechaza_tipos_erroneos_sin_coercion(campo, valor):
    cuerpo = _cuerpo_valido() | {campo: valor}
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(cuerpo)


@pytest.mark.parametrize("campo", ["conversation_id", "user_ref", "texto"])
def test_request_desde_json_rechaza_numero_en_lugar_de_texto(campo):
    cuerpo = _cuerpo_valido() | {campo: 42}
    with pytest.raises(ValidationError) as exc:
        AnalyzeRequest.model_validate_json(json.dumps(cuerpo))
    assert exc.value.errors()[0]["type"] == "string_type"


@pytest.mark.parametrize("campo", ["conversation_id", "user_ref", "texto"])
def test_request_rechaza_cadena_vacia(campo):
    cuerpo = _cuerpo_valido() | {campo: ""}
    with pytest.raises(ValidationError) as exc:
        AnalyzeRequest.model_validate(cuerpo)
    assert exc.value.errors()[0]["type"] == "string_too_short"


@pytest.mark.parametrize("campo", ["conversation_id", "user_ref"])
def test_request_acepta_identificadores_en_los_limites(campo):
    assert AnalyzeRequest.model_validate(_cuerpo_valido() | {campo: "a"})
    assert AnalyzeRequest.model_validate(_cuerpo_valido() | {campo: "a" * 128})


@pytest.mark.parametrize("campo", ["conversation_id", "user_ref"])
def test_request_rechaza_identificadores_de_mas_de_128(campo):
    with pytest.raises(ValidationError) as exc:
        AnalyzeRequest.model_validate(_cuerpo_valido() | {campo: "a" * 129})
    assert exc.value.errors()[0]["type"] == "string_too_long"


def test_tope_del_texto_es_el_doble_del_largo_maximo_por_defecto():
    assert MAX_TEXT_INPUT_LENGTH == 4000
    assert Settings.model_fields["max_message_length"].default * 2 == MAX_TEXT_INPUT_LENGTH


def test_request_acepta_texto_en_el_tope():
    req = AnalyzeRequest.model_validate(_cuerpo_valido() | {"texto": "a" * MAX_TEXT_INPUT_LENGTH})
    assert len(req.texto) == MAX_TEXT_INPUT_LENGTH


def test_request_rechaza_texto_por_encima_del_tope():
    with pytest.raises(ValidationError) as exc:
        AnalyzeRequest.model_validate(_cuerpo_valido() | {"texto": "a" * (MAX_TEXT_INPUT_LENGTH + 1)})
    assert exc.value.errors()[0]["type"] == "string_too_long"


def test_request_rechaza_cuerpo_que_no_es_objeto():
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate_json('["conv-1", "u", "hola"]')


# --------------------------------------------------------------------------- AnalyzeResponse
def _resultado_normal() -> AnalysisResult:
    return AnalysisResult(
        trace_id="trace-123",
        emocion_dominante=Emotion.MIEDO,
        distribucion=EmotionDistribution({Emotion.MIEDO: 0.75, Emotion.TRISTEZA: 0.25}),
        emociones_secundarias=("ansiedad",),
        desacuerdo_modelos=0.125,
        riesgo=RiskLevel.LOW,
        estrategia=StrategyRef(id="respiracion_4_6", nombre="Respiración 4-6"),
        respuesta="Probemos una respiración lenta.",
        es_crisis=False,
        fuentes=(SourceRef(doc_id="doc-1", titulo="Respiración guiada"),),
        degradado=False,
        modelos_usados=("encoder", "llm"),
    )


def _resultado_crisis() -> AnalysisResult:
    return AnalysisResult(
        trace_id="trace-crisis",
        emocion_dominante=None,
        distribucion=None,
        emociones_secundarias=(),
        desacuerdo_modelos=0.0,
        riesgo=RiskLevel.HIGH,
        estrategia=None,
        respuesta="Texto de la plantilla de crisis.",
        es_crisis=True,
        fuentes=(),
    )


def test_from_result_normal_produce_la_forma_exacta():
    cuerpo = AnalyzeResponse.from_result(_resultado_normal()).model_dump(mode="json")
    assert cuerpo == {
        "trace_id": "trace-123",
        "emocion_dominante": "miedo",
        "distribucion": {
            "alegria": 0.0,
            "tristeza": 0.25,
            "miedo": 0.75,
            "ira": 0.0,
            "asco": 0.0,
            "sorpresa": 0.0,
            "neutral": 0.0,
        },
        "emociones_secundarias": ["ansiedad"],
        "desacuerdo_modelos": 0.125,
        "riesgo": "LOW",
        "estrategia": {"id": "respiracion_4_6", "nombre": "Respiración 4-6"},
        "respuesta": "Probemos una respiración lenta.",
        "es_crisis": False,
        "fuentes": [{"doc_id": "doc-1", "titulo": "Respiración guiada"}],
        "degradado": False,
        "modelos_usados": ["encoder", "llm"],
    }


def test_from_result_crisis_produce_nulos_donde_corresponde():
    cuerpo = AnalyzeResponse.from_result(_resultado_crisis()).model_dump(mode="json")
    assert cuerpo == {
        "trace_id": "trace-crisis",
        "emocion_dominante": None,
        "distribucion": None,
        "emociones_secundarias": [],
        "desacuerdo_modelos": 0.0,
        "riesgo": "HIGH",
        "estrategia": None,
        "respuesta": "Texto de la plantilla de crisis.",
        "es_crisis": True,
        "fuentes": [],
        "degradado": False,
        "modelos_usados": [],
    }


def test_from_result_serializa_a_json_sin_perder_forma():
    texto_json = AnalyzeResponse.from_result(_resultado_crisis()).model_dump_json()
    assert json.loads(texto_json)["riesgo"] == "HIGH"
    assert json.loads(texto_json)["estrategia"] is None


@pytest.mark.parametrize("nivel", list(RiskLevel))
def test_from_result_traduce_todo_nivel_de_riesgo_a_su_nombre(nivel):
    resultado = AnalysisResult(
        trace_id="t",
        emocion_dominante=Emotion.NEUTRAL,
        distribucion=EmotionDistribution.certain(Emotion.NEUTRAL),
        emociones_secundarias=(),
        desacuerdo_modelos=0.0,
        riesgo=nivel,
        estrategia=None,
        respuesta="r",
        es_crisis=nivel is RiskLevel.HIGH,
        fuentes=(),
    )
    assert AnalyzeResponse.from_result(resultado).riesgo == nivel.name


def test_respuesta_rechaza_riesgo_desconocido():
    cuerpo = AnalyzeResponse.from_result(_resultado_crisis()).model_dump() | {"riesgo": "CRITICO"}
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(cuerpo)


def test_respuesta_rechaza_campos_extra():
    cuerpo = AnalyzeResponse.from_result(_resultado_crisis()).model_dump() | {"texto_usuario": "x"}
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(cuerpo)


def test_esquema_de_respuesta_tiene_exactamente_los_campos_de_la_spec():
    assert set(AnalyzeResponse.model_fields) == {
        "trace_id",
        "emocion_dominante",
        "distribucion",
        "emociones_secundarias",
        "desacuerdo_modelos",
        "riesgo",
        "estrategia",
        "respuesta",
        "es_crisis",
        "fuentes",
        "degradado",
        "modelos_usados",
    }


@pytest.mark.parametrize("campo", ["emocion_dominante", "distribucion", "estrategia"])
def test_esquema_documenta_los_campos_nulos_en_crisis(campo):
    descripcion = AnalyzeResponse.model_fields[campo].description or ""
    assert "crisis" in descripcion and "null" in descripcion


def test_esquema_json_declara_los_niveles_de_riesgo_permitidos():
    esquema = AnalyzeResponse.model_json_schema()
    assert esquema["properties"]["riesgo"]["enum"] == ["NONE", "LOW", "MEDIUM", "HIGH"]


# --------------------------------------------------------------------------- estrategias
def _estrategia() -> Strategy:
    return Strategy(
        "respiracion_4_6",
        "Respiración 4-6",
        StrategyType.RESPIRACION,
        frozenset({Emotion.IRA, Emotion.MIEDO}),
        "Inhalar 4 segundos, exhalar 6.",
    )


def test_strategy_out_produce_la_forma_exacta():
    assert StrategyOut.from_strategy(_estrategia()).model_dump(mode="json") == {
        "id": "respiracion_4_6",
        "nombre": "Respiración 4-6",
        "tipo": "respiracion",
        "emociones_objetivo": ["miedo", "ira"],
        "descripcion": "Inhalar 4 segundos, exhalar 6.",
    }


def test_strategy_out_ordena_las_emociones_de_forma_determinista():
    estrategia = Strategy(
        "x", "X", StrategyType.GROUNDING, frozenset({Emotion.NEUTRAL, Emotion.ALEGRIA, Emotion.ASCO})
    )
    assert StrategyOut.from_strategy(estrategia).emociones_objetivo == ["alegria", "asco", "neutral"]


def test_lista_de_estrategias_conserva_el_orden_del_catalogo():
    otra = Strategy(
        "validacion_emocional",
        "Validación emocional",
        StrategyType.VALIDACION_EMOCIONAL,
        frozenset({Emotion.TRISTEZA}),
    )
    cuerpo = StrategyListOut.from_strategies([otra, _estrategia()]).model_dump(mode="json")
    assert [e["id"] for e in cuerpo["estrategias"]] == ["validacion_emocional", "respiracion_4_6"]
    assert set(cuerpo) == {"estrategias"}


def test_lista_de_estrategias_vacia():
    assert StrategyListOut.from_strategies([]).model_dump(mode="json") == {"estrategias": []}


def test_strategy_out_rechaza_campos_extra():
    cuerpo = StrategyOut.from_strategy(_estrategia()).model_dump() | {"pesos": [1]}
    with pytest.raises(ValidationError):
        StrategyOut.model_validate(cuerpo)
