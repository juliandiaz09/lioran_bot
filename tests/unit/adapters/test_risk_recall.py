"""Recall del léxico de riesgo versionado sobre `tests/data/risk_phrases.yaml` (meta: HIGH ≥ 0,95)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from chatbot.adapters.outbound.risk.rules_risk_detector import RulesRiskDetector
from chatbot.domain.models import RiskLevel

RAIZ = Path(__file__).resolve().parents[3]
LEXICO = RAIZ / "config" / "risk_lexicon.yaml"
FRASES = RAIZ / "tests" / "data" / "risk_phrases.yaml"


@pytest.fixture(scope="module")
def detector() -> RulesRiskDetector:
    return RulesRiskDetector.from_file(LEXICO)


@pytest.fixture(scope="module")
def frases() -> list[dict[str, Any]]:
    return list(yaml.safe_load(FRASES.read_text(encoding="utf-8"))["frases"])


def test_conjunto_tiene_al_menos_30_frases_y_los_grupos_pedidos(frases: list[dict[str, Any]]) -> None:
    assert len(frases) >= 30
    grupos = {f["grupo"] for f in frases}
    assert {"directa", "indirecta", "ortografia", "medio", "bajo", "neutra", "idiomatica"} <= grupos


def test_recall_de_high_es_al_menos_095(
    detector: RulesRiskDetector, frases: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    high = [f for f in frases if f["nivel"] == "HIGH"]
    pred = {f["texto"]: detector.detect(f["texto"]).level for f in frases}
    aciertos = [f for f in high if pred[f["texto"]] is RiskLevel.HIGH]
    recall = len(aciertos) / len(high)

    # Informativo: precisión de HIGH y matriz esperado → obtenido.
    predichos_high = [t for t, nivel in pred.items() if nivel is RiskLevel.HIGH]
    precision = len(aciertos) / len(predichos_high) if predichos_high else 0.0
    matriz = Counter((f["nivel"], pred[f["texto"]].name) for f in frases)
    with capsys.disabled():
        print(
            f"\n[riesgo] frases={len(frases)} HIGH={len(high)} "
            f"recall_HIGH={recall:.3f} precision_HIGH={precision:.3f}"
        )
        for (esperado, obtenido), n in sorted(matriz.items()):
            print(f"[riesgo]   esperado={esperado:<6} obtenido={obtenido:<6} n={n}")
    fallos = [f["texto"] for f in high if pred[f["texto"]] is not RiskLevel.HIGH]
    assert recall >= 0.95, f"recall HIGH {recall:.3f} < 0,95; sin detectar: {fallos}"


def test_las_frases_no_high_no_se_clasifican_como_high(
    detector: RulesRiskDetector, frases: list[dict[str, Any]]
) -> None:
    """Informativo-estricto: las neutras, idiomáticas, LOW y MEDIUM del conjunto no deben ser HIGH."""
    falsos = [
        f["texto"] for f in frases if f["nivel"] != "HIGH" and detector.detect(f["texto"]).level.is_crisis
    ]
    assert not falsos, f"falsos positivos de HIGH: {falsos}"


def test_el_nivel_de_las_frases_medium_y_low_es_al_menos_el_esperado(
    detector: RulesRiskDetector, frases: list[dict[str, Any]]
) -> None:
    for f in frases:
        if f["nivel"] in ("LOW", "MEDIUM"):
            assert detector.detect(f["texto"]).level >= RiskLevel[f["nivel"]], f["texto"]


def test_el_lexico_oficial_declara_version_y_pendiente_de_revision(detector: RulesRiskDetector) -> None:
    datos = yaml.safe_load(LEXICO.read_text(encoding="utf-8"))
    assert detector.version == datos["version"] == "0.1.0-borrador"
    assert datos["revisado_por"] == "PENDIENTE"


def test_resultado_solo_lleva_nombres_de_categoria_nunca_texto(detector: RulesRiskDetector) -> None:
    r = detector.detect("quiero morirme, soy una carga para todos")
    assert r.level is RiskLevel.HIGH
    assert all("morirme" not in c and " " not in c for c in r.categorias)
