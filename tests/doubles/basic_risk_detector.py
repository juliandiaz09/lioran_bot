"""Detector de riesgo PROVISIONAL (léxico mínimo embebido).

Se reemplaza en la Fase 4 por `rules_risk_detector.py`, con el léxico completo en un archivo
de configuración versionado y revisado por psicología.
"""

from __future__ import annotations

from chatbot.application.dto import RiskDetection
from chatbot.domain.models import RiskLevel
from chatbot.domain.services.secondary_emotions import normalize_text

_HIGH: dict[str, tuple[str, ...]] = {
    "ideacion_suicida": (
        "suicid",
        "quitarme la vida",
        "matarme",
        "no quiero vivir",
        "no vale la pena vivir",
        "acabar con todo",
        "quiero morirme",
    ),
    "autolesion": ("hacerme dano", "cortarme", "lastimarme"),
    "violencia_hacia_otros": ("matar a", "voy a pegarle", "hacerle dano a"),
}
_MEDIUM = ("no puedo mas", "sin salida", "desesper", "no aguanto")


class BasicRiskDetector:
    def detect(self, texto: str) -> RiskDetection:
        norm = normalize_text(texto)
        found = tuple(cat for cat, terms in _HIGH.items() if any(t in norm for t in terms))
        if found:
            return RiskDetection(RiskLevel.HIGH, found)
        if any(t in norm for t in _MEDIUM):
            return RiskDetection(RiskLevel.MEDIUM, ("desesperanza",))
        return RiskDetection(RiskLevel.NONE)
