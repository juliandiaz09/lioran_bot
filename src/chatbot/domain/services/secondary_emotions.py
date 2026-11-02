"""Deriva emociones secundarias (ansiedad, estrés, frustración, soledad...) a partir de la
distribución primaria y de señales del texto. Reglas declarativas y configurables."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chatbot.domain.errors import InvalidRule
from chatbot.domain.models.emotion import Emotion, EmotionDistribution


def normalize_text(text: str) -> str:
    """Minúsculas y sin tildes (solo para comparar; el texto original no se altera)."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


@dataclass(frozen=True, slots=True)
class SecondaryRule:
    """Una emoción secundaria se activa si TODAS las probabilidades mínimas se cumplen.

    - `keywords`: señales del texto. Si aparece alguna, los umbrales se multiplican por
      `keyword_threshold_factor` (< 1 los relaja).
    - `require_keyword`: si es True, la regla no se activa sin alguna palabra clave.
    """

    name: str
    min_probs: Mapping[Emotion, float]
    keywords: tuple[str, ...] = ()
    keyword_threshold_factor: float = 1.0
    require_keyword: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvalidRule("La regla requiere nombre")
        if not self.min_probs:
            raise InvalidRule(f"La regla {self.name!r} requiere al menos un umbral")
        if any(not 0.0 <= v <= 1.0 for v in self.min_probs.values()):
            raise InvalidRule(f"Umbrales fuera de [0, 1] en la regla {self.name!r}")
        if not 0.0 < self.keyword_threshold_factor <= 1.0:
            raise InvalidRule(f"keyword_threshold_factor debe estar en (0, 1] ({self.name!r})")
        if self.require_keyword and not self.keywords:
            raise InvalidRule(f"La regla {self.name!r} exige palabras clave pero no define ninguna")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SecondaryRule:
        try:
            return cls(
                name=data["name"],
                min_probs={Emotion(k): float(v) for k, v in data["min_probs"].items()},
                keywords=tuple(data.get("keywords", ())),
                keyword_threshold_factor=float(data.get("keyword_threshold_factor", 1.0)),
                require_keyword=bool(data.get("require_keyword", False)),
            )
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise InvalidRule(f"Regla mal formada: {exc}") from exc


DEFAULT_RULES: tuple[SecondaryRule, ...] = (
    SecondaryRule(
        name="ansiedad",
        min_probs={Emotion.MIEDO: 0.40},
        keywords=(
            "ansiedad",
            "ansioso",
            "ansiosa",
            "nervios",
            "nervioso",
            "nerviosa",
            "angustia",
            "me preocupa",
            "no puedo parar de pensar",
            "taquicardia",
        ),
        keyword_threshold_factor=0.5,
    ),
    SecondaryRule(
        name="estres",
        min_probs={Emotion.MIEDO: 0.20},
        keywords=(
            "estres",
            "estresado",
            "estresada",
            "parcial",
            "parciales",
            "entrega",
            "entregas",
            "no me da el tiempo",
            "no alcanzo",
            "agobiado",
            "agobiada",
            "sobrecargado",
            "sobrecargada",
        ),
        require_keyword=True,
    ),
    SecondaryRule(
        name="frustracion",
        min_probs={Emotion.IRA: 0.35},
        keywords=(
            "no me sale",
            "no entiendo",
            "otra vez",
            "no sirve",
            "me rindo",
            "harto",
            "harta",
        ),
        keyword_threshold_factor=0.6,
    ),
    SecondaryRule(
        name="soledad",
        min_probs={Emotion.TRISTEZA: 0.30},
        keywords=("solo", "sola", "soledad", "nadie", "sin amigos", "aislado", "aislada"),
        require_keyword=True,
    ),
    SecondaryRule(
        name="desanimo",
        min_probs={Emotion.TRISTEZA: 0.40},
        keywords=("sin ganas", "desmotivado", "desmotivada", "no tengo animo", "para que"),
        keyword_threshold_factor=0.6,
    ),
)


class SecondaryEmotionMapper:
    def __init__(self, rules: Iterable[SecondaryRule] = DEFAULT_RULES, max_results: int = 3) -> None:
        self._rules = tuple(rules)
        if max_results < 1:
            raise ValueError("max_results debe ser >= 1")
        self._max = max_results
        self._patterns = {
            r.name: tuple(re.compile(rf"\b{re.escape(normalize_text(k))}\b") for k in r.keywords)
            for r in self._rules
        }

    @classmethod
    def from_dicts(cls, rules: Sequence[Mapping[str, Any]], max_results: int = 3) -> SecondaryEmotionMapper:
        """Construye el mapeador desde reglas declarativas (p. ej. un JSON versionado)."""
        return cls((SecondaryRule.from_dict(r) for r in rules), max_results)

    def derive(self, distribution: EmotionDistribution, text: str = "") -> list[str]:
        """Emociones secundarias activas, ordenadas por intensidad de la señal primaria."""
        norm = normalize_text(text)
        scored: list[tuple[float, int, str]] = []
        for index, rule in enumerate(self._rules):
            has_keyword = any(p.search(norm) for p in self._patterns[rule.name])
            if rule.require_keyword and not has_keyword:
                continue
            factor = rule.keyword_threshold_factor if has_keyword else 1.0
            if all(distribution[e] >= thr * factor for e, thr in rule.min_probs.items()):
                strength = max(distribution[e] for e in rule.min_probs)
                scored.append((-strength, index, rule.name))
        scored.sort()
        return [name for _, _, name in scored[: self._max]]
