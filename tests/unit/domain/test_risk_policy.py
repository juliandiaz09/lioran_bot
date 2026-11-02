import itertools

import pytest

from chatbot.domain.models import RiskLevel
from chatbot.domain.services import RiskPolicy

N, L, M, H = RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH


def test_orden_de_niveles():
    assert N < L < M < H


@pytest.mark.parametrize("a,b,c", list(itertools.product(RiskLevel, repeat=3)))
def test_el_maximo_siempre_gana(a, b, c):
    r = RiskPolicy().assess(rules=a, llm=b, recent_levels=[c] if c else [])
    assert r.level >= max(a, b)


def test_reglas_high_prevalece_aunque_llm_diga_none():
    r = RiskPolicy().assess(rules=H, llm=N)
    assert r.level is H and r.is_crisis and r.source == "reglas"


def test_llm_puede_elevar_a_high():
    r = RiskPolicy().assess(rules=N, llm=H)
    assert r.level is H and r.source == "llm"


def test_llm_caido_no_reduce_ni_rompe():
    assert RiskPolicy().assess(rules=M, llm=None).level is M


def test_historial_eleva_un_nivel_menos():
    p = RiskPolicy(history_window=5)
    assert p.history_level([H]) is M
    assert p.history_level([M]) is L
    assert p.history_level([L]) is N
    assert p.history_level([]) is N


def test_historial_eleva_el_riesgo_del_mensaje_actual():
    r = RiskPolicy().assess(rules=N, llm=N, recent_levels=[N, H, N])
    assert r.level is M and r.source == "historial"


def test_historial_nunca_alcanza_high_por_si_solo():
    assert RiskPolicy().assess(N, N, [H, H, H]).level is M


def test_ventana_de_historial():
    p = RiskPolicy(history_window=2)
    assert p.history_level([H, N, N]) is N  # el HIGH quedó fuera de la ventana
    assert p.history_level([N, H, N]) is M


def test_source_combinada_y_ninguna():
    assert RiskPolicy().assess(M, M).source == "reglas+llm"
    assert RiskPolicy().assess(N, N).source == "reglas+llm+historial"


def test_ventana_invalida():
    with pytest.raises(ValueError):
        RiskPolicy(history_window=0)


def test_parse_nivel():
    assert RiskLevel.parse("high") is H
    assert RiskLevel.parse(1) is L
    assert RiskLevel.parse(M) is M
    with pytest.raises(ValueError):
        RiskLevel.parse("extremo")
