"""Pruebas del motor del detector de riesgo por reglas (léxicos de prueba, no el definitivo)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from chatbot.adapters.outbound.risk.normalization import (
    NormalizationConfig,
    TextNormalizer,
    contiene_invisibles,
)
from chatbot.adapters.outbound.risk.rules_risk_detector import InvalidLexicon, RulesRiskDetector
from chatbot.domain.models import RiskLevel

LEXICOS = Path(__file__).resolve().parents[2] / "data" / "lexicons"
VALIDO = LEXICOS / "valido.yaml"

#: Caracteres invisibles, escritos por su nombre para que se vean en el código.
ZWSP = "\N{ZERO WIDTH SPACE}"
ZWJ = "\N{ZERO WIDTH JOINER}"
UNION = "\N{WORD JOINER}"
GUION_SUAVE = "\N{SOFT HYPHEN}"
BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"
RELLENO = "\N{HANGUL FILLER}"
SELECTOR = "\N{VARIATION SELECTOR-16}"
UNIDOR_GRAFEMAS = "\N{COMBINING GRAPHEME JOINER}"

ABREVIATURAS = NormalizationConfig(
    abreviaturas=(("q", "que"), ("xq", "porque"), ("tmb", "también"), ("nose", "no sé"))
)


@pytest.fixture(scope="module")
def detector() -> RulesRiskDetector:
    return RulesRiskDetector.from_file(VALIDO)


def escribir_lexico(tmp_path: Path, contenido: str) -> Path:
    ruta = tmp_path / "lexico.yaml"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


# ----------------------------------------------------------------------------- normalización
def test_normalizacion_pasa_a_minusculas_y_quita_tildes():
    assert TextNormalizer().normalize("ÁNIMO Más DAÑO") == "animo mas dano"


@pytest.mark.parametrize(
    "texto,esperado",
    [("muuuerto", "muerto"), ("nooo puedooo", "no puedo"), ("quiiieeero morirmeee", "quiero morirme")],
)
def test_normalizacion_colapsa_letras_repetidas(texto, esperado):
    assert TextNormalizer().normalize(texto) == esperado


def test_normalizacion_no_colapsa_digitos():
    assert TextNormalizer().normalize("semestre 2000") == "semestre 2000"


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("xq", "porque"),
        ("q hago", "que hago"),
        ("tmb", "tambien"),
        ("nose", "no se"),
        ("Nose q hacer xq tmb", "no se que hacer porque tambien"),
        ("xq?", "porque"),
        ("nose_que", "no se que"),
    ],
)
def test_normalizacion_expande_abreviaturas(texto, esperado):
    assert TextNormalizer(ABREVIATURAS).normalize(texto) == esperado


def test_normalizacion_no_expande_abreviaturas_dentro_de_palabras():
    assert TextNormalizer(ABREVIATURAS).normalize("aquel queso nosey") == "aquel queso nosey"


def test_normalizacion_limpia_puntuacion_y_espacios():
    assert TextNormalizer().normalize("  ¿¡No... puedo,   más!?  ") == "no puedo mas"


def test_normalizacion_respeta_las_banderas_desactivadas():
    config = NormalizationConfig(minusculas=False, quitar_tildes=False, colapsar_letras_repetidas=False)
    assert TextNormalizer(config).normalize("Muuuy DAÑO.") == "Muuuy DAÑO"


def test_normalizacion_de_regex_conserva_las_secuencias_de_escape():
    normalizador = TextNormalizer()
    assert normalizador.normalize_regex(r"\bMuuERTO\S+\bbasta") == r"\bmuerto\S+\bbasta"
    assert normalizador.normalize_regex(r"DAÑO\W\D\B") == r"dano\W\D\B"
    assert normalizador.normalize_regex(r"(?P<Llanto>LLORO)(?P=Llanto)") == r"(?P<lanto>loro)(?P=lanto)"
    assert normalizador.normalize_regex(r"\N{LATIN SMALL LETTER A}AA") == r"\N{LATIN SMALL LETTER A}a"


# ----------------------------------------------------------------------------- detección
def test_el_detector_carga_el_lexico_valido_y_expone_su_version(detector):
    assert detector.version == "prueba-1.0"


@pytest.mark.parametrize(
    "texto,nivel",
    [
        ("Quiero MORIRME", RiskLevel.HIGH),
        ("quisiera estar muuuuerto", RiskLevel.HIGH),
        ("pienso en el suicidio", RiskLevel.HIGH),
        ("ya no sé q hacer", RiskLevel.MEDIUM),
        ("ya no puedo más!!!", RiskLevel.MEDIUM),
        ("hoy me muero de sueño", RiskLevel.LOW),
        ("hoy fue un buen día", RiskLevel.NONE),
        ("", RiskLevel.NONE),
    ],
)
def test_el_detector_asigna_el_nivel_esperado(detector, texto, nivel):
    assert detector.detect(texto).level is nivel


def test_el_detector_devuelve_el_nivel_mas_alto_y_solo_nombres_de_categoria(detector):
    deteccion = detector.detect("quiero morirme, no puedo más y me muero")
    assert deteccion.level is RiskLevel.HIGH
    assert deteccion.categorias == ("ideacion_suicida", "desesperanza_moderada", "malestar_agudo")


def test_el_detector_sin_coincidencias_no_devuelve_categorias(detector):
    assert detector.detect("hoy fue un buen día").categorias == ()


@pytest.mark.parametrize(
    "texto",
    [
        f"quiero mori{ZWSP}rme",
        f"quiero mori{GUION_SUAVE}rme",
        f"qui{ZWJ}ero morirme",
        "ｑｕｉｅｒｏ ｍｏｒｉｒｍｅ",  # ancho completo
    ],
)
def test_los_caracteres_invisibles_o_de_ancho_completo_no_ocultan_un_high(detector, texto):
    assert detector.detect(texto).level is RiskLevel.HIGH


@pytest.mark.parametrize(
    "texto",
    [
        f"quiero{ZWSP}morirme",  # aparece al copiar y pegar desde la web o desde mensajería
        f"me{ZWSP}mato",
        f"hacerme{ZWSP}daño",
        f"quiero{UNION}morirme",
        f"quiero{GUION_SUAVE}morirme",
        f"QUIERO{ZWSP}{ZWSP}MORIRME",  # varios seguidos
        f"hoy{ZWSP}quiero{ZWSP}morirme{ZWSP}ya",  # el mensaje entero separado así
        f"{BOM}ho{GUION_SUAVE}la, quiero{ZWSP}morirme",  # uno dentro de una palabra y otro entre dos
        f"quiero{RELLENO}morirme",  # "letra" en blanco: Unicode la clasifica como letra
        f"quiero mori{RELLENO}rme",
        f"quiero{SELECTOR}morirme",  # selector de variante: es una marca, no un formato
        f"quiero{UNIDOR_GRAFEMAS}morirme",
    ],
)
def test_un_caracter_invisible_entre_dos_palabras_no_oculta_un_high(detector, texto):
    # Borrarlo pegaría las dos palabras ("quieromorirme") y la frase HIGH dejaría de coincidir.
    assert detector.detect(texto).level is RiskLevel.HIGH


@pytest.mark.parametrize("texto", ["quiero mori\x00rme", "quie\x7fro morirme", "quiero\x00morirme"])
def test_un_caracter_de_control_no_oculta_un_high(detector, texto):
    assert detector.detect(texto).level is RiskLevel.HIGH


@pytest.mark.parametrize(
    "texto,nivel",
    [
        (f"me{ZWSP}muero", RiskLevel.LOW),
        (f"me mue{GUION_SUAVE}ro", RiskLevel.LOW),
        (f"ya no{ZWSP}puedo{ZWSP}más", RiskLevel.MEDIUM),
        (f"me{ZWSP}muero{ZWSP}de{ZWSP}la{ZWSP}risa", RiskLevel.NONE),  # la excepción sigue rebajando
    ],
)
def test_un_caracter_invisible_tampoco_oculta_un_low_ni_un_medium(detector, texto, nivel):
    assert detector.detect(texto).level is nivel


def test_un_mensaje_con_caracteres_invisibles_y_sin_riesgo_no_se_vuelve_riesgo(detector):
    texto = f"hoy{ZWSP}fue un{GUION_SUAVE} buen{ZWSP} día ❤{SELECTOR} 👨{ZWJ}👩{ZWJ}👧"
    assert detector.detect(texto).level is RiskLevel.NONE


def test_normalizacion_trata_un_caracter_invisible_como_parte_de_la_palabra_o_como_espacio():
    normalizador = TextNormalizer()
    texto = f"quiero{ZWSP}mori{GUION_SUAVE}r\x00me"
    assert normalizador.normalize(texto) == "quieromorirme"
    assert normalizador.normalize(texto, invisibles_como_espacio=True) == "quiero mori r me"


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("hola", False),
        ("hola\tmundo\n", False),  # los espacios de control no son invisibles: ya separan palabras
        ("día ❤ 👨", False),
        (f"ho{ZWSP}la", True),
        ("ho\x00la", True),
        (f"{BOM}hola", True),
        (f"hola{SELECTOR}", True),
    ],
)
def test_contiene_invisibles_distingue_formato_y_control_de_los_espacios(texto, esperado):
    assert contiene_invisibles(texto) is esperado


def test_normalizacion_de_regex_quita_caracteres_de_formato_invisibles():
    assert TextNormalizer().normalize_regex(f"mori{ZWSP}rme\\b") == "morirme\\b"


def test_una_negacion_no_rebaja_un_patron_high(detector):
    deteccion = detector.detect("no pienso hacerme daño")
    assert deteccion.level is RiskLevel.HIGH
    assert deteccion.categorias == ("autolesion",)


def test_una_excepcion_idiomatica_no_suprime_un_high(detector):
    deteccion = detector.detect("me mato estudiando")
    assert deteccion.level is RiskLevel.HIGH
    assert deteccion.categorias == ("ideacion_suicida",)


def test_una_excepcion_idiomatica_rebaja_una_coincidencia_low(detector):
    assert detector.detect("jajaja me muero de la risa").level is RiskLevel.NONE


def test_una_excepcion_idiomatica_rebaja_una_coincidencia_medium(detector):
    assert detector.detect("no puedo más de la risa").level is RiskLevel.NONE


@pytest.mark.parametrize("texto", ["supermato", "teme matorrales", "me matoneaban en el colegio"])
def test_una_frase_solo_coincide_con_palabras_completas(detector, texto):
    assert detector.detect(texto).level is RiskLevel.NONE


def test_una_excepcion_idiomatica_en_forma_regex_rebaja_una_coincidencia_low(tmp_path):
    contenido = LEXICO_OK + (
        "  malestar_agudo:\n    nivel: LOW\n    patrones: [me muero]\n"
        "excepciones_idiomaticas:\n  - regex: '\\bme muero de (?:la )?risa\\b'\n"
    )
    detector = RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))
    assert detector.detect("me muero de risa").level is RiskLevel.NONE
    assert detector.detect("me muero").level is RiskLevel.LOW


def test_una_excepcion_idiomatica_no_oculta_otro_riesgo_del_mismo_mensaje(detector):
    assert detector.detect("me muero de la risa, pero no puedo más").level is RiskLevel.MEDIUM
    assert detector.detect("me muero de la risa, igual quiero morirme").level is RiskLevel.HIGH


# ----------------------------------------------------------------------------- fail-fast
def test_falla_al_arrancar_si_el_archivo_no_existe(tmp_path):
    with pytest.raises(InvalidLexicon, match="No existe"):
        RulesRiskDetector.from_file(tmp_path / "no_existe.yaml")


def test_falla_al_arrancar_si_el_yaml_esta_mal_formado():
    with pytest.raises(InvalidLexicon, match="mal formado"):
        RulesRiskDetector.from_file(LEXICOS / "mal_formado.yaml")


def test_falla_al_arrancar_si_un_patron_no_compila():
    with pytest.raises(InvalidLexicon, match="regex inválida"):
        RulesRiskDetector.from_file(LEXICOS / "patron_invalido.yaml")


def test_falla_al_arrancar_si_un_nivel_es_desconocido():
    with pytest.raises(InvalidLexicon, match="nivel desconocido"):
        RulesRiskDetector.from_file(LEXICOS / "nivel_desconocido.yaml")


CATEGORIA_OK = "categorias:\n  ideacion_suicida:\n    nivel: HIGH\n    patrones: [quiero morirme]\n"
CABECERA_OK = 'version: "1"\nrevisado_por: PENDIENTE\n'
#: Cabecera válida + una categoría a medio escribir: cada caso completa su cuerpo.
UNA_CATEGORIA = CABECERA_OK + "categorias:\n  ideacion_suicida:\n"
LEXICO_OK = CABECERA_OK + CATEGORIA_OK


@pytest.mark.parametrize(
    "contenido,mensaje",
    [
        ("- una lista\n- no un mapa\n", "mapa"),
        ("revisado_por: PENDIENTE\n" + CATEGORIA_OK, "version"),
        ('version: "1"\n' + CATEGORIA_OK, "revisado_por"),
        ("version: 1.0\nrevisado_por: PENDIENTE\n" + CATEGORIA_OK, "version"),
        ('version: "  "\nrevisado_por: PENDIENTE\n' + CATEGORIA_OK, "version"),
        (CABECERA_OK, "categorias"),
        (CABECERA_OK + "categorias: {}\n", "categorias"),
        (CABECERA_OK + "categorias:\n  ideacion_suicida: HIGH\n", "ideacion_suicida"),
        (UNA_CATEGORIA + "    nivel: HIGH\n    patrones: []\n", "patrones"),
        (UNA_CATEGORIA + "    patrones: [morir]\n", "nivel"),
        (UNA_CATEGORIA + "    nivel: NONE\n    patrones: [morir]\n", "nivel"),
        (UNA_CATEGORIA + "    nivel: 3\n    patrones: [morir]\n", "nivel"),
        (CABECERA_OK + "categorias:\n  Ideación Suicida:\n    nivel: HIGH\n    patrones: [x]\n", "nombre"),
        (UNA_CATEGORIA + "    nivel: HIGH\n    patrones:\n      - no\n", "texto"),
        (UNA_CATEGORIA + "    nivel: HIGH\n    patrones: ['¡¡!!']\n", "vacía"),
        (UNA_CATEGORIA + "    nivel: HIGH\n    patrones: [{regla: morir}]\n", "regex"),
        (LEXICO_OK + "excepciones_idiomaticas: me muero de la risa\n", "excepciones"),
        (LEXICO_OK + "normalizacion: [minusculas]\n", "normalizacion"),
        (LEXICO_OK + "normalizacion:\n  minusculas: 'si'\n", "minusculas"),
        (LEXICO_OK + "normalizacion:\n  abreviaturas: [q]\n", "abreviaturas"),
        (LEXICO_OK + "normalizacion:\n  abreviaturas:\n    '..': que\n", "abreviatura"),
        (LEXICO_OK + "normalizacion:\n  abreviaturas:\n    q: 3\n", "abreviatura"),
        (LEXICO_OK + "categorias:\n  otra:\n    nivel: LOW\n    patrones: [x]\n", "duplicada"),
    ],
)
def test_falla_al_arrancar_si_el_lexico_es_invalido(tmp_path, contenido, mensaje):
    ruta = escribir_lexico(tmp_path, contenido)
    with pytest.raises(InvalidLexicon, match=mensaje):
        RulesRiskDetector.from_file(ruta)


@pytest.mark.parametrize("bandera", ["minusculas", "quitar_tildes", "colapsar_letras_repetidas"])
def test_falla_al_arrancar_si_el_lexico_desactiva_una_bandera_de_normalizacion(tmp_path, bandera):
    # Sin ese paso, "Quiero morirme", "hacerme dano" o "morirmeee" dejarían de ser HIGH en silencio.
    ruta = escribir_lexico(tmp_path, LEXICO_OK + f"normalizacion:\n  {bandera}: false\n")
    with pytest.raises(InvalidLexicon, match=f"'normalizacion.{bandera}' debe ser true"):
        RulesRiskDetector.from_file(ruta)


def test_falla_al_arrancar_si_el_archivo_no_es_utf8(tmp_path):
    ruta = tmp_path / "lexico.yaml"
    ruta.write_bytes(LEXICO_OK.encode("utf-8") + b"# \xff\xfe\n")
    with pytest.raises(InvalidLexicon, match="No se pudo leer"):
        RulesRiskDetector.from_file(ruta)


def test_las_anclas_yaml_con_fusion_no_cuentan_como_claves_duplicadas(tmp_path):
    contenido = CABECERA_OK + (
        "base: &alto {nivel: HIGH}\n"
        "categorias:\n"
        "  ideacion_suicida:\n    <<: *alto\n    patrones: [quiero morirme]\n"
    )
    detector = RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))
    assert detector.detect("quiero morirme").level is RiskLevel.HIGH


@pytest.mark.parametrize(
    "abreviatura,regex,texto",
    [
        # Una abreviatura que reescribe una palabra que la regex HIGH necesita.
        ("suicidio: tema", r"\bsuicid\w*", "pienso en el suicidio"),
        # Una regex HIGH que contiene una abreviatura literal.
        ("xq: porque", r"\bxq\s+vivir", "xq vivir"),
    ],
)
def test_una_abreviatura_no_anula_un_patron_high_escrito_como_regex(tmp_path, abreviatura, regex, texto):
    contenido = CABECERA_OK + (
        f"normalizacion:\n  abreviaturas:\n    {abreviatura}\n"
        f"categorias:\n  ideacion_suicida:\n    nivel: HIGH\n    patrones:\n      - regex: '{regex}'\n"
    )
    detector = RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))
    assert detector.detect(texto).level is RiskLevel.HIGH


def lexico_con_regex_high(*regex: str) -> str:
    patrones = "".join(f"      - regex: '{r}'\n" for r in regex)
    return UNA_CATEGORIA + "    nivel: HIGH\n    patrones:\n" + patrones


@pytest.mark.parametrize(
    "regex",
    [r"(?:\w+\s?)+\.", r"(\w+\s?)+\b(?:morir)\b\.", r"(a*)*b", r"(?:\s+\w+){0,3}morir", r"(?>(a+)+b)"],
)
def test_falla_al_arrancar_si_una_regex_anida_cuantificadores(tmp_path, regex):
    # Retroceso catastrófico: bloquearía `detect` y con él la ruta de crisis de todos los usuarios.
    with pytest.raises(InvalidLexicon, match="cuantificadores anidados"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))


def test_falla_al_arrancar_si_una_excepcion_anida_cuantificadores(tmp_path):
    contenido = LEXICO_OK + "excepciones_idiomaticas:\n  - regex: '(?:\\w+\\s?)+\\.'\n"
    with pytest.raises(InvalidLexicon, match=r"excepcion #1: .*cuantificadores anidados"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))


@pytest.mark.parametrize("regex", ["", ".*", "x?", r"\b", "(?=morir)", "morir|"])
def test_falla_al_arrancar_si_un_patron_regex_puede_coincidir_con_un_texto_vacio(tmp_path, regex):
    # Como patrón HIGH volvería crisis cualquier mensaje.
    with pytest.raises(InvalidLexicon, match="texto vacío"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))


@pytest.mark.parametrize("excepcion", [r"{regex: '\s'}", "{regex: '.'}", "{regex: '[a-z]'}", "ay", "'¡uf!'"])
def test_falla_al_arrancar_si_una_excepcion_es_demasiado_corta(tmp_path, excepcion):
    # Una excepción de una letra o un espacio partiría todo el texto y borraría todo LOW y MEDIUM.
    contenido = LEXICO_OK + f"excepciones_idiomaticas:\n  - {excepcion}\n"
    with pytest.raises(InvalidLexicon, match="excepcion #1: .*demasiado corta"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))


def test_falla_al_arrancar_si_una_excepcion_puede_coincidir_con_un_texto_vacio(tmp_path):
    # Como excepción borraría en silencio todas las detecciones LOW y MEDIUM.
    contenido = LEXICO_OK + "excepciones_idiomaticas:\n  - regex: '.*'\n"
    with pytest.raises(InvalidLexicon, match="excepcion #1: .*texto vacío"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))


@pytest.mark.parametrize("regex", [r"mu{2,}erto", r"l{2}oro", r"(?:u){3,5}"])
def test_falla_al_arrancar_si_una_regex_exige_una_letra_repetida(tmp_path, regex):
    # El texto llega con cada racha reducida a una letra: ese patrón jamás coincidiría.
    with pytest.raises(InvalidLexicon, match="letra repetida"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))


def test_falla_al_arrancar_si_el_lexico_no_tiene_ninguna_categoria_high(tmp_path):
    contenido = CABECERA_OK + "categorias:\n  malestar_agudo:\n    nivel: LOW\n    patrones: [me muero]\n"
    with pytest.raises(InvalidLexicon, match="ninguna categoría HIGH"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))


@pytest.mark.parametrize(
    "regex",
    [
        # Las cuatro de la prueba de concepto de la revisión: se aceptaban y bloqueaban `detect`
        # durante segundos o minutos con "no quiero no quiero...", "quiero abab..." o "jajaja...".
        r".*\bno\b.*\bquiero\b.*\bvivir",  # varios ".*" encadenados: coste O(n^4)
        r"\bquiero\s*\w*\s*\w*\s*\w*\s*morir",  # cuantificadores adyacentes que se solapan
        r"\bquiero\b.*\bmorir.*\bya",
        r"\b(?:ja|jaja|jajaja)+\s+me\s+muero",  # alternativas que se solapan: coste exponencial
        # Otras formas con el mismo problema.
        r"\bno\b.*\bvivir",  # un solo ".*" ya es cuadrático con "no no no..."
        r"\w*matar\w*",  # comodín sin tope al principio
        r"\w+\s+morir",
        r"[a-z]{2,}\d{2}",
        r"(?:\w++\s?)+\.",  # ni siquiera posesiva: recorre la palabra entera en cada posición
        r"(?>\w+\s?)+\.",
        r"\bquiero(?:\s++\w++){0,3}\s+morir",
        r"\bsuicid\w*\b",  # el "\w*" solo puede ir sin tope si es lo último de la regex
        r"(?=\w*)morir",
        r"\d+ veces",
    ],
)
def test_falla_al_arrancar_si_una_regex_repite_sin_maximo(tmp_path, regex):
    with pytest.raises(InvalidLexicon, match="repetición sin máximo"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))


def test_falla_al_arrancar_si_una_excepcion_repite_sin_maximo(tmp_path):
    # Una excepción se evalúa con cada mensaje: "jajaja..." (60 letras) la bloqueaba segundos.
    contenido = LEXICO_OK + "excepciones_idiomaticas:\n  - regex: '\\b(?:ja|jaja|jajaja)+\\s+me\\s+muero'\n"
    with pytest.raises(InvalidLexicon, match=r"excepcion #1: .*repetición sin máximo"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))


@pytest.mark.parametrize(
    "regex",
    [
        r"\bno\b.{0,40}\bquiero\b.{0,40}\bvivir\b.{0,40}\bya",  # tres huecos: 41³ combinaciones
        r"\w{0,30}\w{0,30}\w{0,30}x",  # cuantificadores adyacentes, aunque tengan tope
        r"\b(?:ja|jaja|jajaja){1,30}\s+me\s+muero",  # alternativas que se solapan: 3^30
        r"(?>(?:ja|jaja){1,30})\s+me\s+muero",  # el grupo atómico retrocede por dentro
        r"morir.{0,2000}ya",
        r"morir\w{2000,}",  # al final y sin tope, pero obliga a recorrer 2 000 letras en cada posición
    ],
)
def test_falla_al_arrancar_si_una_regex_admite_demasiadas_combinaciones(tmp_path, regex):
    with pytest.raises(InvalidLexicon, match="demasiadas combinaciones"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))


@pytest.mark.parametrize("regex", [r"(\w{1,20}) \1", r"(me )?(?(1)mato|muero)"])
def test_falla_al_arrancar_si_una_regex_usa_referencias_a_grupos(tmp_path, regex):
    # Su coste depende de lo capturado: no se puede acotar, así que no se supone inocuo.
    with pytest.raises(InvalidLexicon, match="no se puede acotar"):
        RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))


def test_tras_retirar_una_excepcion_sigue_habiendo_un_solo_espacio_entre_palabras(tmp_path):
    # El análisis de coste admite "\s+" sin tope porque el texto nunca tiene dos espacios seguidos.
    contenido = LEXICO_OK + (
        "  marcador:\n    nivel: LOW\n    patrones:\n      - regex: '\\bantes \\| \\| despues\\b'\n"
        "excepciones_idiomaticas: [me muero de la risa]\n"
    )
    detector = RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))
    texto = "antes me muero de la risa, me muero de la risa después"
    assert detector.detect(texto).categorias == ("marcador",)


@pytest.mark.parametrize(
    "regex,texto",
    [
        (r"\bsuicid\w*", "pienso en el suicidio"),  # sin tope, pero es lo último de la regex
        (r"\bquiero\s+morir\w*", "quiero morirme"),  # sin tope sobre espacios
        (r"\bquiero +morir", "quiero morir"),
        (r"\b(?:me |te ){1,2}mato", "me mato"),  # pocas vueltas de una alternativa: cabe en el tope
        (r"\bquiero(?:\s\w{1,20}+){0,3}\s+morir", "quiero de una vez morir"),  # de 0 a 3 palabras
        (r"\bya\s+no\s+(?:\w{1,20}+\s){0,2}vivir", "ya no quiero vivir"),
        (r"\bno\b.{0,40}\bvivir\b", "no tengo ganas de vivir"),  # un hueco con tope
        (r"\b(?:me|te)\s+(?:mato|muero)\b", "si no me mato"),
        (r"(?>\bno\s+quiero|\bno\s+puedo)\s+vivir", "no puedo vivir"),
        (r"\b(?:ja){2,10}\s+me\s+mato", "jajaja me mato"),
        (r"\b(?:ja|jaja|jajaja){1,30}+\s+me\s+mato", "jajajajaja me mato"),  # posesiva: no retrocede
        (r"[a-z]{2,10}\d{2}", "clave ab12"),
    ],
)
def test_acepta_una_regex_de_coste_acotado_y_detecta_con_ella(tmp_path, regex, texto):
    detector = RulesRiskDetector.from_file(escribir_lexico(tmp_path, lexico_con_regex_high(regex)))
    assert detector.detect(texto).level is RiskLevel.HIGH
    assert detector.detect("hoy fue un buen día").level is RiskLevel.NONE


#: Mensajes largos (~4 000 caracteres, el límite de la API) hechos con las palabras de las propias
#: regex: son los que disparan el retroceso, no los textos genéricos.
ADVERSARIOS = [
    "no quiero " * 400,
    "quiero morir " * 307,
    "quiero " + "ab" * 1996,
    "quiero " + "ab " * 1331,
    "ja" * 1999 + "x",
    "ja" * 1990 + " me muero",
    "no " * 1333,
    "a" * 4000,
    "a1" * 2000,
    f"me{ZWSP}muero " * 444,
]


def test_detect_con_textos_adversarios_largos_es_rapido(tmp_path):
    # Las variantes con tope de las regex que antes bloqueaban `detect` durante segundos o minutos.
    contenido = (
        lexico_con_regex_high(
            r"\bno\b.{0,40}\bvivir",
            r"\bquiero(?:\s\w{1,20}+){0,3}\s+morir",
            r"\bquiero\b.{0,60}\bmorir\w*",
            r"\bno(?:\s\w{1,20}+){0,4}\squiero(?:\s\w{1,20}+){0,4}\svivir",
            r"\bsuicid\w*",
        )
        + "  malestar_agudo:\n    nivel: LOW\n    patrones: [me muero]\n"
        + "excepciones_idiomaticas:\n  - regex: '\\b(?:ja){1,20}\\s+me\\s+muero'\n"
    )
    detector = RulesRiskDetector.from_file(escribir_lexico(tmp_path, contenido))
    inicio = time.perf_counter()
    for texto in ADVERSARIOS:
        detector.detect(texto)
    assert time.perf_counter() - inicio < 0.5


def test_normalizacion_sin_expandir_abreviaturas_conserva_la_palabra_original():
    assert TextNormalizer(ABREVIATURAS).normalize("¡Xq, tmb!", expandir_abreviaturas=False) == "xq tmb"


def test_un_lexico_minimo_sin_normalizacion_ni_excepciones_es_valido(tmp_path):
    ruta = escribir_lexico(tmp_path, LEXICO_OK)
    assert RulesRiskDetector.from_file(ruta).detect("Quiero morirme").level is RiskLevel.HIGH


# ----------------------------------------------------------------------------- privacidad
def test_el_log_de_arranque_tiene_la_version_y_no_el_texto_del_lexico_ni_del_usuario(caplog):
    secreto = "zorro_secreto_8841"
    with caplog.at_level(logging.DEBUG):
        detector = RulesRiskDetector.from_file(VALIDO)
        detector.detect(f"{secreto} quiero morirme")
    assert any(getattr(r, "lexico_version", None) == "prueba-1.0" for r in caplog.records)
    volcado = " ".join(str(r.__dict__) for r in caplog.records)
    for prohibido in (secreto, "morirme", "hacerme", "risa", "estudiando", "SUICÍD", "suicid", "porque"):
        assert prohibido not in volcado
