"""Normalización del texto para el detector de riesgo por reglas.

Pipeline (se aplica al texto del usuario Y a las frases del léxico, para que ambos lados se
comparen en la misma forma):

    minúsculas → quitar tildes → colapsar letras repetidas → expandir abreviaturas → limpiar puntuación

- Antes de todo, siempre: NFKC (formas de compatibilidad, "ｑｕｉｅｒｏ" → "quiero") y se tratan los
  caracteres invisibles (ver `es_invisible`: espacio de ancho cero, guion suave, selectores de
  variante, caracteres de control...). No se sabe si uno de ellos va DENTRO de una palabra
  ("mori|rme") o ENTRE dos ("quiero|morirme"), así que hay dos lecturas: por defecto se quitan y,
  con `invisibles_como_espacio=True`, se cambian por un espacio. El detector busca en las dos.
- Colapsar letras repetidas deja UNA sola letra en cualquier racha ("muuuerto" → "muerto", pero
  también "llorar" → "lorar"). Como se aplica a los dos lados, la comparación sigue siendo
  coherente; las colisiones ("perro"/"pero") solo pueden producir falsos positivos, nunca omitir
  un patrón. Los dígitos no se colapsan.
- Las abreviaturas se expanden por palabra completa ("q" → "que", pero "queso" no cambia).
- Limpiar puntuación cambia todo lo que no es letra, dígito o espacio por un espacio y deja un
  solo espacio entre palabras.

Las expresiones regulares del léxico no pueden pasar por todo el pipeline sin romper su sintaxis;
`normalize_regex` aplica solo los pasos seguros (minúsculas, tildes y letras repetidas) y respeta
las secuencias de escape (`\\b`, `\\S`, `\\W`, `\\N{...}`) y el prefijo `(?P` de los grupos con
nombre. Por eso una regex se escribe ya con las abreviaturas expandidas y sin puntuación: se
compara contra el texto normalizado. No se ponen cuantificadores sobre letras repetidas
(`mu{2,}erto`): el texto ya llega con cada racha reducida a una letra (el detector lo rechaza).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LETRAS_REPETIDAS = re.compile(r"([^\W\d_])\1+")
_PUNTUACION = re.compile(r"[^\w\s]|_")
_PALABRAS = re.compile(r"[^\W_]+(?: [^\W_]+)*")
# Sintaxis de regex que no se altera: "\N{NOMBRE}", cualquier escape ("\b", "\S"...) y "(?P".
_SINTAXIS_INTACTA = re.compile(r"\\N\{[^}]*\}|\\.|\(\?P", re.DOTALL)


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Pasos activables del pipeline. Por defecto todos activos y sin abreviaturas."""

    minusculas: bool = True
    quitar_tildes: bool = True
    colapsar_letras_repetidas: bool = True
    #: Pares (abreviatura, expansión), p. ej. ("xq", "porque"); se normalizan al construir.
    abreviaturas: tuple[tuple[str, str], ...] = ()


#: Invisibles que Unicode no clasifica como formato (Cf): "letras" de relleno en blanco (Lo) y
#: marcas sin forma propia (Mn: unidor de grafemas, selectores de variante). Son los "ignorables
#: por defecto" de Unicode que no son Cf. Se escriben por su código para que se vean.
_INVISIBLES_NO_CF = frozenset(
    chr(codigo)
    for codigo in (
        0x034F,  # unidor de grafemas
        0x115F,  # rellenos hangul
        0x1160,
        0x3164,
        0xFFA0,
        0x17B4,  # vocales jemer inherentes
        0x17B5,
        *range(0x180B, 0x1810),  # selectores de variante mongoles (0x180E es Cf)
        *range(0xFE00, 0xFE10),  # selectores de variante
        *range(0xE0100, 0xE01F0),  # selectores de variante suplementarios
    )
)


def es_invisible(caracter: str) -> bool:
    """¿Es un carácter que no se ve? Formato (Cf), control que no es un espacio o ignorable."""
    categoria = unicodedata.category(caracter)
    return (
        categoria == "Cf" or (categoria == "Cc" and not caracter.isspace()) or caracter in _INVISIBLES_NO_CF
    )


def contiene_invisibles(texto: str) -> bool:
    """¿Hay algún carácter invisible? Si no lo hay, las dos lecturas del texto son la misma."""
    return any(es_invisible(c) for c in texto)


def _sin_tildes(texto: str) -> str:
    """Quita las marcas diacríticas ("á" → "a", "ñ" → "n")."""
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


class TextNormalizer:
    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self._config = config or NormalizationConfig()
        expansiones: dict[str, str] = {}
        for abreviatura, expansion in self._config.abreviaturas:
            clave = " ".join(self._pasos_basicos(abreviatura).split())
            if not _PALABRAS.fullmatch(clave):
                raise ValueError("Una abreviatura debe estar formada solo por letras o dígitos")
            expansiones[clave] = self._pasos_basicos(expansion)
        self._expansiones = expansiones
        self._abreviaturas: re.Pattern[str] | None = None
        if expansiones:
            alternativas = "|".join(re.escape(k) for k in sorted(expansiones, key=len, reverse=True))
            # Límite de palabra que trata "_" como separador ("nose_que" → "no se_que").
            self._abreviaturas = re.compile(rf"(?<![^\W_])(?:{alternativas})(?![^\W_])")

    def normalize(
        self, texto: str, *, expandir_abreviaturas: bool = True, invisibles_como_espacio: bool = False
    ) -> str:
        """Aplica el pipeline completo; devuelve palabras separadas por un solo espacio.

        Con `expandir_abreviaturas=False` se omite solo ese paso: es la segunda forma del texto en
        la que el detector busca los patrones HIGH (una abreviatura nunca debe anular un HIGH).
        Con `invisibles_como_espacio=True` los caracteres invisibles separan palabras en lugar de
        desaparecer: es la otra lectura posible del mismo texto.
        """
        resultado = self._pasos_basicos(texto, invisibles_como_espacio=invisibles_como_espacio)
        if expandir_abreviaturas and self._abreviaturas is not None:
            resultado = self._abreviaturas.sub(lambda m: self._expansiones[m.group(0)], resultado)
        return " ".join(_PUNTUACION.sub(" ", resultado).split())

    def normalize_regex(self, patron: str) -> str:
        """Minúsculas, tildes y letras repetidas sobre una regex, sin tocar sus secuencias de escape.

        También quita los caracteres invisibles. NFKC no se aplica: podría convertir un
        carácter en sintaxis de regex ("（" → "(").
        """
        patron = "".join(c for c in patron if not es_invisible(c))
        piezas: list[tuple[str, bool]] = []  # (texto, es_escape)
        i = 0
        while i < len(patron):
            atomica = _SINTAXIS_INTACTA.match(patron, i)
            if atomica:
                piezas.append((atomica.group(0), True))
                i = atomica.end()
                continue
            caracter = patron[i]
            if self._config.minusculas:
                caracter = caracter.lower()
            if self._config.quitar_tildes:
                caracter = _sin_tildes(caracter)
            if caracter:
                piezas.append((caracter, False))
            i += 1
        salida: list[str] = []
        anterior: tuple[str, bool] | None = None
        for pieza in piezas:
            texto, es_escape = pieza
            repetida = (
                self._config.colapsar_letras_repetidas
                and not es_escape
                and texto.isalpha()
                and anterior == (texto, False)
            )
            if not repetida:
                salida.append(texto)
            anterior = pieza
        return "".join(salida)

    def _pasos_basicos(self, texto: str, *, invisibles_como_espacio: bool = False) -> str:
        # Siempre, sin bandera: formas de compatibilidad ("ｑ" → "q") y caracteres invisibles, que
        # se quitan (lectura "dentro de una palabra") o separan palabras (lectura "entre dos").
        sustituto = " " if invisibles_como_espacio else ""
        texto = "".join(sustituto if es_invisible(c) else c for c in unicodedata.normalize("NFKC", texto))
        if self._config.minusculas:
            texto = texto.lower()
        if self._config.quitar_tildes:
            texto = _sin_tildes(texto)
        if self._config.colapsar_letras_repetidas:
            texto = _LETRAS_REPETIDAS.sub(r"\1", texto)
        return texto
