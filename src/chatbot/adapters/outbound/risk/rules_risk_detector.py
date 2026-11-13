"""Detector de riesgo por reglas con léxico versionado en archivo (§5.3.a de la especificación).

Implementa `RiskDetectorPort`: síncrono y sin E/S en `detect` (el archivo se lee una sola vez, al
construir el detector con `from_file`). Formato del léxico (YAML):

    version: "0.1.0-borrador"        # obligatorio, texto
    fecha: "2026-11-11"              # opcional
    revisado_por: PENDIENTE          # obligatorio, texto
    normalizacion:                   # opcional; ver `normalization.py`
      minusculas: true               # las tres banderas solo admiten true (o no ponerlas)
      quitar_tildes: true
      colapsar_letras_repetidas: true
      abreviaturas: {q: que, xq: porque}
    categorias:                      # obligatorio, al menos una
      ideacion_suicida:              # nombre en snake_case: es lo único que sale del detector
        nivel: HIGH                  # LOW | MEDIUM | HIGH
        patrones:
          - quiero morirme           # frase: se normaliza y se compara por palabras completas
          - regex: '\\bsuicid\\w*'     # regex: se compara contra el texto ya normalizado
    excepciones_idiomaticas:         # opcional; mismas formas que los patrones
      - me muero de la risa

Criterio de decisión (ante la duda, el nivel más alto):

- El resultado es el nivel MÁS ALTO entre las categorías encontradas, con los nombres de todas
  ellas (primero las de mayor nivel). Nunca se devuelve texto del usuario ni del léxico.
- Las excepciones idiomáticas se retiran del texto SOLO para buscar patrones LOW y MEDIUM. Los
  patrones HIGH se buscan siempre en el texto completo: una excepción jamás suprime un HIGH
  ("me mato estudiando" sigue siendo HIGH si "me mato" es un patrón HIGH).
- Una excepción debe ser la frase idiomática COMPLETA: retira todo el tramo que cubre, así que
  una excepción que solo solapa en parte un patrón LOW/MEDIUM también lo rebaja (con la excepción
  "mas de la risa", "no puedo mas de la risa" deja de contar como "no puedo mas"). Por eso el
  arranque rechaza una excepción que pueda cubrir menos de 4 caracteres (`\\s`, `.`, "ay").
- Los patrones HIGH se buscan en DOS formas del texto: normalizado con las abreviaturas
  expandidas y normalizado sin expandirlas. Basta una coincidencia en cualquiera de las dos, así
  que una abreviatura nunca deja muerta una regex HIGH (p. ej. `suicidio: tema` frente a
  `\\bsuicid\\w*`). LOW y MEDIUM se buscan solo en la forma expandida y sin las excepciones (una
  regex LOW o MEDIUM se escribe con las abreviaturas ya expandidas).
- Un carácter invisible (espacio de ancho cero, guion suave, un carácter de control...) puede ir
  dentro de una palabra o entre dos, y no se puede saber cuál. Si el mensaje trae alguno, todos
  los niveles se buscan en las dos lecturas: quitándolos y cambiándolos por un espacio. Límite
  conocido: un mismo patrón con un invisible dentro de una palabra Y otro entre dos palabras.
- No hay detección de negaciones: "no pienso hacerme daño" coincide con el patrón "hacerme daño"
  y es HIGH. Una negación no permite descartar el riesgo con certeza y un falso negativo tiene
  costo humano; afinar esos casos es tarea del léxico (patrones más específicos), no del motor.

El arranque falla (`InvalidLexicon`) si el archivo falta, el YAML está mal formado o tiene claves
duplicadas, falta la cabecera, un nivel es desconocido, no hay ninguna categoría HIGH, una
categoría no tiene patrones, una bandera de normalización es false o una regex no compila: un
léxico a medias dejaría huecos silenciosos en la detección. Además se rechaza una regex (patrón o
excepción) que:

- puede coincidir con un texto vacío (`''`, `.*`, `x?`, `\\b`, `(?=...)`): como patrón HIGH
  podría volver crisis cualquier mensaje; como excepción, borrar todo LOW y MEDIUM;
- exige una letra repetida (`mu{2,}erto`, `l{2}oro`): el texto llega con cada racha reducida a
  una letra y el patrón nunca coincidiría (se escribe `muerto`). Límite conocido: no se ven
  todas las formas de un patrón muerto (`[u]u`, `u{1}u`); lo delata el recall del léxico;
- podría tardar demasiado. `detect` corre sin tiempo límite dentro del bucle de eventos: una
  regex lenta bloquearía la ruta de crisis de todos. Por eso el tiempo de cada regex debe crecer
  solo en proporción a la longitud del mensaje, y se exige (ver `_coste`):
    * que no haya cuantificadores anidados (`(a+)+`, `(?:\\s\\w{1,20}){0,3}`);
    * que toda repetición tenga máximo: nada de `*`, `+` ni `{n,}`. Se escribe `\\w{0,30}` o
      `.{0,40}`. Sin máximo solo se admite sobre espacios (`\\s+`) y sobre un solo carácter
      como último elemento de la regex (`\\bsuicid\\w*`);
    * que las combinaciones que el motor puede probar desde una posición no pasen de un tope:
      un hueco variable (`\\bno\\b.{0,40}\\bvivir`) cabe; dos o tres encadenados
      (`a.{0,40}b.{0,40}c`), alternativas que se solapan dentro de una repetición
      (`(?:ja|jaja){1,30}`) o cuantificadores adyacentes (`\\w{0,30}\\w{0,30}`), no;
    * que no haya referencias a grupos (`\\1`, `(?(1)...)`): su coste no se puede acotar.
  Para dejar pasar de 0 a 3 palabras entre dos se usa una repetición posesiva, que no vuelve
  atrás: `\\bquiero(?:\\s\\w{1,20}+){0,3}\\s+morir`, `\\bya\\s+no\\s+(?:\\w{1,20}+\\s){0,2}vivir`.

Este análisis es estático y conservador (rechaza algunas regex inofensivas), pero no depende de
medir tiempos al arrancar ni de adivinar el mensaje que dispara el retroceso.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from chatbot.adapters.outbound.risk.normalization import (
    NormalizationConfig,
    TextNormalizer,
    contiene_invisibles,
)
from chatbot.application.dto import RiskDetection
from chatbot.domain.models import RiskLevel

logger = logging.getLogger(__name__)

_NOMBRE_CATEGORIA = re.compile(r"[a-z][a-z0-9_]*")
_BANDERAS = ("minusculas", "quitar_tildes", "colapsar_letras_repetidas")
_NIVELES_VALIDOS = ("LOW", "MEDIUM", "HIGH")
#: Una excepción idiomática es una frase completa: menos de esto la vuelve un comodín.
_ANCHO_MINIMO_EXCEPCION = 4


class InvalidLexicon(ValueError):
    """El léxico de riesgo no se puede usar: el servicio no debe arrancar con él."""


@dataclass(frozen=True, slots=True)
class RiskCategory:
    nombre: str
    nivel: RiskLevel
    patrones: tuple[re.Pattern[str], ...]


class RulesRiskDetector:
    def __init__(
        self,
        *,
        version: str,
        categorias: Sequence[RiskCategory],
        excepciones: Sequence[re.Pattern[str]] = (),
        normalizador: TextNormalizer | None = None,
    ) -> None:
        if not categorias:
            raise InvalidLexicon("El léxico de riesgo no tiene categorias")
        if not any(c.nivel is RiskLevel.HIGH for c in categorias):
            # Sin HIGH, la ruta de crisis nunca se activaría por reglas.
            raise InvalidLexicon("El léxico de riesgo no tiene ninguna categoría HIGH")
        self._version = version
        # Orden estable: primero las de mayor nivel; dentro del mismo nivel, el del léxico.
        self._categorias = tuple(sorted(categorias, key=lambda c: -c.nivel))
        self._excepciones = tuple(excepciones)
        self._normalizador = normalizador or TextNormalizer()
        # Solo metadatos: ni patrones ni excepciones (ni, por supuesto, texto del usuario).
        logger.info(
            "lexico_riesgo_cargado",
            extra={
                "lexico_version": version,
                "lexico_categorias": len(self._categorias),
                "lexico_patrones": sum(len(c.patrones) for c in self._categorias),
                "lexico_excepciones": len(self._excepciones),
            },
        )

    @property
    def version(self) -> str:
        return self._version

    def detect(self, texto: str) -> RiskDetection:
        formas_high: list[str] = []
        formas_bajas: list[str] = []
        # Un carácter invisible puede ir dentro de una palabra o entre dos: ante la duda se busca
        # en las dos lecturas (casi nunca hay ninguno, y entonces solo existe la primera).
        lecturas = (False, True) if contiene_invisibles(texto) else (False,)
        for como_espacio in lecturas:
            expandida = self._normalizador.normalize(texto, invisibles_como_espacio=como_espacio)
            # HIGH se busca además sin expandir las abreviaturas: así una abreviatura que reescribe
            # una palabra que una regex HIGH necesita no la deja muerta en silencio.
            sin_expandir = self._normalizador.normalize(
                texto, expandir_abreviaturas=False, invisibles_como_espacio=como_espacio
            )
            formas_high += [expandida, sin_expandir]
            formas_bajas.append(self._sin_excepciones(expandida))
        # Sin formas repetidas (sin abreviaturas ni invisibles todas son iguales).
        altas, bajas = tuple(dict.fromkeys(formas_high)), tuple(dict.fromkeys(formas_bajas))
        encontradas = [
            c
            for c in self._categorias
            if any(
                p.search(forma)
                for forma in (altas if c.nivel is RiskLevel.HIGH else bajas)
                for p in c.patrones
            )
        ]
        if not encontradas:
            return RiskDetection(RiskLevel.NONE)
        return RiskDetection(max(c.nivel for c in encontradas), tuple(c.nombre for c in encontradas))

    def _sin_excepciones(self, forma: str) -> str:
        """Retira las excepciones idiomáticas de una forma del texto."""
        for excepcion in self._excepciones:
            # Se reemplaza por un separador que ninguna frase del léxico puede contener.
            forma, retiradas = excepcion.subn(" | ", forma)
            if retiradas:
                # Sigue habiendo un solo espacio entre palabras: el análisis de coste de las regex
                # (`_coste`) cuenta con que nunca hay dos seguidos.
                forma = " ".join(forma.split())
        return forma

    # ------------------------------------------------------------------------- carga
    @classmethod
    def from_file(cls, ruta: str | Path) -> RulesRiskDetector:
        """Lee y valida el léxico; lanza `InvalidLexicon` ante cualquier problema (fail-fast)."""
        ruta = Path(ruta)
        try:
            contenido = ruta.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise InvalidLexicon(f"No existe el léxico de riesgo: {ruta}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise InvalidLexicon(f"No se pudo leer el léxico de riesgo {ruta}: {exc}") from exc
        try:
            datos = yaml.load(contenido, Loader=_CargadorSinDuplicados)
        except yaml.YAMLError as exc:
            raise InvalidLexicon(f"YAML mal formado en {ruta}: {_describir_error_yaml(exc)}") from exc
        return cls._from_data(datos)

    @classmethod
    def _from_data(cls, datos: object) -> RulesRiskDetector:
        if not isinstance(datos, Mapping):
            raise InvalidLexicon("El léxico de riesgo debe ser un mapa YAML con cabecera y categorias")
        version = _texto_obligatorio(datos, "version")
        _texto_obligatorio(datos, "revisado_por")
        normalizador = _normalizador(datos.get("normalizacion"))
        categorias_crudas = datos.get("categorias")
        if not isinstance(categorias_crudas, Mapping) or not categorias_crudas:
            raise InvalidLexicon("El léxico debe definir al menos una entrada en 'categorias'")
        categorias = [_categoria(nombre, cfg, normalizador) for nombre, cfg in categorias_crudas.items()]
        excepciones_crudas = datos.get("excepciones_idiomaticas") or []
        if not isinstance(excepciones_crudas, list):
            raise InvalidLexicon("'excepciones_idiomaticas' debe ser una lista")
        excepciones = [
            _excepcion(p, normalizador, f"excepcion #{i + 1}") for i, p in enumerate(excepciones_crudas)
        ]
        return cls(version=version, categorias=categorias, excepciones=excepciones, normalizador=normalizador)


# ----------------------------------------------------------------------------- validación
class _CargadorSinDuplicados(yaml.SafeLoader):
    """`SafeLoader` que rechaza claves repetidas (PyYAML se queda en silencio con la última)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        vistas: list[object] = []
        for nodo_clave, _ in node.value:
            if nodo_clave.tag == "tag:yaml.org,2002:merge":
                continue  # "<<: *ancla" lo resuelve SafeLoader; no es una clave del léxico
            clave = self.construct_object(nodo_clave, deep=deep)
            if clave in vistas:
                raise yaml.constructor.ConstructorError(
                    None, None, f"clave duplicada {clave!r}", nodo_clave.start_mark
                )
            vistas.append(clave)
        return super().construct_mapping(node, deep=deep)


def _describir_error_yaml(exc: yaml.YAMLError) -> str:
    """Problema y posición, sin copiar el fragmento del archivo."""
    if isinstance(exc, yaml.MarkedYAMLError):
        problema = exc.problem or "error de sintaxis"
        marca = exc.problem_mark
        if marca is not None:
            return f"{problema} (línea {marca.line + 1}, columna {marca.column + 1})"
        return problema
    return "error de sintaxis"


def _texto_obligatorio(datos: Mapping[Any, Any], clave: str) -> str:
    valor = datos.get(clave)
    if not isinstance(valor, str) or not valor.strip():
        raise InvalidLexicon(f"Falta la cabecera '{clave}' (texto no vacío, entre comillas si es un número)")
    return valor.strip()


def _normalizador(crudo: object) -> TextNormalizer:
    if crudo is None:
        return TextNormalizer()
    if not isinstance(crudo, Mapping):
        raise InvalidLexicon("'normalizacion' debe ser un mapa")
    # Las banderas se aceptan (forma de §5.3.a), pero solo con true: apagar un paso deja de comparar
    # el texto y los patrones en la misma forma y pierde HIGH en silencio ("Quiero morirme").
    for bandera in _BANDERAS:
        if crudo.get(bandera, True) is not True:
            raise InvalidLexicon(
                f"'normalizacion.{bandera}' debe ser true: desactivarla deja pasar riesgos HIGH sin detectar"
            )
    abreviaturas = crudo.get("abreviaturas") or {}
    if not isinstance(abreviaturas, Mapping):
        raise InvalidLexicon("'normalizacion.abreviaturas' debe ser un mapa abreviatura: expansión")
    pares: list[tuple[str, str]] = []
    for abreviatura, expansion in abreviaturas.items():
        if not isinstance(abreviatura, str) or not isinstance(expansion, str):
            raise InvalidLexicon("Cada abreviatura y su expansión deben ser texto (entre comillas)")
        pares.append((abreviatura, expansion))
    try:
        return TextNormalizer(NormalizationConfig(abreviaturas=tuple(pares)))
    except ValueError as exc:
        raise InvalidLexicon(f"'normalizacion.abreviaturas' inválida: {exc}") from exc


def _categoria(nombre: object, cfg: object, normalizador: TextNormalizer) -> RiskCategory:
    if not isinstance(nombre, str) or not _NOMBRE_CATEGORIA.fullmatch(nombre):
        raise InvalidLexicon("El nombre de una categoría debe estar en snake_case (p. ej. ideacion_suicida)")
    if not isinstance(cfg, Mapping):
        raise InvalidLexicon(f"La categoría '{nombre}' debe ser un mapa con 'nivel' y 'patrones'")
    nivel_crudo = cfg.get("nivel")
    if not isinstance(nivel_crudo, str) or nivel_crudo.strip().upper() not in _NIVELES_VALIDOS:
        raise InvalidLexicon(
            f"La categoría '{nombre}' tiene un nivel desconocido; usa {', '.join(_NIVELES_VALIDOS)}"
        )
    patrones = cfg.get("patrones")
    if not isinstance(patrones, list) or not patrones:
        raise InvalidLexicon(f"La categoría '{nombre}' debe tener una lista de 'patrones' no vacía")
    return RiskCategory(
        nombre=nombre,
        nivel=RiskLevel.parse(nivel_crudo),
        patrones=tuple(
            _compilar(p, normalizador, f"categoría '{nombre}', patrón #{i + 1}")
            for i, p in enumerate(patrones)
        ),
    )


def _excepcion(patron: object, normalizador: TextNormalizer, donde: str) -> re.Pattern[str]:
    """Como un patrón, pero además debe cubrir al menos `_ANCHO_MINIMO_EXCEPCION` caracteres."""
    compilado = _compilar(patron, normalizador, donde)
    if _re_parser.parse(compilado.pattern).getwidth()[0] < _ANCHO_MINIMO_EXCEPCION:
        # Una excepción de una letra o un espacio partiría todo el texto y borraría LOW y MEDIUM.
        raise InvalidLexicon(
            f"{donde}: la excepción es demasiado corta (menos de {_ANCHO_MINIMO_EXCEPCION} "
            "caracteres); debe ser la frase idiomática completa"
        )
    return compilado


def _compilar(patron: object, normalizador: TextNormalizer, donde: str) -> re.Pattern[str]:
    """Frase (texto) → palabras completas tras normalizar; `{regex: ...}` → regex normalizada.

    Los mensajes de error indican la posición del patrón, no su contenido.
    """
    if isinstance(patron, str):
        palabras = normalizador.normalize(patron).split()
        if not palabras:
            raise InvalidLexicon(f"{donde}: la frase queda vacía tras normalizarla")
        return re.compile(r"\b" + r"\s+".join(re.escape(p) for p in palabras) + r"\b")
    if isinstance(patron, Mapping) and set(patron) == {"regex"} and isinstance(patron["regex"], str):
        normalizada = normalizador.normalize_regex(patron["regex"])
        try:
            compilado = re.compile(normalizada)
        except re.error as exc:
            raise InvalidLexicon(f"{donde}: regex inválida ({exc})") from exc
        _validar_regex(normalizada, donde)
        return compilado
    raise InvalidLexicon(f"{donde}: debe ser una frase (texto entre comillas) o un mapa {{regex: '...'}}")


# ----------------------------------------------------------------------------- reglas de regex
#: Analizador interno de `re` (CPython 3.11+). Se carga por nombre porque no es API pública.
_re_parser: Any = importlib.import_module("re._parser")


def _validar_regex(regex: str, donde: str) -> None:
    """Rechaza al arrancar las regex que pondrían en riesgo la detección o su tiempo de respuesta."""
    arbol = _re_parser.parse(regex)
    if arbol.getwidth()[0] == 0:
        raise InvalidLexicon(
            f"{donde}: la regex puede coincidir con un texto vacío (no obliga a que haya ninguna "
            "palabra); añade una parte que consuma texto"
        )
    if _anida_cuantificadores(arbol, dentro_de_repeticion=False):
        raise InvalidLexicon(
            f"{donde}: regex con cuantificadores anidados (riesgo de bloqueo); si de verdad hace "
            "falta, usa una repetición posesiva con máximo (p. ej. '(?:\\s\\w{1,20}+){0,3}')"
        )
    if _exige_letra_repetida(arbol):
        raise InvalidLexicon(
            f"{donde}: la regex exige una letra repetida (p. ej. 'u{{2,}}'), pero el texto llega con "
            "cada racha reducida a una letra y nunca coincidiría; escribe la letra una sola vez"
        )
    try:
        _, trabajo = _coste(arbol, final=True)
    except _RegexRechazada as exc:
        raise InvalidLexicon(f"{donde}: {exc}") from exc
    if trabajo > _TRABAJO_MAXIMO:
        raise InvalidLexicon(
            f"{donde}: la regex admite demasiadas combinaciones por posición del texto (riesgo de "
            "lentitud); baja los máximos, deja un solo hueco variable o haz posesivas las "
            "repeticiones (p. ej. '(?:\\s\\w{1,20}+){0,3}')"
        )


def _nodos(arbol: Any) -> Iterator[tuple[Any, Any]]:
    """Todos los nodos (operación, argumento) del árbol, en profundidad."""
    for operacion, argumento in arbol:
        yield operacion, argumento
        for hijo in _hijos(argumento):
            yield from _nodos(hijo)


def _exige_letra_repetida(arbol: Any) -> bool:
    for operacion, argumento in _nodos(arbol):
        if operacion in (_re_parser.MAX_REPEAT, _re_parser.MIN_REPEAT, _re_parser.POSSESSIVE_REPEAT):
            minimo, _, cuerpo = argumento
            if minimo >= 2 and len(cuerpo) == 1:
                operacion_cuerpo, valor = cuerpo[0]
                if operacion_cuerpo is _re_parser.LITERAL and chr(valor).isalpha():
                    return True
    return False


def _hijos(argumento: object) -> Iterator[Any]:
    """Subpatrones contenidos en el argumento de un nodo del árbol (grupos, ramas, aserciones)."""
    if isinstance(argumento, _re_parser.SubPattern):
        yield argumento
    elif isinstance(argumento, tuple | list):
        for elemento in argumento:
            yield from _hijos(elemento)


def _anida_cuantificadores(arbol: Any, *, dentro_de_repeticion: bool) -> bool:
    """¿Hay una repetición de longitud variable, no posesiva, dentro de otra que se repite?

    Es la forma típica del retroceso catastrófico (`(a+)+`, `(\\w+\\s?)+`). Un grupo atómico o una
    repetición posesiva no dejan volver a entrar desde fuera, así que reinician el contexto (se
    sigue revisando lo que contienen, y lo que retroceden por dentro lo acota `_coste`).
    """
    for operacion, argumento in arbol:
        if operacion in (_re_parser.MAX_REPEAT, _re_parser.MIN_REPEAT):
            minimo, maximo, cuerpo = argumento
            if dentro_de_repeticion and minimo != maximo and maximo > 1:
                return True
            if _anida_cuantificadores(cuerpo, dentro_de_repeticion=dentro_de_repeticion or maximo > 1):
                return True
        elif operacion in (_re_parser.POSSESSIVE_REPEAT, _re_parser.ATOMIC_GROUP):
            if any(_anida_cuantificadores(h, dentro_de_repeticion=False) for h in _hijos(argumento)):
                return True
        elif any(
            _anida_cuantificadores(h, dentro_de_repeticion=dentro_de_repeticion) for h in _hijos(argumento)
        ):
            return True
    return False


# ----------------------------------------------------------------------------- coste de una regex
#: Tope del trabajo de búsqueda por posición del texto (ver `_coste`). Con 4 000 caracteres (el
#: límite de la API) son como mucho unos 4 millones de pasos por regex: decenas de milisegundos.
_TRABAJO_MAXIMO = 1000
#: "Demasiado": a partir de aquí no se sigue sumando (evita enteros enormes con `(a|b){1,9999}`).
_SATURADO = 10**9


class _RegexRechazada(Exception):
    """La regex no se puede acotar; el mensaje dice por qué, sin copiar su contenido."""


def _coste(arbol: Any, *, final: bool) -> tuple[int, int]:
    """Cota de (caminos, trabajo) al intentar la secuencia `arbol` desde UNA posición del texto.

    `caminos` es de cuántas formas distintas puede terminar de coincidir y `trabajo`, cuántos pasos
    da el motor como mucho al probarlas todas (retroceso incluido): cada camino de lo anterior
    vuelve a intentar lo que sigue. Como no depende de la longitud del texto, la búsqueda completa
    cuesta como mucho `longitud × trabajo`: es lineal.

    Para que esa cota exista, toda repetición debe tener máximo. Sin máximo solo se admiten dos
    casos que no multiplican el trabajo: sobre espacios (el texto normalizado nunca tiene dos
    seguidos) y un solo carácter repetido al final de la regex (`final`: después ya no hay nada
    que pueda fallar y obligar a volver atrás, así que o falla antes del mínimo o hay coincidencia
    y la búsqueda termina).
    """
    caminos, trabajo = 1, 0
    for i, (operacion, argumento) in enumerate(arbol):
        c, t = _coste_nodo(operacion, argumento, final=final and i == len(arbol) - 1)
        trabajo = min(_SATURADO, trabajo + caminos * t)
        caminos = min(_SATURADO, caminos * c)
    return caminos, trabajo


def _coste_nodo(operacion: Any, argumento: Any, *, final: bool) -> tuple[int, int]:
    if operacion in _UN_CARACTER or operacion in (_re_parser.AT, _re_parser.FAILURE):
        return 1, 1
    if operacion is _re_parser.SUBPATTERN:
        return _coste(argumento[3], final=final)
    if operacion is _re_parser.BRANCH:
        ramas = [_coste(rama, final=final) for rama in argumento[1]]
        return sum(c for c, _ in ramas), 1 + sum(t for _, t in ramas)
    if operacion in (_re_parser.ASSERT, _re_parser.ASSERT_NOT):
        # Una aserción no se reintenta: una vez resuelta deja un solo camino.
        return 1, 1 + _coste(argumento[1], final=False)[1]
    if operacion is _re_parser.ATOMIC_GROUP:
        # No se vuelve a entrar desde fuera, pero por dentro sí retrocede hasta coincidir.
        return 1, 1 + _coste(argumento, final=False)[1]
    if operacion in (_re_parser.MAX_REPEAT, _re_parser.MIN_REPEAT, _re_parser.POSSESSIVE_REPEAT):
        return _coste_repeticion(operacion, *argumento, final=final)
    # Referencias a grupos (`\1`, `(?(1)...)`), cuyo coste depende de lo capturado, o un nodo que
    # este análisis no conoce (otra versión de Python): se rechaza, nunca se supone inocuo.
    raise _RegexRechazada(
        "la regex usa una construcción cuyo coste no se puede acotar (p. ej. una referencia a un grupo)"
    )


def _coste_repeticion(
    operacion: Any, minimo: int, maximo: int, cuerpo: Any, *, final: bool
) -> tuple[int, int]:
    un_caracter = len(cuerpo) == 1 and cuerpo[0][0] in _UN_CARACTER
    if maximo == _re_parser.MAXREPEAT:
        if un_caracter and _solo_espacios(*cuerpo[0]):
            maximo = max(minimo, 1)  # en el texto normalizado no hay dos espacios seguidos
        elif un_caracter and final:
            return 1, 1 + minimo  # o falla antes de llegar al mínimo o la regex entera ya coincidió
        else:
            raise _RegexRechazada(
                "regex con una repetición sin máximo ('*', '+' o '{n,}'): con un mensaje largo "
                "bloquearía la detección; ponle un tope (p. ej. '\\w{0,30}', '.{0,40}'). Sin tope solo "
                "se admite sobre espacios ('\\s+') o sobre un carácter al final de la regex ('\\w*')"
            )
    c, t = _coste(cuerpo, final=False)
    if operacion is _re_parser.POSSESSIVE_REPEAT:
        return 1, 1 + maximo * t  # cada vuelta se queda con su primera coincidencia y no retrocede
    if c == 1:
        return maximo - minimo + 1, 1 + maximo * t
    caminos, trabajo, variantes = 0, 1, 1  # variantes = c ** k: formas de haber dado k vueltas
    for k in range(maximo + 1):
        if k >= minimo:
            caminos += variantes
        if k < maximo:
            trabajo += variantes * t
        variantes *= c
        if trabajo > _SATURADO:
            return _SATURADO, _SATURADO
    return caminos, trabajo


def _solo_espacios(operacion: Any, argumento: Any) -> bool:
    """¿El carácter solo puede ser un espacio? (` `, `\\s`, `[ \\s]`)."""
    if operacion is _re_parser.LITERAL:
        return chr(argumento).isspace()
    if operacion is _re_parser.IN:
        return all(
            elemento == (_re_parser.CATEGORY, _re_parser.CATEGORY_SPACE)
            or (elemento[0] is _re_parser.LITERAL and chr(elemento[1]).isspace())
            for elemento in argumento
        )
    return False


#: Nodos que consumen exactamente un carácter.
_UN_CARACTER = (_re_parser.LITERAL, _re_parser.NOT_LITERAL, _re_parser.ANY, _re_parser.IN)

# Canario: si otra versión de Python cambia la forma del árbol y el análisis deja de ver los casos
# de libro, el módulo falla al importarse en lugar de aceptar en silencio regex peligrosas.
if not _anida_cuantificadores(_re_parser.parse("(?:a+)+b"), dentro_de_repeticion=False) or (
    _coste(_re_parser.parse("(?:a|ab){1,30}c"), final=True)[1] <= _TRABAJO_MAXIMO
):
    raise RuntimeError("re._parser cambió: el análisis de las regex del léxico ya no es fiable")
