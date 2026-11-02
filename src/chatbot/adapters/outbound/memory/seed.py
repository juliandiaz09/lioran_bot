"""Datos de EJEMPLO para desarrollo y demostración. No son contenido clínico revisado.

El catálogo semilla real (5-8 técnicas revisadas) se carga en la Fase 7. Aquí NO hay
teléfonos: los contactos oficiales los debe aportar Bienestar Universitario.
"""

from __future__ import annotations

from chatbot.application.dto import CatalogDocument, DocumentType
from chatbot.domain.models import Emotion, Strategy, StrategyType

E = Emotion


def seed_strategies() -> list[Strategy]:
    return [
        Strategy(
            "respiracion_4_6",
            "Respiración 4-6",
            StrategyType.RESPIRACION,
            frozenset({E.MIEDO, E.IRA}),
            "Inhalar 4 segundos, exhalar 6.",
        ),
        Strategy(
            "grounding_5_sentidos",
            "Anclaje de los 5 sentidos",
            StrategyType.GROUNDING,
            frozenset({E.MIEDO, E.SORPRESA}),
            "Nombrar lo que se ve, oye, toca, huele y saborea.",
        ),
        Strategy(
            "reencuadre_cognitivo",
            "Reencuadre de pensamientos",
            StrategyType.REENCUADRE_COGNITIVO,
            frozenset({E.TRISTEZA, E.MIEDO}),
            "Examinar el pensamiento y buscar una alternativa.",
        ),
        Strategy(
            "validacion_emocional",
            "Validación emocional",
            StrategyType.VALIDACION_EMOCIONAL,
            frozenset({E.TRISTEZA, E.ASCO, E.NEUTRAL}),
            "Nombrar y aceptar lo que se siente.",
        ),
        Strategy(
            "planificacion_bloques",
            "Plan en bloques pequeños",
            StrategyType.PLANIFICACION,
            frozenset({E.MIEDO, E.IRA, E.NEUTRAL}),
            "Dividir la tarea en pasos de 25 minutos.",
        ),
    ]


def seed_documents() -> list[CatalogDocument]:
    return [
        CatalogDocument(
            "Respiración 4-6",
            "Siéntate cómodo y apoya la espalda. Inhala por la nariz contando "
            "hasta 4 y exhala despacio por la boca contando hasta 6. Repite durante dos minutos.",
            DocumentType.TECNICA,
            "tec-respiracion-4-6",
            "v1",
            frozenset({E.MIEDO, E.IRA}),
            "respiracion_4_6",
        ),
        CatalogDocument(
            "Anclaje de los 5 sentidos",
            "Nombra cinco cosas que ves, cuatro que tocas, tres que "
            "oyes, dos que hueles y una que saboreas. Esto trae la atención al presente.",
            DocumentType.TECNICA,
            "tec-grounding-5-sentidos",
            "v1",
            frozenset({E.MIEDO, E.SORPRESA}),
            "grounding_5_sentidos",
        ),
        CatalogDocument(
            "Reencuadre de pensamientos",
            "Escribe el pensamiento que más te pesa. Pregúntate qué "
            "evidencia tienes a favor y en contra, y escribe una versión más equilibrada.",
            DocumentType.TECNICA,
            "tec-reencuadre",
            "v1",
            frozenset({E.TRISTEZA, E.MIEDO}),
            "reencuadre_cognitivo",
        ),
        CatalogDocument(
            "Validación emocional",
            "Ponle nombre a lo que sientes sin juzgarlo: es normal sentirse "
            "así cuando algo importa. Permítete sentirlo y luego decide un pequeño paso.",
            DocumentType.TECNICA,
            "tec-validacion",
            "v1",
            frozenset({E.TRISTEZA, E.ASCO, E.NEUTRAL}),
            "validacion_emocional",
        ),
        CatalogDocument(
            "Plan en bloques pequeños",
            "Divide la tarea en bloques de 25 minutos con 5 de descanso. "
            "Empieza por el más fácil para ganar impulso.",
            DocumentType.TECNICA,
            "tec-bloques",
            "v1",
            frozenset({E.MIEDO, E.IRA, E.NEUTRAL}),
            "planificacion_bloques",
        ),
        CatalogDocument(
            "Bienestar Universitario UPTC Sogamoso",
            "Bienestar Universitario de la UPTC Sogamoso ofrece orientación psicológica a "
            "estudiantes. [Datos de contacto oficiales pendientes de carga por Bienestar.]",
            DocumentType.INSTITUCIONAL,
            "inst-bienestar",
            "v1",
        ),
    ]
