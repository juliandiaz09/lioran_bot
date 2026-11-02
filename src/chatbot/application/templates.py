"""Textos fijos del sistema.

⚠️ PENDIENTE DE APROBACIÓN por Bienestar Universitario / psicología antes de usarse con
estudiantes reales. Los datos de contacto NUNCA se escriben aquí: salen del catálogo
institucional (RAG) y se insertan en `{contactos}`.
"""

from __future__ import annotations

from collections.abc import Sequence

CRISIS_TEMPLATE = (
    "Siento mucho que estés pasando por algo tan difícil, y gracias por contármelo. "
    "Lo que sientes es importante y no tienes que enfrentarlo en soledad. "
    "Quiero que hables ahora mismo con una persona que pueda acompañarte.\n\n"
    "{contactos}\n\n"
    "Si sientes que estás en peligro inmediato, acude a urgencias o pídele a alguien "
    "cercano que se quede contigo. ¿Hay alguien contigo en este momento?"
)

CRISIS_NO_CONTACTS = (
    "Busca hoy mismo apoyo en Bienestar Universitario de tu sede o en los servicios de "
    "urgencias de tu ciudad."
)

SAFE_TEMPLATE = (
    "Gracias por contarme cómo te sientes; lo que vives importa. Ahora mismo no logro darte "
    "una sugerencia precisa, pero estoy aquí para escucharte. Si quieres, cuéntame un poco más "
    "sobre lo que está pasando. Y si lo necesitas, Bienestar Universitario puede acompañarte. "
    "¿Te sirvió esta respuesta?"
)


def render_crisis_response(contact_texts: Sequence[str]) -> str:
    contactos = "\n".join(contact_texts) if contact_texts else CRISIS_NO_CONTACTS
    return CRISIS_TEMPLATE.format(contactos=contactos)
