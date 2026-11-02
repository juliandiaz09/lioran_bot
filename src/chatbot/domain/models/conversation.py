"""Mensajes y conversaciones. `user_ref` es SIEMPRE un identificador seudónimo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from chatbot.domain.errors import InvalidConversation

MAX_USER_REF_LEN = 128


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    id: str
    conversation_id: str
    texto: str
    rol: Role
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.id or not self.conversation_id:
            raise InvalidConversation("El mensaje requiere id y conversation_id")
        if not self.texto.strip():
            raise InvalidConversation("El mensaje no puede estar vacío")


@dataclass(slots=True)
class Conversation:
    id: str
    user_ref: str
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidConversation("La conversación requiere id")
        if not self.user_ref.strip() or len(self.user_ref) > MAX_USER_REF_LEN:
            raise InvalidConversation(
                f"user_ref debe ser un seudónimo no vacío de máximo {MAX_USER_REF_LEN} caracteres"
            )

    def add_message(self, message: Message) -> None:
        if message.conversation_id != self.id:
            raise InvalidConversation("El mensaje pertenece a otra conversación")
        self.messages.append(message)

    def recent(self, n: int) -> list[Message]:
        if n < 1:
            raise ValueError("n debe ser >= 1")
        return self.messages[-n:]
