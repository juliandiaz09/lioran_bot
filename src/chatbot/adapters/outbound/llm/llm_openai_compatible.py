"""LLM por API compatible con OpenAI (`POST {base_url}/chat/completions`).

Privacidad: NUNCA se registra la llave ni el prompt (ni el texto del usuario). Solo modelo, duración,
código HTTP y número de intentos.

Reintentos HTTP: únicamente ante 429, 5xx, timeout y errores de conexión, con espera exponencial
(`backoff_base_s * 2**n`, con tope). Un 4xx distinto de 429 no se reintenta.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from chatbot.adapters.outbound.llm.errors import LLMInvalidResponse, LLMRequestRejected, LLMUnavailable
from chatbot.adapters.outbound.llm.prompt_loader import PromptLoader
from chatbot.application.dto import GenerationContext, LLMClassification
from chatbot.domain.models import EMOTION_ORDER, Emotion, EmotionDistribution, Message, RiskLevel, Role

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = "classify_emotion"
GENERATE_PROMPT = "generate_response"

_MAX_JUSTIFICATION = 200
_SUM_TOLERANCE = 1e-3
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)
_RETRYABLE_STATUS = frozenset({429}) | frozenset(range(500, 600))
_LT, _GT = "\u2039", "\u203a"  # ‹ ›


def neutralize(text: str) -> str:
    """Quita `<` y `>` del texto del usuario: así no puede cerrar ni abrir los delimitadores del prompt."""
    return text.replace("<", _LT).replace(">", _GT)


# --------------------------------------------------------------------------- JSON de la clasificación
class _ClassificationPayload(BaseModel):
    """Forma exacta que debe devolver el LLM al clasificar."""

    model_config = ConfigDict(extra="ignore")

    probabilidades: dict[str, float]
    riesgo: str
    justificacion_breve: str = ""

    @field_validator("probabilidades")
    @classmethod
    def _check_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        expected = {e.value for e in EMOTION_ORDER}
        if set(value) != expected:
            raise ValueError(f"debe traer exactamente las claves {sorted(expected)}")
        if any(not 0.0 <= p <= 1.0 for p in value.values()):
            raise ValueError("cada probabilidad debe estar entre 0 y 1")
        total = sum(value.values())
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise ValueError(f"las probabilidades deben sumar 1 (suman {total:.4f})")
        return value

    @field_validator("riesgo")
    @classmethod
    def _check_risk(cls, value: str) -> str:
        RiskLevel.parse(value)  # ValueError -> error de validación
        return value.strip().upper()

    @model_validator(mode="after")
    def _trim_justification(self) -> _ClassificationPayload:
        # Recortar, no fallar: un texto largo no invalida una clasificación correcta.
        self.justificacion_breve = self.justificacion_breve.strip()[:_MAX_JUSTIFICATION]
        return self

    def to_domain(self) -> LLMClassification:
        dist = EmotionDistribution({Emotion(k): v for k, v in self.probabilidades.items()})
        return LLMClassification(dist, RiskLevel.parse(self.riesgo), self.justificacion_breve)


def _strip_fences(raw: str) -> str:
    """Quita ```json ... ``` y, si hay charla extra, recorta de la primera `{` a la última `}`."""
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{") and "{" in text and "}" in text:
        text = text[text.index("{") : text.rindex("}") + 1]
    return text


def _describe_error(exc: Exception) -> str:
    """Resumen corto del fallo de validación para el reintento (sin eco del texto del usuario)."""
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(x) for x in e['loc']) or 'raíz'}: {e['msg']}" for e in exc.errors()]
        return "; ".join(parts)[:500]
    return str(exc)[:500]


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompts: PromptLoader,
        timeout_s: float = 20.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 8.0,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("El LLM requiere base_url y model")
        if max_retries < 0 or timeout_s <= 0:
            raise ValueError("max_retries debe ser >= 0 y timeout_s > 0")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._prompts = prompts
        self._max_retries = max_retries
        self._sleep = sleep
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._timeout_s = timeout_s
        # Fail-fast: sin prompts válidos el servicio no debe arrancar.
        prompts.get(CLASSIFY_PROMPT)
        prompts.get(GENERATE_PROMPT)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------------- LLMPort
    async def classify_emotion(self, texto: str, contexto: Sequence[Message]) -> LLMClassification:
        prompt = self._prompts.render(
            CLASSIFY_PROMPT,
            texto_usuario=neutralize(texto),
            historial=self._format_history(contexto),
        )
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        raw = await self._chat(messages, temperature=0.0)
        try:
            return self._parse_classification(raw)
        except (ValueError, ValidationError) as first_error:
            logger.info("llm_json_invalido_reintento", extra={"modelo": self._model})
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Tu respuesta anterior no cumple el formato: "
                        f"{_describe_error(first_error)}. "
                        "Devuelve SOLO el objeto JSON corregido, con las 7 probabilidades sumando 1."
                    ),
                },
            ]
        raw = await self._chat(messages, temperature=0.0)
        try:
            return self._parse_classification(raw)
        except (ValueError, ValidationError) as exc:
            raise LLMInvalidResponse(
                f"El LLM no devolvió un JSON válido tras reintentar: {_describe_error(exc)}"
            ) from exc

    async def generate(self, contexto: GenerationContext) -> str:
        prompt = self._prompts.render(
            GENERATE_PROMPT,
            texto_usuario=neutralize(contexto.texto_usuario),
            emocion=contexto.emocion.value if contexto.emocion else "desconocida",
            emociones_secundarias=", ".join(contexto.emociones_secundarias) or "ninguna",
            riesgo=contexto.riesgo.name,
            estrategia=self._format_strategy(contexto),
            fragmentos=self._format_chunks(contexto),
            historial=self._format_history(contexto.historial),
            correcciones=(
                "\n".join(f"- {c}" for c in contexto.correcciones) if contexto.correcciones else "(ninguna)"
            ),
        )
        text = (await self._chat([{"role": "user", "content": prompt}], temperature=0.4)).strip()
        if not text:
            raise LLMInvalidResponse("El LLM devolvió una respuesta vacía")
        return text

    # ------------------------------------------------------------------------- formato de prompt
    @staticmethod
    def _parse_classification(raw: str) -> LLMClassification:
        data = json.loads(_strip_fences(raw))
        if not isinstance(data, dict):
            raise ValueError("se esperaba un objeto JSON")
        return _ClassificationPayload.model_validate(data).to_domain()

    @staticmethod
    def _format_history(messages: Sequence[Message]) -> str:
        if not messages:
            return "(sin historial)"
        lines = []
        for m in messages:
            who = "usuario" if m.rol is Role.USER else "asistente"
            lines.append(f"[{who}] {neutralize(m.texto)}")
        return "\n".join(lines)

    @staticmethod
    def _format_strategy(ctx: GenerationContext) -> str:
        s = ctx.estrategia
        if s is None:
            return "ninguna (acompaña y valida lo que la persona siente)"
        return f"{s.nombre} ({s.tipo.value}): {s.descripcion}".strip()

    @staticmethod
    def _format_chunks(ctx: GenerationContext) -> str:
        if not ctx.fragmentos:
            return "(sin contexto)"
        return "\n".join(f"[{c.tipo} | {c.titulo}] {c.texto}" for c in ctx.fragmentos)

    # ------------------------------------------------------------------------- HTTP
    def _backoff(self, attempt: int) -> float:
        return float(min(self._backoff_cap_s, self._backoff_base_s * (2**attempt)))

    async def _chat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        body: dict[str, Any] = {"model": self._model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        started = time.perf_counter()
        attempts = self._max_retries + 1
        last_reason = "desconocido"
        for attempt in range(attempts):
            status: int | None = None
            try:
                response = await self._client.post(
                    self._url, json=body, headers=headers, timeout=self._timeout_s
                )
                status = response.status_code
                if status in _RETRYABLE_STATUS:
                    last_reason = f"http_{status}"
                elif status >= 400:
                    self._log_call(started, status, attempt + 1)
                    raise LLMRequestRejected(f"El proveedor rechazó la petición (HTTP {status})")
                else:
                    content = self._extract_content(response)
                    self._log_call(started, status, attempt + 1)
                    return content
            except httpx.TimeoutException:
                last_reason = "timeout"
            except httpx.TransportError as exc:
                last_reason = type(exc).__name__
            if attempt < attempts - 1:
                await self._sleep(self._backoff(attempt))
        self._log_call(started, None, attempts, failed=last_reason)
        raise LLMUnavailable(f"El LLM no respondió tras {attempts} intento(s): {last_reason}")

    @staticmethod
    def _extract_content(response: httpx.Response) -> str:
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponse("Respuesta del proveedor con formato inesperado") from exc
        if not isinstance(content, str):
            raise LLMInvalidResponse("El proveedor no devolvió texto en 'content'")
        return content

    def _log_call(self, started: float, status: int | None, attempts: int, failed: str | None = None) -> None:
        extra: dict[str, Any] = {
            "modelo": self._model,
            "duracion_ms": round((time.perf_counter() - started) * 1000),
            "http": status,
            "intentos": attempts,
        }
        if failed:
            extra["motivo"] = failed
        logger.info("llm_llamada", extra=extra)
