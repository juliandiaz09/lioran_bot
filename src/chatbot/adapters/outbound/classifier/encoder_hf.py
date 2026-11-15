"""Encoder de emociones (HuggingFace) con carga LOCAL y calibración por temperatura.

- El modelo ajustado vive en `ENCODER_MODEL_PATH` (por defecto `models/encoder/`, fuera de git). NUNCA se
  descarga de HuggingFace Hub en tiempo de ejecución (`local_files_only=True`).
- Para cambiar de modelo solo se reemplaza el contenido de esa carpeta y el JSON de temperatura.
- `torch` y `transformers` se importan dentro de `_load_model`: sin el extra `ml` el módulo se importa igual.
- Privacidad: no se registra el texto del usuario; solo su longitud y la duración.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chatbot.domain.models import Emotion, EmotionDistribution

logger = logging.getLogger(__name__)

#: Mapeo explícito etiqueta del modelo -> emoción del dominio. Una etiqueta fuera de esta tabla es un error.
LABEL_TO_EMOTION: dict[str, Emotion] = {
    "joy": Emotion.ALEGRIA,
    "sadness": Emotion.TRISTEZA,
    "anger": Emotion.IRA,
    "fear": Emotion.MIEDO,
    "surprise": Emotion.SORPRESA,
    "disgust": Emotion.ASCO,
    "others": Emotion.NEUTRAL,
    "other": Emotion.NEUTRAL,
    "neutral": Emotion.NEUTRAL,
}

InferFn = Callable[[str], Sequence[float]]


class EncoderError(Exception):
    """Error de carga o configuración del encoder."""


class EncoderModelNotFound(EncoderError):
    """No existe el modelo local del encoder."""


class HFEncoder:
    """`EmotionClassifierPort` sobre un modelo `AutoModelForSequenceClassification` local.

    Para pruebas sin modelo: `infer` (devuelve LOGITS de un texto) + `id2label`.
    """

    def __init__(
        self,
        model_path: str | Path = "models/encoder",
        temperature_file: str | Path | None = "config/encoder_temperature.json",
        *,
        device: str = "cpu",
        model_name: str = "",
        infer: InferFn | None = None,
        id2label: Mapping[int | str, str] | None = None,
    ) -> None:
        if infer is not None and not id2label:
            raise ValueError("Con una función `infer` inyectada hay que indicar también `id2label`")
        self._model_path = Path(model_path)
        self._temperature_file = Path(temperature_file) if temperature_file else None
        self._device = device
        self._model_name = model_name
        self._injected_infer = infer
        self._injected_id2label = dict(id2label) if id2label else None
        self._infer: InferFn | None = None
        self._emotion_by_index: list[Emotion] = []
        self._temperature = 1.0
        self._ready = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------------- ciclo de vida
    def is_ready(self) -> bool:
        return self._ready

    @property
    def temperature(self) -> float:
        return self._temperature

    async def warmup(self) -> None:
        """Carga única (idempotente y segura ante llamadas concurrentes)."""
        if self._ready:
            return
        async with self._lock:
            if not self._ready:
                await asyncio.to_thread(self._load)

    # ------------------------------------------------------------------------- EmotionClassifierPort
    async def classify(self, texto: str) -> EmotionDistribution:
        if not self._ready:
            await self.warmup()
        started = time.perf_counter()
        distribution = await asyncio.to_thread(self._predict, texto)
        logger.info(
            "encoder_clasificacion",
            extra={
                "longitud_texto": len(texto),
                "duracion_ms": round((time.perf_counter() - started) * 1000),
                "modelo": self._model_name,
            },
        )
        return distribution

    # ------------------------------------------------------------------------- carga
    def _load(self) -> None:
        if self._injected_infer is not None and self._injected_id2label is not None:
            infer: InferFn = self._injected_infer
            id2label: Mapping[int | str, str] = self._injected_id2label
            temperature = self._read_temperature()
        else:
            self._check_model_dir()
            temperature = self._read_temperature()
            infer, id2label = self._load_model()
        emotions = self._map_labels(id2label)  # falla ante una etiqueta desconocida
        self._infer, self._emotion_by_index, self._temperature = infer, emotions, temperature
        self._ready = True
        logger.info(
            "encoder_cargado",
            extra={
                "modelo": self._model_name,
                "ruta": str(self._model_path),
                "temperatura": temperature,
                "dispositivo": self._device,
            },
        )

    def _check_model_dir(self) -> None:
        # `models/encoder/` solo trae `.gitkeep` en un checkout limpio: exigimos `config.json`.
        if not self._model_path.is_dir() or not (self._model_path / "config.json").is_file():
            raise EncoderModelNotFound(f"No se encontró el modelo del encoder en {self._model_path}")

    def _read_temperature(self) -> float:
        path = self._temperature_file
        if path is None or not path.is_file():
            return 1.0
        try:
            value = json.loads(path.read_text(encoding="utf-8"))["temperature"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EncoderError(f"Archivo de temperatura inválido: {path}") from exc
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise EncoderError(f"La temperatura debe ser un número > 0 (archivo {path})")
        return float(value)

    @staticmethod
    def _map_labels(id2label: Mapping[int | str, str]) -> list[Emotion]:
        if not id2label:
            raise EncoderError("El modelo no declara etiquetas (id2label vacío)")
        by_index: dict[int, Emotion] = {}
        for raw_index, label in id2label.items():
            emotion = LABEL_TO_EMOTION.get(str(label).strip().lower())
            if emotion is None:
                raise EncoderError(f"Etiqueta desconocida del modelo: '{label}'")
            by_index[int(raw_index)] = emotion
        if sorted(by_index) != list(range(len(by_index))):
            raise EncoderError("Los índices de id2label deben ser consecutivos desde 0")
        return [by_index[i] for i in range(len(by_index))]

    def _load_model(self) -> tuple[InferFn, Mapping[int | str, str]]:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise EncoderError("ENCODER_BACKEND=hf requiere los extras de ML: pip install '.[ml]'") from exc

        path = str(self._model_path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        model.to(self._device)
        model.eval()

        max_length = getattr(tokenizer, "model_max_length", 512)
        if not isinstance(max_length, int) or not 0 < max_length <= 8192:  # tokenizers sin tope (1e30)
            max_length = 512
        device = self._device

        def infer(texto: str) -> list[float]:
            encoded: Any = tokenizer(texto, return_tensors="pt", truncation=True, max_length=max_length)
            batch = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**batch).logits[0]
            return [float(x) for x in logits.tolist()]

        return infer, dict(model.config.id2label)

    # ------------------------------------------------------------------------- inferencia
    def _predict(self, texto: str) -> EmotionDistribution:
        if self._infer is None:  # pragma: no cover - protegido por warmup
            raise EncoderError("El encoder no está cargado")
        logits = [float(x) for x in self._infer(texto)]
        if len(logits) != len(self._emotion_by_index):
            raise EncoderError(
                f"El modelo devolvió {len(logits)} logits y declara {len(self._emotion_by_index)} etiquetas"
            )
        if not all(math.isfinite(x) for x in logits):
            raise EncoderError("El modelo devolvió logits no finitos")
        scaled = [x / self._temperature for x in logits]  # calibración: softmax(logits / T)
        peak = max(scaled)
        exps = [math.exp(x - peak) for x in scaled]
        total = sum(exps)
        scores: dict[Emotion, float] = {}
        for emotion, e in zip(self._emotion_by_index, exps, strict=True):
            scores[emotion] = (
                scores.get(emotion, 0.0) + e / total
            )  # etiquetas que mapean a la misma emoción suman
        return EmotionDistribution.from_scores(scores)  # emociones ausentes = 0; renormaliza (suma 1)
