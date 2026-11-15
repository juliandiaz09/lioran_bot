"""HFEncoder sin descargar ni cargar modelos reales (inferencia inyectada o módulos falsos)."""

import asyncio
import contextlib
import json
import logging
import math
import sys
import types

import pytest

from chatbot.adapters.outbound.classifier.encoder_hf import (
    EncoderError,
    EncoderModelNotFound,
    HFEncoder,
)
from chatbot.domain.models import Emotion

ID2LABEL = {0: "others", 1: "joy", 2: "sadness", 3: "anger", 4: "surprise", 5: "disgust", 6: "fear"}
#            others  joy   sad   anger surp  disg  fear
LOGITS = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 3.0]


def encoder(tmp_path=None, logits=LOGITS, id2label=None, temperature=None, **kw):
    tfile = None
    if temperature is not None:
        tfile = tmp_path / "t.json"
        tfile.write_text(json.dumps(temperature) if not isinstance(temperature, str) else temperature)
    return HFEncoder(
        model_path="models/no-existe",
        temperature_file=tfile,
        infer=lambda _t: logits,
        id2label=id2label or ID2LABEL,
        **kw,
    )


# ------------------------------------------------------------------ mapeo de etiquetas
async def test_mapea_etiquetas_incluyendo_others_a_neutral():
    dist = await encoder().classify("tengo miedo")
    assert dist.top() is Emotion.MIEDO
    assert dist[Emotion.NEUTRAL] > 0 and dist[Emotion.ALEGRIA] > 0 and dist[Emotion.ASCO] > 0
    assert math.isclose(sum(dist.probabilities.values()), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("neutral_label", ["others", "other", "neutral", "NEUTRAL", " Other "])
async def test_others_other_y_neutral_son_neutral(neutral_label):
    enc = encoder(logits=[5.0, 0.0], id2label={0: neutral_label, 1: "joy"})
    assert (await enc.classify("x")).top() is Emotion.NEUTRAL


async def test_etiqueta_desconocida_falla_en_warmup_con_su_nombre():
    enc = encoder(id2label={**ID2LABEL, 6: "LABEL_6"})
    with pytest.raises(EncoderError, match="LABEL_6"):
        await enc.warmup()
    assert not enc.is_ready()


async def test_emociones_ausentes_reciben_cero_y_se_renormaliza():
    enc = encoder(logits=[1.0, 2.0, 3.0], id2label={0: "joy", 1: "sadness", 2: "fear"})
    dist = await enc.classify("x")
    assert dist[Emotion.IRA] == 0.0 and dist[Emotion.NEUTRAL] == 0.0
    assert math.isclose(sum(dist.probabilities.values()), 1.0, abs_tol=1e-9)


async def test_dos_etiquetas_a_la_misma_emocion_suman():
    enc = encoder(logits=[0.0, 0.0, 10.0], id2label={0: "others", 1: "neutral", 2: "joy"})
    dist = await enc.classify("x")
    assert math.isclose(dist[Emotion.NEUTRAL], 1 - dist[Emotion.ALEGRIA], rel_tol=1e-9)


async def test_acepta_id2label_con_claves_texto_como_las_del_config_json():
    enc = encoder(logits=[0.0, 9.0], id2label={"0": "others", "1": "joy"})
    assert (await enc.classify("x")).top() is Emotion.ALEGRIA


@pytest.mark.parametrize("bad", [{}, {0: "joy", 2: "fear"}])
async def test_id2label_vacio_o_con_huecos_falla(bad):
    enc = HFEncoder(infer=lambda _t: [0.0], id2label={0: "joy"})
    enc._injected_id2label = bad
    with pytest.raises(EncoderError):
        await enc.warmup()


async def test_cantidad_de_logits_distinta_a_la_de_etiquetas_falla():
    with pytest.raises(EncoderError, match="logits"):
        await encoder(logits=[1.0, 2.0]).classify("x")


async def test_logits_no_finitos_fallan():
    with pytest.raises(EncoderError, match="no finitos"):
        await encoder(logits=[float("nan")] * 7).classify("x")


def test_infer_inyectado_exige_id2label():
    with pytest.raises(ValueError):
        HFEncoder(infer=lambda _t: [0.0])


# ------------------------------------------------------------------ temperatura
async def test_temperatura_mayor_que_uno_aplana_la_distribucion(tmp_path):
    base = await encoder(tmp_path).classify("x")
    suave = await encoder(tmp_path, temperature={"temperature": 3.0}).classify("x")
    fuerte = await encoder(tmp_path, temperature={"temperature": 0.5}).classify("x")
    assert suave.entropy() > base.entropy() > fuerte.entropy()
    assert suave[Emotion.MIEDO] < base[Emotion.MIEDO] < fuerte[Emotion.MIEDO]
    assert suave.top() is base.top() is fuerte.top()


async def test_sin_archivo_de_temperatura_usa_uno(tmp_path):
    enc = HFEncoder(temperature_file=tmp_path / "no-existe.json", infer=lambda _t: LOGITS, id2label=ID2LABEL)
    await enc.warmup()
    assert enc.temperature == 1.0


async def test_lee_la_temperatura_del_json(tmp_path):
    enc = encoder(tmp_path, temperature={"temperature": 1.7})
    await enc.warmup()
    assert enc.temperature == 1.7


@pytest.mark.parametrize(
    "content",
    [
        {"temperature": 0},
        {"temperature": -1.0},
        {"temperature": "alta"},
        {"temperature": True},
        {"t": 1},
        "{no es json",
    ],
)
async def test_temperatura_invalida_falla_al_cargar(tmp_path, content):
    with pytest.raises(EncoderError, match="(?i)temperatura"):
        await encoder(tmp_path, temperature=content).warmup()


# ------------------------------------------------------------------ carga y ciclo de vida
async def test_falta_la_carpeta_del_modelo_da_error_claro(tmp_path):
    enc = HFEncoder(model_path=tmp_path / "models" / "encoder", temperature_file=None)
    with pytest.raises(EncoderModelNotFound, match="No se encontró el modelo del encoder en .*encoder"):
        await enc.warmup()
    assert not enc.is_ready()


async def test_carpeta_con_solo_gitkeep_tambien_da_error_claro(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    with pytest.raises(EncoderModelNotFound, match="No se encontró el modelo del encoder"):
        await HFEncoder(model_path=tmp_path, temperature_file=None).warmup()


async def test_classify_sin_modelo_propaga_el_error_claro(tmp_path):
    with pytest.raises(EncoderModelNotFound):
        await HFEncoder(model_path=tmp_path / "x", temperature_file=None).classify("hola")


async def test_is_ready_solo_tras_cargar_y_classify_carga_perezosamente():
    enc = encoder()
    assert not enc.is_ready()
    await enc.classify("hola")
    assert enc.is_ready()
    await enc.warmup()  # idempotente
    assert enc.is_ready()


async def test_carga_unica_bajo_llamadas_concurrentes(monkeypatch):
    enc = encoder()
    calls = 0
    original = enc._load

    def slow_load():
        nonlocal calls
        calls += 1
        import time

        time.sleep(0.05)
        original()

    monkeypatch.setattr(enc, "_load", slow_load)
    results = await asyncio.gather(*(enc.classify(f"t{i}") for i in range(10)))
    assert calls == 1 and len(results) == 10


async def test_la_inferencia_no_bloquea_el_event_loop():
    import time

    enc = HFEncoder(infer=lambda _t: (time.sleep(0.2), LOGITS)[1], id2label=ID2LABEL, temperature_file=None)
    await enc.warmup()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await enc.classify("x")
    task.cancel()
    assert ticks >= 5  # el loop siguió corriendo mientras el modelo "pensaba"


# ------------------------------------------------------------------ privacidad
async def test_no_registra_el_texto_solo_longitud_y_duracion(caplog):
    caplog.set_level(logging.DEBUG)
    secreto = "texto-privado-del-estudiante"
    await encoder(model_name="robertuito-x").classify(secreto)
    volcado = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert secreto not in volcado
    assert f"'longitud_texto': {len(secreto)}" in volcado and "duracion_ms" in volcado


# ------------------------------------------------------------------ camino real con módulos falsos
class _Tensor:
    def __init__(self, values):
        self.values = values

    def to(self, device):
        self.device = device
        return self

    def tolist(self):
        return list(self.values)

    def __getitem__(self, i):
        return _Tensor(self.values[i])


def _install_fake_ml(monkeypatch, model_dir, *, id2label, tokenizer_max=512):
    seen: dict = {}

    class Tok:
        model_max_length = tokenizer_max

        @classmethod
        def from_pretrained(cls, path, **kw):
            seen["tok"] = (path, kw)
            return cls()

        def __call__(self, texto, **kw):
            seen["tok_call"] = kw
            return {"input_ids": _Tensor([1, 2, 3])}

    class Model:
        config = types.SimpleNamespace(id2label=id2label)

        @classmethod
        def from_pretrained(cls, path, **kw):
            seen["model"] = (path, kw)
            return cls()

        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            seen["eval"] = True

        def __call__(self, **batch):
            return types.SimpleNamespace(logits=_Tensor([LOGITS]))

    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = Tok
    transformers.AutoModelForSequenceClassification = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return seen


async def test_camino_real_carga_local_sin_red_y_usa_id2label_del_modelo(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    seen = _install_fake_ml(monkeypatch, tmp_path, id2label=ID2LABEL)
    enc = HFEncoder(model_path=tmp_path, temperature_file=None, device="cpu", model_name="m")
    dist = await enc.classify("Tengo miedo")
    assert dist.top() is Emotion.MIEDO
    assert seen["tok"][1] == {"local_files_only": True} and seen["model"][1] == {"local_files_only": True}
    assert seen["tok_call"] == {"return_tensors": "pt", "truncation": True, "max_length": 512}
    assert seen["device"] == "cpu" and seen["eval"] is True


async def test_camino_real_tokenizer_sin_tope_usa_512(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    seen = _install_fake_ml(monkeypatch, tmp_path, id2label=ID2LABEL, tokenizer_max=int(1e30))
    await HFEncoder(model_path=tmp_path, temperature_file=None).classify("x")
    assert seen["tok_call"]["max_length"] == 512


async def test_camino_real_etiqueta_desconocida_en_el_modelo_falla(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    _install_fake_ml(monkeypatch, tmp_path, id2label={**ID2LABEL, 6: "LABEL_6"})
    with pytest.raises(EncoderError, match="LABEL_6"):
        await HFEncoder(model_path=tmp_path, temperature_file=None).warmup()


async def test_sin_los_extras_ml_el_error_indica_como_instalarlos(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "torch", None)  # fuerza ImportError
    with pytest.raises(EncoderError, match=r"\.\[ml\]"):
        await HFEncoder(model_path=tmp_path, temperature_file=None).warmup()
