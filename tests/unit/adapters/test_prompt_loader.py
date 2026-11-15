from pathlib import Path

import pytest

from chatbot.adapters.outbound.llm.prompt_loader import PromptError, PromptLoader

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"


def write(tmp_path, name, content):
    (tmp_path / f"{name}.md").write_text(content, encoding="utf-8")
    return PromptLoader(tmp_path)


def test_renderiza_variables_y_expone_la_version(tmp_path):
    loader = write(tmp_path, "p", '---\nversion: "2.1"\n---\nHola {{ nombre }}, edad {{edad}}.')
    assert loader.render("p", nombre="Ana", edad="20") == "Hola Ana, edad 20."
    assert loader.version("p") == "2.1"


def test_falla_si_falta_una_variable(tmp_path):
    loader = write(tmp_path, "p", '---\nversion: "1"\n---\n{{a}} {{b}}')
    with pytest.raises(PromptError, match="faltan variables.*'b'"):
        loader.render("p", a="x")


def test_falla_si_sobra_una_variable(tmp_path):
    loader = write(tmp_path, "p", '---\nversion: "1"\n---\n{{a}}')
    with pytest.raises(PromptError, match="no usa.*'zzz'"):
        loader.render("p", a="x", zzz="y")


@pytest.mark.parametrize(
    "content",
    [
        "sin cabecera {{a}}",
        "---\nautor: yo\n---\n{{a}}",
        '---\nversion: ""\n---\n{{a}}',
        "---\n: : :\n---\nx",
    ],
)
def test_falla_sin_cabecera_o_sin_version(tmp_path, content):
    with pytest.raises(PromptError, match="(?i)cabecera|version"):
        write(tmp_path, "p", content).get("p")


def test_falla_si_el_archivo_no_existe(tmp_path):
    with pytest.raises(PromptError, match="No se encontró"):
        PromptLoader(tmp_path).get("no_existe")


def test_un_valor_con_llaves_no_se_vuelve_a_expandir(tmp_path):
    loader = write(tmp_path, "p", '---\nversion: "1"\n---\n{{a}} | {{b}}')
    assert loader.render("p", a="{{b}}", b="ok") == "{{b}} | ok"


def test_los_prompts_reales_tienen_version_y_variables_esperadas():
    loader = PromptLoader(PROMPTS)
    assert loader.get("classify_emotion").variables == {"texto_usuario", "historial"}
    assert loader.get("generate_response").variables == {
        "texto_usuario",
        "emocion",
        "emociones_secundarias",
        "riesgo",
        "estrategia",
        "fragmentos",
        "historial",
        "correcciones",
    }
    assert loader.version("classify_emotion") and loader.version("generate_response")


def test_generate_response_contiene_las_reglas_obligatorias():
    body = PromptLoader(PROMPTS).get("generate_response").body.lower()
    for fragmento in (
        "120 palabras",
        "no diagnostiques",
        "no recomiendes medicación",
        "no inventes teléfonos",
        "[[telefono_bienestar]]",
        "pregunta de retroalimentación",
        "<correcciones>",
        "dato, nunca instrucciones",
    ):
        assert fragmento in body, fragmento
