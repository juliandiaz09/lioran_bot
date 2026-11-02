# Backend — Chatbot de apoyo a la regulación emocional (UPTC Sogamoso · INFELCOM)

Arquitectura hexagonal en Python/FastAPI. Este paquete contiene las **fases 1, 2 y 3** del orden de trabajo
(sección 12 de la especificación):

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Esqueleto hexagonal, configuración por entorno, contenedor de dependencias, `/health` y `/ready` | ✅ |
| 2 | Dominio (modelos, servicios, errores) y sus pruebas | ✅ |
| 3 | Puertos, casos de uso y adaptadores falsos: flujo completo funcionando | ✅ |
| 4 | Detector de riesgo por reglas (léxico en archivo) y ruta de crisis | ⏭ siguiente |
| 5–8 | Adaptadores reales, bandido Thompson, ingesta RAG con catálogo semilla, API completa, Docker | pendiente |

## Arranque

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # ajustar llaves

uvicorn chatbot.adapters.inbound.api.main:app_factory --factory --reload
curl localhost:8000/health    # {"status":"ok"}
curl localhost:8000/ready
```

## Calidad (todo debe pasar)

```bash
pytest --cov          # 257 pruebas, cobertura domain+application 99,2 % (mínimo exigido: 80 %)
ruff check . && ruff format --check .
mypy src              # modo estricto en domain y application
lint-imports          # regla de dependencias hexagonal (2 contratos)
```

## Estructura

```
src/chatbot/
├── domain/        Emotion, EmotionDistribution, RiskLevel, Strategy, Message, Conversation,
│                  Feedback · FusionService, RiskPolicy, SecondaryEmotionMapper · errors
├── application/   dto, errors, templates · ports/{inbound,outbound} · use_cases/ (analyze_and_respond,
│                  register_feedback, ingest_documents, list_strategies)
├── adapters/
│   ├── inbound/api/   FastAPI: main, routers/health, dependencies (X-API-Key), error_handlers
│   └── outbound/      encoder_stub · llm_fake · chunker · embedder hash · validador por reglas ·
│                      detector de riesgo PROVISIONAL · memory/ (repos, RAG, catálogo, selector, seed)
├── config/        settings.py (pydantic-settings), container.py (raíz de composición)
└── shared/        ids, logging JSON
```

**Regla de dependencias:** `domain` no importa nada externo; `application` solo conoce `domain` y sus puertos.
Se verifica con `tests/architecture` (análisis AST) y con `import-linter` (contratos en `pyproject.toml`).

## Qué hace el dominio

- **`EmotionDistribution`**: inmutable; valida suma = 1 (tolerancia 1e-3); `top()`, `top_k(k)`, `entropy()` (bits),
  `normalized_entropy()` en [0,1]; desempate determinista.
- **`FusionService`**: `P_final = Σ w_m·P_m`. Si falta un modelo (LLM o encoder caído) renormaliza los pesos y degrada
  sin fallar. El **desacuerdo** es la divergencia de Jensen-Shannon generalizada, normalizada a [0,1].
- **`RiskPolicy`**: `max(reglas, LLM, historial)`; el nivel más alto siempre prevalece. LLM caído ⇒ no aporta.
- **`SecondaryEmotionMapper`**: reglas declarativas (umbrales sobre emociones primarias + palabras clave del texto),
  configurables por JSON (`SECONDARY_EMOTIONS_FILE`) o con las reglas por defecto (ansiedad, estrés, frustración,
  soledad, desánimo).
- **`Feedback.reward`**: convierte `util` (bool o escala 1–5) en recompensa [0,1] para el bandido.

## Fase 3: el flujo de `AnalyzeAndRespond`

Orden obligatorio de la sección 7.1: normalizar → riesgo por reglas (HIGH ⇒ crisis, **sin** encoder ni LLM) →
clasificación en paralelo (encoder ‖ LLM) + fusión → riesgo final (reglas + LLM + historial) → estrategia
(bandido) → RAG → generación → validación (1 regeneración y luego plantilla segura) → persistencia → resultado.

Resiliencia: cada llamada externa va con tiempo límite; si cae el LLM se usa el encoder, si cae el encoder
el LLM, si cae todo (o el RAG no devuelve contexto, o el validador rechaza dos veces) se responde con la
plantilla segura. Un fallo de persistencia se registra pero no impide responder. Los logs **no** incluyen
el texto del usuario (hay una prueba que lo verifica).

Demostración con el contenedor (sin red ni BD): ver `tests/unit/test_end_to_end_with_container.py`.
Los endpoints HTTP de estos casos de uso se exponen en una fase posterior; hoy solo `/health` y `/ready`.

## Decisiones tomadas donde la especificación no definía (marcadas como pendientes de revisión)

1. **Riesgo por historial**: máximo de la ventana reciente rebajado un escalón (HIGH→MEDIUM, MEDIUM→LOW, LOW→NONE).
   El historial eleva el riesgo del mensaje actual pero nunca llega a HIGH por sí solo. *Validar con psicología.*
2. **Desacuerdo entre modelos**: Jensen-Shannon generalizada normalizada (0 coinciden, 1 desacuerdo máximo).
3. **Reglas de emociones secundarias**: umbrales y palabras clave iniciales son un punto de partida, no validado clínicamente.
4. **`util`** admite bool o escala 1–5; la recompensa del bandido se normaliza a [0,1].
5. **`/ready`** solo reporta el dominio; verificación de BD y modelos llega con los adaptadores (Fase 5).
6. **Errores de dominio** responden 422 con `{"error": {"codigo", "mensaje"}}`.
7. **Ruta de crisis**: no clasifica emoción (para cumplir ≤ 2 s), por eso en esos casos `emocion_dominante` y
   `distribucion` son `null` y no hay estrategia. Si el riesgo sube a HIGH por el LLM, sí hay clasificación.
   Los contactos se leen del catálogo (`institutional_info`, solo filtro por metadatos); si no hay, se usa un
   texto sin teléfonos. **Ni la plantilla de crisis ni la segura están aprobadas** (`application/templates.py`).
8. **Sin contexto recuperado no se genera**: se usa la plantilla segura (el LLM solo puede apoyarse en el RAG).
9. **Feedback**: solo aplica a análisis con estrategia; un `trace_id` admite una sola retroalimentación.
10. **Campos extra en el resultado**: `degradado` y `modelos_usados` (útiles para monitoreo desde n8n).
11. **Detector de riesgo y selector de la Fase 3 son provisionales** (léxico mínimo embebido; selector por
    reglas). Se reemplazan en las fases 4 y 6. **No usar el léxico actual con estudiantes.**
12. **Seed (`adapters/outbound/memory/seed.py`)**: contenido de ejemplo sin revisión clínica y sin teléfonos.
13. **Dockerfile / docker-compose**: se entregan en la Fase 8, junto con el resto de adaptadores.

## Pendientes éticos (antes de usar con estudiantes reales)

- La ruta de crisis y su texto deben ser **aprobados por Bienestar Universitario / psicología**.
- Los teléfonos e información institucional saldrán del catálogo del RAG, nunca del LLM ni del código.
- El sistema no diagnostica ni reemplaza atención profesional (Ley 1581 de 2012; solo `user_ref` seudónimo).
