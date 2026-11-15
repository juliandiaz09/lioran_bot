# Backend — Chatbot de apoyo a la regulación emocional (UPTC Sogamoso · INFELCOM)

Arquitectura hexagonal en Python/FastAPI. Este paquete contiene las **fases 1 a 5** del orden de trabajo
(sección 12 de la especificación):

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Esqueleto hexagonal, configuración por entorno, contenedor de dependencias, `/health` y `/ready` | ✅ |
| 2 | Dominio (modelos, servicios, errores) y sus pruebas | ✅ |
| 3 | Puertos, casos de uso y adaptadores falsos: flujo completo funcionando | ✅ |
| 4 | Detector de riesgo por reglas (léxico en archivo) y ruta de crisis | ✅ |
| 5 | Adaptadores **reales** (encoder HF, LLM compatible con OpenAI, embedder + pgvector, Postgres/Supabase) elegibles solo por configuración; `/ready` real; migraciones | ✅ |
| 6 | Bandido Thompson y `POST /v1/feedback` | ⏭ siguiente |
| 7–8 | Endpoint de ingesta y catálogo semilla (7); Docker, latencias, README final (8) | pendiente |

## Arranque

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # núcleo + adaptadores falsos: no requiere nada más
pip install -e ".[dev,ml,db]"           # solo si usarás encoder HF / embedder real / Postgres
cp .env.example .env                    # ajustar llaves

uvicorn chatbot.adapters.inbound.api.main:app_factory --factory --reload
curl localhost:8000/health    # {"status":"ok"}
curl localhost:8000/ready
```

## Calidad (todo debe pasar)

```bash
pytest --cov          # 855 pruebas, cobertura domain+application 98 % (mínimo exigido: 80 %)
ruff check . && ruff format --check .
mypy src              # modo estricto en domain y application
lint-imports          # regla de dependencias hexagonal (2 contratos)
```

Las cuatro puertas pasan **sin instalar** los extras `ml` y `db`. Las pruebas de integración (BD real,
modelo real) están marcadas `@pytest.mark.integration` y se saltan solas si falta `TEST_DATABASE_URL` o el modelo:

```bash
TEST_DATABASE_URL=postgresql+asyncpg://usuario:clave@localhost/chatbot_test pytest -m integration
```

`TEST_DATABASE_URL` debe ser una base **desechable**: las pruebas aplican las migraciones y vacían las tablas.


## Fase 5: adaptadores reales (se eligen solo por configuración)

`domain` y `application` **no se tocaron**. Cada puerto tiene una fábrica en `config/container.py`:

| Variable | Falso / memoria (por defecto) | Real |
|---|---|---|
| `ENCODER_BACKEND` | `stub` → `StubEncoder` | `hf` → `HFEncoder` (modelo **local**, calibrado por temperatura) |
| `LLM_BACKEND` | `fake` → `FakeLLM` | `openai_compatible` → `OpenAICompatibleLLM` (`POST {LLM_BASE_URL}/chat/completions`) |
| `PERSISTENCE_BACKEND` | `memory` | `postgres` → repos SQL de conversación, feedback y eventos de crisis |
| `RETRIEVER_BACKEND` | `memory` → `InMemoryKnowledgeBase` + `HashEmbedder` | `pgvector` → `PgVectorRetriever` + `SentenceTransformerEmbedder` |

Con todo en `stub`/`fake`/`memory` el sistema se comporta exactamente como en la Fase 4.
El catálogo de estrategias sale **siempre** de `config/strategies.yaml` (`YamlStrategyCatalog`).

### Puesta en marcha con servicios reales

```bash
# 1. Base de datos (Supabase o Postgres con pgvector): aplicar el esquema, idempotente
python -m chatbot.scripts.migrate            # usa DATABASE_URL y EMBEDDING_DIM

# 2. Encoder: copiar el modelo ajustado a models/encoder/ (config.json, pesos, tokenizer) y,
#    opcionalmente, config/encoder_temperature.json con {"temperature": 1.0}. Sin cambios de código.

# 3. Arrancar (el warmup carga encoder y embedder; si falla, el servicio NO arranca)
uvicorn chatbot.adapters.inbound.api.main:app_factory --factory
curl localhost:8000/ready                    # 200 si todo está bien; 503 con el detalle por componente
```

`/ready` verifica de verdad: `SELECT 1`, extensión `vector`, `is_ready()` del encoder y del embedder, léxico,
plantillas y catálogo (≥ 1 estrategia). Con adaptadores falsos esos componentes se reportan `no_aplica`.
Un error de BD solo muestra el **tipo** de la excepción (nunca su mensaje, que podría traer datos de conexión).

### Nota Supabase: *pooler* en modo transacción (puerto 6543)

Con el pooler en modo transacción **asyncpg no puede usar sentencias preparadas**. El backend detecta el puerto
`6543` en `DATABASE_URL` y configura `statement_cache_size=0`, `prepared_statement_cache_size=0` y nombres de
sentencia únicos. En conexión directa (5432) no cambia nada. También acepta el formato `postgresql://` que
entrega Supabase y lo convierte a `postgresql+asyncpg://`. El backend se conecta con un **rol de servicio**
(que omite RLS): todas las tablas tienen RLS habilitado y **sin políticas**, de modo que `anon`/`authenticated`
no pueden leer ni escribir nada por la API REST de Supabase.

### Privacidad

- Solo `user_ref` seudónimo; nada de nombre ni usuario de Telegram.
- Los logs **nunca** llevan el texto del usuario, la llave del LLM ni el prompt (hay pruebas sobre cada adaptador).
- SQLAlchemy corre con `echo=False` y `hide_parameters=True`.
- `STORE_USER_TEXT=false`: `messages.texto = '[omitido]'` (la columna `longitud` guarda el largo real) y el
  comentario del feedback se guarda como `NULL`. La respuesta del asistente sí se guarda.

## Estructura

```
src/chatbot/
├── domain/        Emotion, EmotionDistribution, RiskLevel, Strategy, Message, Conversation,
│                  Feedback · FusionService, RiskPolicy, SecondaryEmotionMapper · errors
├── application/   dto, errors, templates · ports/{inbound,outbound} · use_cases/ (analyze_and_respond,
│                  register_feedback, ingest_documents, list_strategies)
├── adapters/
│   ├── inbound/api/   FastAPI: main, routers/health, dependencies (X-API-Key), error_handlers
│   └── outbound/      classifier/ (stub, hf) · llm/ (fake, openai_compatible, prompt_loader) ·
│                      rag/ (chunker, embedder hash + sentence-transformers, pgvector_retriever) ·
│                      persistence/ (database, repos SQL, eventos de crisis) · strategies/ (YAML) ·
│                      risk/ · validation/ · memory/ (repos y RAG en memoria, selector, seed)
├── config/        settings.py (pydantic-settings), container.py (raíz de composición)
├── scripts/       migrate.py (python -m chatbot.scripts.migrate)
└── shared/        ids, logging JSON
config/            risk_lexicon.yaml · strategies.yaml · templates/
migrations/        001_init.sql (esquema, HNSW, RLS)
prompts/           classify_emotion.md · generate_response.md (versionados)
models/encoder/    modelo ajustado del encoder (local, fuera de git)
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
5. **`/ready`** verifica de verdad BD, extensión `vector`, encoder, embedder, léxico, plantillas y catálogo (Fase 5).
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

## Decisiones de la Fase 5

1. **Ruta local del encoder** (`ENCODER_MODEL_PATH`, por defecto `models/encoder`): el modelo ajustado se inserta a
   mano y **no se descarga de HuggingFace** (`local_files_only=True`). `ENCODER_MODEL_NAME` es solo informativo.
   `warmup()` exige que exista `config.json` (en un checkout limpio la carpeta solo trae `.gitkeep`).
2. **`pgvector` + `memory` se rechaza al arrancar** (`Settings`), por simplicidad.
3. **`YamlStrategyCatalog` se usa siempre**, también con `memory`; `seed_strategies()` queda solo para pruebas
   (hay una prueba que exige que ambos coincidan).
4. **Reintentos del LLM**: solo ante 429, 5xx, timeout y errores de conexión; hasta `LLM_MAX_RETRIES` (2) con espera
   exponencial `0,5 s · 2ⁿ` (tope 8 s). Un 4xx distinto de 429 no se reintenta. Aparte, ante un JSON inválido hay
   **un único** reintento indicando el error de validación al modelo; si falla otra vez, `LLMInvalidResponse`.
5. **Neutralización de `<` y `>`** en el texto del usuario (y del historial) dentro del prompt: se reemplazan por
   `‹` y `›`, de modo que no pueden cerrar `</mensaje_usuario>`. El prompt declara ese contenido como *dato*.
6. **`bandit_state`** se crea en la migración pero **no se usa** hasta la Fase 6.
7. **`crisis_events`** ya se escribe en SQL (con el log WARNING de la Fase 4, emitido *antes* de tocar la BD). El
   aviso a humanos (webhook a n8n/Bienestar) sigue siendo una extensión futura.
8. **Esquema SQL diseñado a partir de la guía**: el «Apéndice B del prompt original» no estaba disponible, así que
   `001_init.sql` implementa lo que la guía especifica (tablas, HNSW, RLS, `vector(N)`, PK de feedback, `[omitido]`)
   y el resto se infiere de los DTO. **Conviene contrastarlo con el apéndice.** Añadí una columna `seq` en
   `messages` y `analyses` para ordenar de forma estable cuando dos timestamps coinciden.
9. **`STORE_USER_TEXT=false` también omite el comentario del feedback** (es texto libre del usuario). Consecuencia
   conocida: el historial que ve el LLM en turnos posteriores muestra `[omitido]` en los mensajes del usuario.
10. **El prompt se envía como un único mensaje `user`** (máxima compatibilidad entre proveedores compatibles con OpenAI);
    la defensa contra inyección es el delimitado + la neutralización + el validador de respuestas.
11. **No se usa `response_format=json_object`**: no todos los proveedores compatibles lo soportan. Se pide JSON por
    prompt y se tolera ```` ```json ````; la validación estricta es de Pydantic.
12. **El paquete `pgvector` de Python no se usa**: los vectores viajan como literal de texto (`CAST(:v AS vector)`).
    Se mantiene en el extra `db` porque así lo pide la guía; puede retirarse sin efecto.
13. **`Container.readiness()`** sigue siendo síncrono (compatible con las pruebas existentes) y
    `Container.check_readiness()` añade las verificaciones de BD; `/ready` usa esta última y responde 503 si algo falla.
14. **`httpx` pasó a las dependencias base** (el adaptador LLM lo necesita en ejecución).

## Pendientes éticos (antes de usar con estudiantes reales)

- La ruta de crisis y su texto deben ser **aprobados por Bienestar Universitario / psicología**.
- Los teléfonos e información institucional saldrán del catálogo del RAG, nunca del LLM ni del código.
- El sistema no diagnostica ni reemplaza atención profesional (Ley 1581 de 2012; solo `user_ref` seudónimo).
