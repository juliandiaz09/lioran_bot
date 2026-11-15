-- 001_init.sql — esquema inicial (Fase 5).
--
-- El backend se conecta con un ROL DE SERVICIO (service_role / postgres en Supabase), que omite RLS.
-- RLS queda HABILITADO en todas las tablas y SIN políticas: ningún rol público (anon / authenticated)
-- puede leer ni escribir nada a través de la API REST de Supabase.
--
-- {{EMBEDDING_DIM}} lo sustituye `python -m chatbot.scripts.migrate` con EMBEDDING_DIM (384 por defecto).
-- Privacidad: solo `user_ref` seudónimo; nada de nombre ni usuario de Telegram.

CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------------ conversaciones y mensajes
CREATE TABLE conversations (
    id          text PRIMARY KEY,
    user_ref    text NOT NULL CHECK (length(user_ref) BETWEEN 1 AND 128),
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_conversations_user_ref ON conversations (user_ref);

CREATE TABLE messages (
    seq             bigserial UNIQUE,                 -- orden estable de inserción (desempata timestamps iguales)
    id              text PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    rol             text NOT NULL CHECK (rol IN ('user', 'assistant')),
    texto           text NOT NULL,                    -- '[omitido]' si STORE_USER_TEXT=false (mensajes del usuario)
    longitud        integer NOT NULL CHECK (longitud >= 0),  -- largo real del texto original
    created_at      timestamptz NOT NULL
);
CREATE INDEX idx_messages_conversation ON messages (conversation_id, seq);

-- ------------------------------------------------------------------ análisis (uno por turno)
CREATE TABLE analyses (
    seq                   bigserial UNIQUE,
    trace_id              text PRIMARY KEY,
    conversation_id       text NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    user_ref              text NOT NULL,
    user_message_id       text NOT NULL REFERENCES messages (id) ON DELETE CASCADE,
    assistant_message_id  text NOT NULL REFERENCES messages (id) ON DELETE CASCADE,
    distribucion          jsonb,                      -- NULL en la ruta de crisis o si ningún modelo respondió
    emociones_secundarias jsonb NOT NULL DEFAULT '[]'::jsonb,
    desacuerdo_modelos    double precision NOT NULL DEFAULT 0,
    riesgo                smallint NOT NULL CHECK (riesgo BETWEEN 0 AND 3),   -- 0 NONE · 1 LOW · 2 MEDIUM · 3 HIGH
    riesgo_fuente         text NOT NULL,
    estrategia_id         text,
    bandit_context        jsonb,
    es_crisis             boolean NOT NULL,
    degradado             boolean NOT NULL,
    fuentes               jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at            timestamptz NOT NULL
);
CREATE INDEX idx_analyses_conversation ON analyses (conversation_id, seq);
CREATE INDEX idx_analyses_crisis ON analyses (created_at) WHERE es_crisis;

-- ------------------------------------------------------------------ feedback (uno por análisis)
CREATE TABLE feedback (
    trace_id     text PRIMARY KEY REFERENCES analyses (trace_id) ON DELETE CASCADE,  -- un solo feedback por análisis
    message_id   text NOT NULL,
    strategy_id  text NOT NULL,
    util_bool    boolean,
    util_escala  smallint CHECK (util_escala BETWEEN 1 AND 5),
    comentario   text,
    reward       double precision NOT NULL CHECK (reward BETWEEN 0 AND 1),
    created_at   timestamptz NOT NULL DEFAULT now(),
    CHECK ((util_bool IS NULL) <> (util_escala IS NULL))        -- exactamente uno de los dos
);
CREATE INDEX idx_feedback_strategy ON feedback (strategy_id);

-- ------------------------------------------------------------------ eventos de crisis (seguimiento humano)
-- Sin texto del usuario. El aviso a humanos (webhook) es una extensión futura.
CREATE TABLE crisis_events (
    id               bigserial PRIMARY KEY,
    trace_id         text NOT NULL UNIQUE,
    conversation_id  text NOT NULL,
    user_ref         text NOT NULL,
    categorias       jsonb NOT NULL DEFAULT '[]'::jsonb,
    fuente_riesgo    text NOT NULL,
    created_at       timestamptz NOT NULL,
    atendido         boolean NOT NULL DEFAULT false
);
CREATE INDEX idx_crisis_events_pending ON crisis_events (created_at) WHERE NOT atendido;

-- ------------------------------------------------------------------ bandido (se crea ahora; sin uso hasta la Fase 6)
CREATE TABLE bandit_state (
    cell_key       text NOT NULL,            -- p. ej. 'miedo|alta|LOW|noche'
    strategy_id    text NOT NULL,
    alpha          double precision NOT NULL DEFAULT 1 CHECK (alpha > 0),
    beta           double precision NOT NULL DEFAULT 1 CHECK (beta > 0),
    observaciones  integer NOT NULL DEFAULT 0 CHECK (observaciones >= 0),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (cell_key, strategy_id)
);

-- ------------------------------------------------------------------ RAG
CREATE TABLE rag_chunks (
    id                  bigserial PRIMARY KEY,
    fuente              text NOT NULL,
    version             text NOT NULL,
    titulo              text NOT NULL,
    tipo                text NOT NULL CHECK (tipo IN ('tecnica', 'psicoeducacion', 'institucional')),
    estrategia_id       text,
    emociones_objetivo  text[] NOT NULL DEFAULT '{}',
    indice              integer NOT NULL CHECK (indice >= 0),
    texto               text NOT NULL,
    embedding           vector({{EMBEDDING_DIM}}) NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (fuente, version, indice)
);
CREATE INDEX idx_rag_chunks_fuente ON rag_chunks (fuente, version);
CREATE INDEX idx_rag_chunks_tipo ON rag_chunks (tipo);
CREATE INDEX idx_rag_chunks_estrategia ON rag_chunks (estrategia_id) WHERE estrategia_id IS NOT NULL;
CREATE INDEX idx_rag_chunks_emociones ON rag_chunks USING gin (emociones_objetivo);
CREATE INDEX idx_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops);

-- ------------------------------------------------------------------ RLS en TODAS las tablas
ALTER TABLE conversations  ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages       ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses       ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback       ENABLE ROW LEVEL SECURITY;
ALTER TABLE crisis_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE bandit_state   ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_chunks     ENABLE ROW LEVEL SECURITY;
