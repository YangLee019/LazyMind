# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LazyMind is an enterprise RAG knowledge-base platform with a built-in self-evolution loop (evo). It ingests documents from local files, Feishu, and other sources, serves RAG-powered chat, and can automatically evaluate RAG quality, analyze bad cases, generate code fixes, and run A/B tests.

## Build, Lint, and Test

```bash
# Start the full stack
make up               # start services (add LAZYMIND_DEPLOY_MINERU=1 for PDF parsing)
make up-build         # rebuild images then start

# Stop / clean
make down             # stop services
make clear            # stop + remove volumes + clear Python cache
make reset-kb         # wipe knowledge-base data only (Milvus, OpenSearch, uploads, KB tables)
make reset-all        # wipe ALL persistent data (equivalent to fresh first-run)
make fresh-start      # reset-kb + restart with LAZYMIND_RESET_ALGO_ON_STARTUP=true

# Lint
make lint             # Python (flake8) + Go (gofmt)
make lint-only-diff   # lint changed files only

# Test
make test             # run all tests via tests/run-all.sh (frontend + auth-service + core + algorithm)
# Or run individual suites:
python3 -m pytest tests/algorithm/ -v --tb=short       # algorithm tests only
python3 -m pytest tests/backend/auth-service/ -v --tb=short  # auth-service tests
(cd tests/backend/core && go test ./... -v)             # Go core tests
# Frontend tests:
(cd tests/frontend && npm install --silent && npm test)
```

**pytest config**: `pytest.ini` excludes `tests/algorithm/evo` by default. Evo tests use their own harness.

## Architecture

### Service Dependency Chain

```
db → auth-service → Kong → frontend
db → core (Go, :8000)
db → processor-server → processor-worker → parsing → chat
parsing → Milvus + OpenSearch
evo-api (FastAPI, :8047) — standalone self-evolution service
```

**Request auth chain** (4 layers): Frontend gets JWT from auth-service → Kong validates JWT + RBAC → Core enforces ACL (resource-level, e.g. kb_id) → Algorithm services

### Key Services and Technologies

| Service | Stack | Port | Purpose |
|---------|-------|------|---------|
| frontend | React 18 + Vite + TypeScript + Ant Design + pnpm | 8090 | SPA with modules: chat, dataSource, datasetManagement, knowledge, memory, modelProvider, selfEvolution, admin, signin |
| kong | Kong API Gateway | 8000 | RBAC, JWT validation, routes to auth-service/core/chat |
| auth-service | FastAPI (Python) | — | JWT auth, user/role/group management, permission checks |
| core | Go (gorilla/mux) | 8000 | Main HTTP API: datasets, documents, tasks, chat, agents, conversations, skills, memory, preferences, model providers, prompts, ACL, eval sets |
| chat | Python (lazyllm-based) | — | RAG chat with multi-path retrieval + reranking |
| parsing | Python (lazyllm-based) | — | Document parsing: PDFReader / MinerU / PaddleOCR-VL, vectorization |
| evo-api | FastAPI (Python) | 8047 | Self-evolution loop: dataset_gen → eval → analyze → code fix → A/B test |
| lazyllm-algo | Python (lazyllm) | — | Algorithm/knowledge-base service |
| processor-server/worker | Python | — | Document task queue with lease-based worker model |
| db | PostgreSQL | — | Main database (core + app schemas) |
| redis | Redis | — | Token store, rate limiting, conversation cache |
| milvus | Milvus | 19530 | Vector store (optional built-in or external) |
| opensearch | OpenSearch | 9200 | Segment store (optional built-in or external) |

### Source Code Layout

```
algorithm/
├── lazyllm/          # upstream RAG framework (git submodule, treat as read-only)
├── lazymind/         # custom algorithms on top of lazyllm
│   ├── chat/         #   RAG chat service (api/, engine/, service/)
│   ├── parsing/      #   document parsing (readers, transform, OCR engines)
│   ├── processor/    #   document task queue
│   ├── common/       #   shared model configs, resources
│   ├── review/       #   background review/evolution suggestions
│   ├── rewrite/      #   query rewriting
│   ├── router/       #   query routing
│   └── config.py     #   centralized config via lazyllm Config (all LAZYMIND_* env vars)
├── Dockerfile        # multi-stage: base_env → base_code → algorithm/doc/evo/mineru
└── requirements.txt  # Python 3.11+ dependencies

backend/
├── core/             # Go HTTP API (module lazymind/core, Go 1.24)
│   ├── main.go       #   entrypoint, ACL init, route registration, OpenAPI export
│   ├── routes.go     #   all REST endpoints with permission declarations
│   ├── acl/          #   resource-level ACL (kb-level permissions)
│   ├── store/        #   DB + Redis connection, conversation/prompt store
│   ├── doc/          #   dataset, document, task, upload handling
│   ├── evalset/      #   evaluation set CRUD + import/export
│   ├── chat/         #   chat, conversation, prompt endpoints
│   ├── agent/        #   agent thread orchestration (evo integration)
│   ├── skill/        #   skill CRUD, sharing, suggestion callbacks
│   ├── memory/       #   memory management, suggestion callbacks
│   ├── preference/   #   user preference management
│   ├── evolution/    #   evolution suggestions + personalization
│   ├── modelprovider/#   model provider/group/key management
│   ├── wordgroup/    #   word group (vocabulary) management
│   ├── file/         #   RAG file upload/group endpoints
│   └── common/       #   shared types, ORM, reply helpers
├── auth-service/     # FastAPI: JWT auth, user/role/group management, RBAC
├── file-watcher/     # Go file-watching agent (scans host dirs for new documents)
├── scan-control-plane/# file-watcher control plane
├── office-convert-service/  # Office document format conversion
└── scripts/          # DB migration scripts, API permission extraction

evo/                  # Self-evolution loop (standalone FastAPI service)
├── main.py           #   CLI + pipeline entrypoint
├── harness/          #   pipeline orchestration (plan, react, analysis, clustering)
├── agents/           #   LLM agents (researcher, critic, synthesizer, indexer, etc.)
├── conductor/        #   orchestrator engine (handle store, world model)
├── orchestrator/     #   thread management, LLM-based orchestration
├── apply/            #   code fix generation + testing (opencode integration)
├── abtest/           #   A/B test runner + comparator
├── runtime/          #   config, sessions, telemetry, model gateway
├── tools/            #   analysis tools (code, data, report, evidence, clustering)
├── service/          #   HTTP service entrypoint
├── datagen/          #   dataset generation
└── domain/           #   domain types

frontend/             # React SPA (Vite + TypeScript + pnpm + Ant Design)
├── src/modules/      #   feature modules: chat, dataSource, datasetManagement,
│                     #     knowledge, memory, modelProvider, selfEvolution, admin, signin
├── src/api/generated/#   OpenAPI-generated TypeScript clients (authservice, core, chatbot, etc.)
└── src/components/   #   shared UI components

api/                  # Centralized OpenAPI specs (auth-service, core) — keep in sync
tests/                # Tests mirror source layout (algorithm/, backend/, frontend/, evo/)
docs/                 # Architecture, quick start, CLI docs, Feishu integration
data/                 # Runtime data directory (scanned files, uploads, evo outputs)
```

### Key Architectural Patterns

**Configuration**: All algorithm services use `LAZYMIND_MODEL_CONFIG_PATH`. Default is `dynamic` — per-user model/API-key from the frontend is injected per request. Static configs: `online` (public cloud) or `inner` (intranet). Python config managed via `algorithm/lazymind/config.py` using lazyllm's `Config` class with `LAZYMIND_` prefix.

**Model selection**: Users configure LLM, VLM, embed (up to 3), cross_embed, and reranker models via the frontend. Single-embedding mode auto-activates when only `embed_1` is configured.

**OCR routing**: Selected per-request via the model provider UI (`DynamicPDFReader`). Three tiers: built-in PDFReader, MinerU (on-prem or API), PaddleOCR-VL (GPU).

**Mirror profiles**: Build-time source URLs are selected via `MIRROR_PROFILE` (`cn` for domestic/Aliyun, `intl` for international). Set in `.env` or via `make up MIRROR_PROFILE=intl`.

**OpenAPI workflow**: Both auth-service (FastAPI) and core (Go) auto-generate OpenAPI specs at startup and export them to `api/`. The Go core uses a custom router-based spec builder (`openapi_gen.go`) plus manual annotations (`openapi_manual.go`). Frontend API clients are generated from these specs via `npm run gen:openapi`.

**Permission model**: Each Go route handler is registered with required permissions (e.g., `["document.read"]`, `["qa.write"]`) via `handleAPI()`. Core permissions are extracted by `backend/scripts/extract_api_permissions.py` and fed to Kong for RBAC. Internal algorithm callbacks (skill suggestions, memory) use `nil` permissions — protected by internal service token instead.

**evo pipeline**: Runs as either `auto` (fully automated) or `interactive` (pauses for human approve/revise/cancel). The pipeline `dataset_gen → eval → run (analyze) → apply (code fix) → merge → deploy → abtest` can be triggered via CLI (`evo pipeline`) or HTTP API. Uses `opencode` (external binary) for code generation in the apply stage.

### Coding Standards

From `.cursor/rules/coding-standards.mdc`:
- Write all comments in **English**
- Python: prefer **single quotes** (`'`) for strings when both are valid
- Go: run `gofmt` after modifying files
- Python lint: flake8 (config in `.flake8`), excludes `algorithm/lazyllm` (submodule), max line length 120
- Go module: `lazymind/core` in `backend/core/`

### Env Files

- `.env` — runtime configuration (model paths, credentials, ports, feature flags)
- `.env.example` — documented template
- `.env.mirrors.cn` — domestic mirror URLs (default)
- `.env.mirrors.intl` — international mirror URLs
- `Makefile` loads `.env.mirrors.{profile}` first, then `.env` (`.env` overrides profile)

### Frontend Feature Flags

- `VITE_HIDE_EVO` — set to any value to hide the Self-Evolution page from the UI
