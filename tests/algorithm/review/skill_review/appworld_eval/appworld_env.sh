#!/usr/bin/env bash

# Source this file before running appworld_eval scripts, or pass the same
# values via run_eval.py command-line flags.
export LAZYMIND_APPWORLD_DATA_ROOT="${LAZYMIND_APPWORLD_DATA_ROOT:-/Users/liyang16/workspace/datasets/appworld}"
export LAZYMIND_APPWORLD_REPO_ROOT="${LAZYMIND_APPWORLD_REPO_ROOT:-/Users/liyang16/workspace/evaluate/appworld}"
export LAZYMIND_APPWORLD_ENVIRONMENT_URL="${LAZYMIND_APPWORLD_ENVIRONMENT_URL:-http://127.0.0.1:18000}"
export LAZYMIND_APPWORLD_APIS_URL="${LAZYMIND_APPWORLD_APIS_URL:-http://127.0.0.1:19000}"
export LAZYMIND_CORE_DATABASE_URL="${LAZYMIND_CORE_DATABASE_URL:-postgresql+psycopg://root:123456@localhost:5432/core}"
export LAZYMIND_MODEL_CONFIG_PATH="${LAZYMIND_MODEL_CONFIG_PATH:-inner}"
export LAZYLLM_MINIMAX_API_KEY="${LAZYLLM_MINIMAX_API_KEY:-sk-maas-GDZmEQsilc4uGXXTaWnIHmET9V0eenZ8F6eWk3LaPzE}"

# Optional extra import path for local LazyMind/LazyRAG checkouts outside this workspace.
# The scripts already add ../LazyRAG/algorithm and ../LazyRAG/algorithm/lazyllm
# automatically when that tree exists next to appworld_eval.
export APPWORLD_EVAL_EXTRA_PYTHONPATH="${APPWORLD_EVAL_EXTRA_PYTHONPATH:-}"
