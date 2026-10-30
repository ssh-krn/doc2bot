#!/usr/bin/env bash
# Builds the machine learning corpus and questions, checks them, runs the evaluation, and writes eval/results/.
#   ./run_all.sh               corpus, checks, BM25 retrieval (no models, no keys)
#   ./run_all.sh --models      also dense, hybrid and reranked retrieval (downloads embedding models once)
#   ./run_all.sh --answers     also answer and compare modes (needs GROQ_API_KEY or MISTRAL_API_KEY)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

$PY -m eval.harness_test
$PY -m eval.build_corpus
$PY -m eval.build_qa
$PY -m eval.validate_qa
$PY -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --lexical-only --chunk-sizes 300 600 900

if [[ "${1:-}" == "--models" || "${2:-}" == "--models" ]]; then
  $PY -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --chunk-sizes 300 600 900
fi

if [[ "${1:-}" == "--answers" || "${2:-}" == "--answers" ]]; then
  $PY -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --chunk-sizes 800 --answer-chunk-size 800 \
    --compare-modes traditional crag self_rag agentic
fi
