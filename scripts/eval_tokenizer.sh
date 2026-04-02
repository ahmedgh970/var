#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m var.pipelines.eval_tokenizer "$@"
