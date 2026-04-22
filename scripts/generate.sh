#!/usr/bin/env bash
# CLI: ./scripts/generate.sh
set -euo pipefail

PYTHONPATH=src python -m var.pipelines.generate "$@"
