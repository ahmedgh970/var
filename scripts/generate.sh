#!/usr/bin/env bash
# CLI: ./scripts/generate.sh var_checkpoint_path=/path/to/var/best.pt tokenizer_checkpoint_path=/path/to/tokenizer/best.pt num_samples=16 batch_size=8 device=cuda:0
set -euo pipefail

PYTHONPATH=src python -m var.pipelines.generate "$@"
