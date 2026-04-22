#!/usr/bin/env bash
# CLI: ./scripts/eval_tokenizer.sh checkpoint_path=/data/ahmed.ghorbel/projects/10-var/checkpoints/tokenizer/2026-04-08_14-54-55/best.pt tokenizer.quantizer_type=multi datasets.test_batch_size=4 device=cuda:0
set -euo pipefail

PYTHONPATH=src python -m var.pipelines.eval_tokenizer "$@"
