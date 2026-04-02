#!/usr/bin/env bash
# CLI: ./scripts/tokenize_dataset.sh checkpoint_path=/data/ahmed.ghorbel/projects/10-var/checkpoints/tokenizer/2026-04-02_17-23-18/best.pt tokenizer.quantizer_type=single datasets.test_batch_size=4 device=cuda:0
set -euo pipefail

PYTHONPATH=src python -m var.pipelines.tokenize_dataset "$@"
