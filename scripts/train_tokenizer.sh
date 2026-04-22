#!/usr/bin/env bash
# CLI: CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 ./scripts/train_tokenizer.sh

set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29500}

PYTHONPATH=src torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m var.pipelines.train_tokenizer "$@"
