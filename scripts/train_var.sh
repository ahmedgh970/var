#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29501}

PYTHONPATH=src torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m var.pipelines.train_var "$@"
