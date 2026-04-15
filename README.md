# VAR (Step-by-step Reimplementation)

This repository is a clean reimplementation of **Visual Autoregressive (VAR)** for learning autoregressive image generation.

Current objective:
1. Train a **multiscale VQ-VAE tokenizer**
2. Tokenize images
3. Train a **VAR transformer**
4. Sample and edit images

The code is intentionally minimal and research-oriented.

## Current Dataset Setup

We currently use **FFHQ 256**.

Configured dataset path:
- `/data/tii/data/ffhq256_train_val/images`

Expected structure:
```text
/data/tii/data/ffhq256_train_val/images
├── train/   # ~4000 images
├── val/     # ~500 images
└── test/     # ~500 images
```

## Configs

Main config files currently prepared:
- [`configs/datasets/ffhq.yaml`](./configs/datasets/ffhq.yaml)
- [`configs/tokenizer/vqvae_ffhq.yaml`](./configs/tokenizer/vqvae_ffhq.yaml)
- 

## Project Layout

```text
src/var/
├── datasets/      # image and token datasets + transforms
├── models/        # tokenizer and VAR models
├── training/      # training loops, losses, optim, schedulers
├── pipelines/     # train/tokenize/eval entry points
├── inference/     # generation and decoding
└── editing/       # token/image editing
```

## Status

- Dataset pipeline: Done !
- Tokenizer (VQ-VAE): Done ! 
- VAR training pipeline: Done ! 
