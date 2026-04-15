# VAR — Visual Autoregressive Modeling

A clean, research-oriented reimplementation of **Visual Autoregressive (VAR)** image generation. The model autoregressively predicts discrete image tokens scale-by-scale — from a 1×1 global token up to a 16×16 fine-detail map — rather than pixel-by-pixel or patch-by-patch.

Current target: **unconditional generation on FFHQ-256** as a baseline to validate the architecture and training pipeline before adding conditioning.

---

## How it works

VAR operates in two stages:

```
Stage 1 — Tokenizer (pretrained, frozen)
  Image (256×256) → Multiscale VQ-VAE → 10 token maps (1²+2²+…+16² = 680 tokens)

Stage 2 — VAR Transformer (trained here)
  Token maps → Autoregressive transformer → Next-scale token prediction
  Generation: sample scale-by-scale from coarse (1×1) to fine (16×16)
```

The tokenizer is **not trained from scratch** — pretrained weights from the official VAR release are loaded directly (with automatic key remapping to match this codebase's naming).

---

## Project layout

```
src/var/
├── datasets/
│   ├── image_dataset.py      # image loading and train/val splits
│   ├── token_dataset.py      # token dataset (one .pt per image, list of per-scale tensors)
│   └── transforms.py         # resize/crop/normalize for FFHQ
├── models/
│   ├── tokenizer/
│   │   ├── vqvae.py               # full VQ-VAE (encoder + quantizer + decoder)
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── quantizer.py           # single-scale VQ with STE
│   │   ├── multiscale_quantizer.py # hierarchical residual quantizer with Phi blending
│   │   └── checkpoint.py          # pretrained weight loading with official VAR key remapping
│   ├── var/
│   │   ├── var_model.py           # VAR transformer (scale-causal attention, KV cache)
│   │   └── transformer.py         # TransformerBlock, CausalSelfAttention, FFN, DropPath
│   └── common/
│       ├── mlp.py
│       ├── normalization.py
│       └── utils.py
├── training/
│   ├── var_trainer.py        # training loop with EMA, visual sampling, distributed sync
│   ├── tokenizer_trainer.py  # tokenizer training loop
│   ├── ema.py                # exponential moving average of model parameters
│   ├── losses.py             # reconstruction + VQ losses
│   ├── optim.py              # AdamW with per-parameter weight decay
│   └── schedulers.py         # none / cosine / warmup_cosine / lin0
├── inference/
│   ├── generator.py          # scale-by-scale token generation with KV caching
│   ├── sampler.py            # top-k / top-p / temperature sampling
│   └── decode.py             # token indices → images
├── pipelines/
│   ├── train_var.py          # VAR training entry point
│   ├── train_tokenizer.py    # tokenizer training entry point
│   ├── tokenize_dataset.py   # pre-tokenize images → .pt token files
│   ├── eval_tokenizer.py     # reconstruction quality evaluation
│   ├── eval_var.py           # VAR evaluation
│   └── generate.py           # image generation entry point
├── metrics/
│   ├── fid.py
│   ├── inception.py
│   └── reconstruction_metrics.py
└── utils/
    ├── checkpoint.py
    ├── logger.py
    ├── distributed.py
    ├── seed.py
    └── config.py

configs/
├── train_var.yaml            # main VAR training config
├── train_tokenizer.yaml      # tokenizer training config
├── tokenize_dataset.yaml     # dataset tokenization config
├── eval_tokenizer.yaml       # tokenizer evaluation config
├── generate.yaml             # generation config
├── var/
│   └── var_base.yaml         # VAR model architecture
├── tokenizer/
│   └── vqvae_base.yaml       # VQ-VAE architecture
└── datasets/
    └── ffhq.yaml             # FFHQ-256 dataset paths

scripts/
├── train_var.sh              # torchrun wrapper for VAR training
├── train_tokenizer.sh        # torchrun wrapper for tokenizer training
├── tokenize_dataset.sh       # dataset pre-tokenization
├── eval_tokenizer.sh         # tokenizer evaluation
└── generate.sh               # image generation
```

---

## Dataset

**FFHQ-256** with the following directory structure:

```
/data/tii/data/ffhq256_train_val/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── tokens/
    └── pre_vqvae/
        ├── train/   # one .pt per image, each containing a list of 10 scale tensors
        ├── val/
        └── test/
```

Each `.pt` token file stores `[t_1, t_2, ..., t_10]` where `t_i` has shape `(pn_i, pn_i)` for `pn = (1,2,3,4,5,6,8,10,13,16)`.

---

## Pipeline

### Step 1 — Pre-tokenize the dataset

Encodes all images using the pretrained tokenizer and saves one `.pt` token file per image.

```bash
./scripts/tokenize_dataset.sh \
  checkpoint_path=/path/to/vae_ch160v4096z32.pth \
  tokenizer.quantizer_type=multi \
  device=cuda:0
```

Config: [`configs/tokenize_dataset.yaml`](configs/tokenize_dataset.yaml)

---

### Step 2 — Train the VAR transformer

```bash
# Single GPU
./scripts/train_var.sh \
  tokens_root=/data/tii/data/ffhq256_train_val/tokens/pre_vqvae \
  tokenizer_checkpoint_path=/path/to/vae_ch160v4096z32.pth

# Multi-GPU (e.g. 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ./scripts/train_var.sh \
  tokens_root=/data/tii/data/ffhq256_train_val/tokens/pre_vqvae \
  tokenizer_checkpoint_path=/path/to/vae_ch160v4096z32.pth
```

Config: [`configs/train_var.yaml`](configs/train_var.yaml)

Checkpoints and logs are written to `checkpoints/var/{timestamp}/`. The run directory also contains:
```
{timestamp}/
├── last.pt          # latest checkpoint (model + optimizer + EMA)
├── best.pt          # checkpoint with lowest val loss
├── train.log
└── samples/
    ├── epoch_0050/  # generated images for visual inspection
    ├── epoch_0100/
    └── ...
```

---

### Step 3 — Generate images

```bash
./scripts/generate.sh \
  var_checkpoint_path=/path/to/last.pt \
  tokenizer_checkpoint_path=/path/to/vae_ch160v4096z32.pth
```

Config: [`configs/generate.yaml`](configs/generate.yaml)  
Outputs PNGs to `experiments/var/{timestamp}/samples/`. Set `use_ema: true` (default) to generate with EMA weights.

---

### Evaluate the tokenizer

```bash
./scripts/eval_tokenizer.sh \
  checkpoint_path=/path/to/tokenizer.pth \
  tokenizer.quantizer_type=multi \
  device=cuda:0
```

---

## Key model parameters

**Tokenizer** (`configs/tokenizer/vqvae_base.yaml`):

| Parameter | Value |
|---|---|
| `vocab_size` | 4096 |
| `z_channels` | 32 |
| `ch` | 160 |
| `patch_nums` | (1,2,3,4,5,6,8,10,13,16) |
| `quantizer_type` | `multi` (multiscale residual) |

**VAR Transformer** (`configs/var/var_base.yaml`):

| Parameter | Value |
|---|---|
| `dim` | 384 |
| `depth` | 8 |
| `num_heads` | 6 |
| `mlp_ratio` | 4.0 |
| `dropout` | 0.0 |
| `drop_path_rate` | 0.05 |
| `attn_l2_norm` | true |

---

## Notable implementation details

**Pretrained tokenizer loading** — `checkpoint.py` handles key remapping from the official VAR checkpoint format to this codebase's naming automatically. A pre-remap validation step reports exactly which keys are missing or unexpected if the remap produces a mismatch.

**EMA** — `ModelEMA` tracks an exponential moving average of VAR parameters (decay=0.999 by default). Both validation loss evaluation and periodic sample generation use EMA weights. EMA state is saved in every checkpoint under the `"ema"` key and used by default at generation time.

**Periodic visual samples** — The trainer generates a small batch of images every `sample_every` epochs (default: 50) and saves them to `samples/epoch_{N}/`. This gives a visual timeline of training progress independent of the loss curve.

**KV caching** — Enabled during inference. At each scale step only the new scale's keys and values are computed; previous scales are read from cache. This makes generation time proportional to the number of scales rather than sequence length squared.

**Scale-causal attention mask** — Tokens can attend to all previous scales and the current scale, but not future scales. This is implemented as a static buffer computed once at model init.

---

## Status

- [x] Image dataset pipeline (FFHQ-256)
- [x] Multiscale VQ-VAE tokenizer (pretrained weights, key remapping)
- [x] Dataset pre-tokenization pipeline
- [x] VAR transformer training (single and multi-GPU)
- [x] EMA, visual validation samples, checkpoint hardening
- [x] Autoregressive generation with KV caching and top-k/top-p sampling
- [x] Tokenizer evaluation
- [ ] Conditioned generation (class / text)
