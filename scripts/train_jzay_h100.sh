#!/bin/bash
#SBATCH --job-name=var_train_h100
#SBATCH --output=/gpfswork/rech/vcv/uyy89lr/logs/var_%j.out
#SBATCH --error=/gpfswork/rech/vcv/uyy89lr/logs/var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_p6
#SBATCH --account=vcv@h100
#SBATCH --hint=nomultithread
#SBATCH -C h100

module purge
module load arch/h100
module load pytorch-gpu/py3/2.2.0

export PYTHONPATH=$HOME/var/src:$PYTHONPATH
export TMPDIR=$SCRATCH/tmp/$SLURM_JOB_ID
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

TOKENS=$SCRATCH/dataset/imagenet1k_256px/tokens
TOKENIZER_CKPT=$WORK/checkpoints/tokenizer/vae_ch160v4096z32.pth
CKPT_DIR=$WORK/checkpoints/var_train
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

mkdir -p "$CKPT_DIR"

cd $HOME/var

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:$MASTER_PORT \
    -m var.pipelines.train_var \
    datasets.token_root=$TOKENS \
    tokenizer.checkpoint_path=$TOKENIZER_CKPT \
    checkpoint_dir=$CKPT_DIR \
    var.depth=8 \
    var.dim=512 \
    var.num_heads=8 \
    var.drop_path_rate=0.0333 \
    train.epochs=1000 \
    train.batch_size=256 \
    train.num_workers=4 \
    train.eval_batch_size=256 \
    train.grad_accum_steps=2 \
    var.torch_compile=false \
    ema.enabled=true \
    ema.decay=0.9999 \
    logging.eval_every=1 \
    logging.save_every=10 \
    logging.sample_every=50 \
    logging.num_val_samples=3
