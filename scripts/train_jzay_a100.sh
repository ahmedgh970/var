#!/bin/bash
#SBATCH --job-name=var_train_a100
#SBATCH --output=/gpfswork/rech/vcv/uyy89lr/logs/var_%j.out
#SBATCH --error=/gpfswork/rech/vcv/uyy89lr/logs/var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_p5
#SBATCH --account=vcv@a100
#SBATCH --hint=nomultithread
#SBATCH -C a100

module purge
module load arch/a100
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
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:$MASTER_PORT \
    -m var.pipelines.train_var \
    datasets.token_root=$TOKENS \
    tokenizer.checkpoint_path=$TOKENIZER_CKPT \
    checkpoint_dir=$CKPT_DIR \
    var.depth=16 \
    var.dim=1024 \
    var.num_heads=16 \
    var.drop_path_rate=0.0666667 \
    var.init_adaln_gamma=1.0e-3 \
    train.epochs=200 \
    train.batch_size=96 \
    train.num_workers=7 \
    train.eval_batch_size=144 \
    train.grad_accum_steps=1 \
    scheduler.final_lr_ratio=0.1 \
    logging.eval_every=10 \
    logging.save_every=10 \
    logging.sample_every=50 \
    logging.num_val_samples=3
