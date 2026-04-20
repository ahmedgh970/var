#!/bin/bash
#SBATCH --job-name=var_train
#SBATCH --output=/gpfswork/rech/vcv/uyy89lr/logs/var_%j.out
#SBATCH --error=/gpfswork/rech/vcv/uyy89lr/logs/var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_p5
#SBATCH --account=vcv@a100
#SBATCH --hint=nomultithread

# --- Environnement ---
module purge
module load pytorch-gpu/py3/2.2.0

export PYTHONPATH=$HOME/var/src:$PYTHONPATH
export TMPDIR=$SCRATCH/tmp
mkdir -p $SCRATCH/tmp

# --- Chemins ---
TOKENS=$SCRATCH/dataset/imagenet1k_256px/tokens
TOKENIZER_CKPT=$WORK/checkpoints/tokenizer/vae_ch160v4096z32.pth
CKPT_DIR=$WORK/checkpoints/var

mkdir -p $WORK/logs
mkdir -p $CKPT_DIR

# --- Lancement ---
cd $HOME/var

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29501 \
    -m var.pipelines.train_var \
    datasets.token_root=$TOKENS \
    tokenizer.checkpoint_path=$TOKENIZER_CKPT \
    checkpoint_dir=$CKPT_DIR \
    train.batch_size=64 \
    train.num_workers=8