#!/bin/bash
#SBATCH --job-name=var_test
#SBATCH --output=/gpfswork/rech/vcv/uyy89lr/logs/var_test_%j.out
#SBATCH --error=/gpfswork/rech/vcv/uyy89lr/logs/var_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_p13
#SBATCH --account=vcv@v100
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g        # ← cibler V100 32Go explicitement

# --- Environnement ---
module purge
module load pytorch-gpu/py3/2.2.0

export PYTHONPATH=$HOME/var/src:$PYTHONPATH
export TMPDIR=$SCRATCH/tmp
export PYTHONUNBUFFERED=1
mkdir -p $SCRATCH/tmp

# --- Chemins ---
TOKENS=$SCRATCH/dataset/imagenet1k_256px/tokens
TOKENIZER_CKPT=$WORK/checkpoints/tokenizer/vae_ch160v4096z32.pth
CKPT_DIR=$WORK/checkpoints/var_test

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
    var.depth=4 \
    var.dim=256 \
    var.num_heads=4 \
    var.drop_path_rate=0.0333 \
    train.epochs=50 \
    train.batch_size=8 \
    train.num_workers=4 \
    train.eval_batch_size=8 \
    logging.eval_every=1 \
    logging.save_every=1 \
    logging.sample_every=5 \
    logging.num_val_samples=2