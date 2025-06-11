#!/bin/bash
#SBATCH --job-name=typer-1.5B-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --account=$SLURM_ACCOUNT

if [[ -z "$PROJECT" || -z "$SCRATCH" ]]; then
    echo "PROJECT and SCRATCH must be set."
    exit 1
fi

echo "Loading modules..."
module load Python
module load CUDA/12
module load PyTorch/2.5.1

echo "Activating virtual environment..."
source $PROJECT/TypeShift/typeshift_venv/bin/activate

echo "Setting Hugging Face cache directories..."
export HF_HOME="$SCRATCH/huggingface"
export TRITON_CACHE_DIR="$SCRATCH/triton"

echo "Setting offline mode..."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Navigating to working directory..."
cd $PROJECT/TypeShift

export UNSLOTH_RETURN_LOGITS=1
python train-cpt.py
