#!/bin/bash
#SBATCH --job-name=bup20_maskrcnn
#SBATCH --partition=A100short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

module load Python/3.10.8-GCCcore-12.2.0-bare

cd /home/s29pluit/projects/bup20_uq
mkdir -p logs output
source venv/bin/activate

python train.py
