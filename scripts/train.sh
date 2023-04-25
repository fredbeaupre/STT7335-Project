#!/bin/bash
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=64Gb
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/ad/bin/activate

python train.py