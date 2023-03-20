#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 4:00:00
#SBATCH -A m3329_g
#SBATCH -q regular
#SBATCH -n 4     # number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1     # reserve 1 (of four) GPUs per task

module load python pytorch/1.9.0

export SLURM_CPU_BIND="cores"

pip install ../

python test_igsimple.py
