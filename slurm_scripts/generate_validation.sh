#!/bin/bash
#SBATCH -J generate_val
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/generate_valdation_full.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_short
#SBATCH --time=01:00:00


module load nvidia/cuda/11.2

srun python validation_generator.py /scratch/activelearning-ic/full_0-08-02-2023-12-50-0.ckpt
#srun python validation_generator.py --num_cycles 9 /scratch/activelearning-ic/conf_margin_0-08-02-2023-12-36-8.ckpt