#!/bin/bash
#SBATCH -J cl_co_0
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/cluster+confidence_0.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_short
#SBATCH --time=16:00:00


module load nvidia/cuda/11.2

srun python main.py --debug --batch_size 12 --max_cycles 9 --epochs 10 --run_name cluister+confidence_0 --sample_method cluster+confidence --cluster_mode image --seed 0