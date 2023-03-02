#!/bin/bash
#SBATCH -J co+cl_2
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/confidence+cluster_2.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_short
#SBATCH --time=16:00:00


module load nvidia/cuda/11.2

srun python main.py --batch_size 12 --max_cycles 9 --epochs 10 --run_name confidence+cluster_2 --sample_method cluster+confidence --cluster_mode image --seed 2 --conf_average word --conf_mode margin