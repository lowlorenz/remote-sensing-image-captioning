#!/bin/bash
#SBATCH -J cluster_i_2
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/cluster_image_2.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_short
#SBATCH --time=16:00:00

# set up proxy (no internet on nodes)
if [ ! "$HOSTNAME" == "frontend*" ]; then
 export https_proxy="http://frontend01:3128/"
 export http_proxy="http://frontend01:3128/"
 echo "HTTP proxy set up done"
fi

# debugging flags (optional
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

module load nvidia/cuda/11.2

# srun python mnist.py
srun python main.py --batch_size 12 --max_cycles 9 --epochs 10 --run_name cluster_image_2 --sample_method cluster --cluster_mode image --seed 2