#!/bin/bash
#SBATCH -J test_gpu_pytorch
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/random_2.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_short
#SBATCH --time=8:00:00

# set up proxy (no internet on nodes)
if [ ! "$HOSTNAME" == "frontend*" ]; then
 export https_proxy="http://frontend01:3128/"
 export http_proxy="http://frontend01:3128/"
 echo "HTTP proxy set up done"
fi

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

module load nvidia/cuda/11.2

# srun python mnist.py
srun python main.py --batch_size 12 --max_cycles 9 --epochs 10 --run_name random_2 --num_devices 2 --num_nodes 2 --sample_method random --seed 2