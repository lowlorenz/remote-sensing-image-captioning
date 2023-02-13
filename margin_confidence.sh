#!/bin/bash
#SBATCH -J conf_m_0
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/conf_margin_0.log
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
srun python main.py --batch_size 12 --max_cycles 9 --epochs 10 --run_name conf_margin_0 --sample_method confidence --conf_mode margin --seed 0