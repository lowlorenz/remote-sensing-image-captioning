#!/bin/bash
#SBATCH -J w_c_0
#SBATCH -o /home/users/w/wallburg/activelearning_ic/logs/word_conf_0.log
#SBATCH -D /home/users/w/wallburg/activelearning_ic/
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
srun python main.py --batch_size 12 --max_cycles 9 --epochs 10 --run_name word_conf_0 --sample_method confidence --conf_mode least --conf_average word --seed 0