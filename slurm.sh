#!/bin/bash
#SBATCH -J test_gpu_pytorch
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/%j.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

# set up proxy (no internet on nodes)
if [ ! "$HOSTNAME" == "frontend*" ]; then
 export https_proxy="http://frontend01:3128/"
 export http_proxy="http://frontend01:3128/"
 echo "HTTP proxy set up done"
fi

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load nvidia/cuda/11.2

srun python mnist.py
# python main.py --run_name random_test_bleu --bs 8
#srun python main.py --bs 64 --init_set_size 100 --maxcycles 1 --epochs 10 --run_name overnight_8_gpu_test --val_check_interval 0.1 --num_devices 1 --num_nodes 2 --debug