#!/bin/bash
#SBATCH -J test_gpu_pytorch
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/%j.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=2:00:00

# set up proxy (no internet on nodes)
if [ ! "$HOSTNAME" == "frontend*" ]; then
 export https_proxy="http://frontend01:3128/"
 export http_proxy="http://frontend01:3128/"
 echo "HTTP proxy set up done"
fi

# debugging flags (optional)
export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

module load nvidia/cuda/11.2

srun python main.py --bs 12 --init_set_size 100 --maxcycles 1 --epochs 50 --run_name multi-gpu-full-dataset --num_devices 2 --num_nodes 4 --ckpt_path active_learning/36xxdvbv/checkpoints/epoch=9-step=2630.ckpt