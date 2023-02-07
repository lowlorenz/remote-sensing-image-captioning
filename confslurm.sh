#!/bin/bash
#SBATCH -J random 
#SBATCH -o /home/users/w/wallburg/dif_log/logs/random.log
#SBATCH -D /home/users/w/wallburg/dif_log/
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=08:00:00

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

srun python main.py --max_cycles 9 --epochs 10 --batch_size 12 --sample_method random --data_path ../../../../../../scratch/lhlw --device_type cuda --num_nodes 1 --num_devices 1 --run_name random
