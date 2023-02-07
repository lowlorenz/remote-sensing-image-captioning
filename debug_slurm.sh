#!/bin/bash
#SBATCH -J debug
#SBATCH -o /home/users/w/wallburg/dif_log/test.log
#SBATCH -D /home/users/w/wallburg/dif_log/
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

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
# module load nvidia/cuda/1.6

# srun python mnist.py
srun python main.py --conf_mode "confidence" --debug --max_cycles 3 --epochs 1 --batch_size 12 --sample_method confidence --data_path ../../../../../../scratch/lhlw --device_type cuda --num_nodes 1 --num_devices 1 --run_name debug
#  srun python main.py --batch_size 12 --max_cycles 10 --epochs 10 --run_name dev --num_devices 2 --num_nodes 4 --sample_method cluster
