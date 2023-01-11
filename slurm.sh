#!/bin/bash
#SBATCH -J test_gpu_pytorch
#SBATCH --ntasks=1
#SBATCH -o /home/users/l/lorenz-08-15/activelearning_ic/cluster_outputs/%j.log
#SBATCH -D /home/users/l/lorenz-08-15/activelearning_ic/
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:tesla:1

# set up proxy (no internet on nodes)
if [ ! "$HOSTNAME" == "frontend*" ]; then
 export https_proxy="http://frontend01:3128/"
 export http_proxy="http://frontend01:3128/"
 echo "HTTP proxy set up done"
fi

module load nvidia/cuda/11.2

python main.py --run_name random_test_bleu --bs 32
# python main.py --bs 8 --init_set_size 100 --maxcycles 1 --epochs 3 --run_name full_test_bleu --val_check_interval 0.1