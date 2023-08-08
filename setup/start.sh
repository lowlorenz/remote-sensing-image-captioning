#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenz.hufe@hhi.fraunhofer.de

#         Output (stdout and stderr) of this job will go into a file named with SLURM_JOB_ID (%j) and the job_name (%x)
#SBATCH --output=%j_%x.out

#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4 # num GPUs per node - see https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html#troubleshooting

#SBATCH --cpus-per-task=16

#SBATCH --mem=256G

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"

apptainer run --nv --bind $DATAPOOL3/datasets/NWPU-Captions:/data --bind /data/cluster/users/hufe/llama2/llama-huggingface/:/llama2 app_container.sif 
