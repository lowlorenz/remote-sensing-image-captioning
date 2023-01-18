JOBID=$(sbatch slurm.sh | sed --expression='s/[^0-9]*//g')
./watch.sh $JOBID