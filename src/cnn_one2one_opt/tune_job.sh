#!/bin/bash

#SBATCH -p speech-gpu
#SBATCH --constraint=highmem
#SBATCH --array=1-9

bash -c "`sed "${SLURM_ARRAY_TASK_ID}q;d" tune_commands`"

