#!/bin/bash
#SBATCH -J gpu-job            # the name of the job
#SBATCH --cpus-per-task=8     # CPU core in case we OOM
#SBATCH -p gpu                # get the partition that have GPU
#SBATCH --gres=gpu:1          # get GPU for usage
#SBATCH -t 1-00:00:00         # maxinum running time 

module add cuda10.1/toolkit
module load shared
module load cm-ml-python3deps
module load pytorch-py36-cuda10.1-gcc/1.3.0

python gpu_task.py         