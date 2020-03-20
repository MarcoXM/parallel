#!/bin/bash
#SBATCH -J bert-job           
#SBATCH --cpus-per-task=4          
#SBATCH --gres=gpu:1          
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=xwang423@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=stdout
#SBATCH --error=stderr
#SBATCH --exclude=node[001,002]
#SBATCH --nodes=1


module add cuda10.1/toolkit
module load shared
module load cm-ml-python3deps
module load pytorch-py36-cuda10.1-gcc/1.3.0
pip install --upgrade pip
pip install transformers --user
#module load openmpi/cuda/64/3.1.4


python3.6 singleGPUtrain.py