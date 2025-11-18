#!/bin/bash
#SBATCH --job-name=AD_IBM
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=32 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=ibm_multiple.err # standard error file
#SBATCH --output=ibm_multiple.out # standard output file
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source torch210/bin/activate
python src/run_training_multiple_models.py
deactivate



