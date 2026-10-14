#!/bin/bash
#SBATCH --job-name=AD_IBM
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=8 # 32 tasks per node
#SBATCH --time=48:00:00 # time limits: 1 hour
#SBATCH --error=ibm_multiple.err # standard error file
#SBATCH --output=ibm_multiple.out # standard output file
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# Extra arguments are Hydra overrides of conf/config.yaml, e.g.
#   sbatch slurm_script_multiple_training.sh 'multiple_evaluation.models=[BigST]' model_configs.BigST.use_spatial=true
module load python
source torch210_new/bin/activate
export PYTHONUNBUFFERED=1
python src/run_training_multiple_models.py "$@"
deactivate



