#!/bin/bash
#SBATCH --job-name=AD_ens_sub
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=8 # OR ensembles of the saved test predictions only, no GPU needed
#SBATCH --time=12:00:00 # time limits
#SBATCH --mem=32G
#SBATCH --error=ensemble_subsets.err # standard error file
#SBATCH --output=ensemble_subsets.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# OR ensembles of 1, 2, 3 and 8 subsets of ClouDens-A3T* and ClouDens-BigST-ctx* (src/ensemble_subsets.py),
# outputs in merged_results/ensemble_subsets. Usage (from the project root, merged_results/hpc.csv.gz present):
#   sbatch slurm_script_ensemble_subsets.sh
#   sbatch slurm_script_ensemble_subsets.sh --windows 12 --normalizations global
module load python
source torch210_new/bin/activate
export PYTHONUNBUFFERED=1
python src/ensemble_subsets.py --workers 8 "$@"
deactivate
