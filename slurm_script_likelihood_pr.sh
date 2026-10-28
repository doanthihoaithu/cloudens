#!/bin/bash
#SBATCH --job-name=AD_lik_pr
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=4 # from the saved forecasting errors only, no GPU needed
#SBATCH --time=24:00:00 # time limits
#SBATCH --mem=32G # one model folder in memory at a time (~5 GB for 5xx count, more for the combined subsets)
#SBATCH --error=likelihood_pr.err # standard error file
#SBATCH --output=likelihood_pr.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# AUC-PR / VUS-PR of the likelihood strategies, without retraining nor grid search: the inputs of the likelihood
# (normalised errors aggregated over the top-k sensors) are added to the score files and the auc_pr / vus_pr
# columns of the likelihood rows of the grid-search CSVs filled (rerun_grid_search.py --likelihood-pr).
# Merge the results again afterwards (slurm_script_merge_results.sh). Usage (from the project root):
#   sbatch slurm_script_likelihood_pr.sh                                   # trained_models_log1p
#   sbatch slurm_script_likelihood_pr.sh trained_models_log1p --models BigST A3TGCN
#   sbatch slurm_script_likelihood_pr.sh trained_models_log1p --subsets 'no_group_*xx_*xx_*'   # combined subsets only
ROOT=${1:-trained_models_log1p}
shift

module load python
source torch210_new/bin/activate
export PYTHONUNBUFFERED=1
python src/rerun_grid_search.py "$ROOT" --likelihood-pr "$@" | grep --line-buffered -E '^\[[0-9]+/[0-9]+]|^Updated|^Not updated|^[0-9]+ anomaly score files|Error|Traceback'
deactivate
