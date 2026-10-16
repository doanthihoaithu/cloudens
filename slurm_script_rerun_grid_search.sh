#!/bin/bash
#SBATCH --job-name=AD_regrid
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=4 # grid search on the saved anomaly scores only, no GPU needed
#SBATCH --time=24:00:00 # time limits
#SBATCH --mem=16G
#SBATCH --error=regrid.err # standard error file
#SBATCH --output=regrid.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# Grid search of post-processing strategies (default: mahalanobis) from the saved anomaly scores, without
# retraining; replaces their rows in the grid-search CSVs. Usage (from the project root):
#   sbatch slurm_script_rerun_grid_search.sh                                   # mahalanobis, trained_models_log1p
#   sbatch slurm_script_rerun_grid_search.sh trained_models_log1p --models BigST A3TGCN
#   sbatch slurm_script_rerun_grid_search.sh trained_models_log1p --recompute-mahalanobis   # new Mahalanobis settings
ROOT=${1:-trained_models_log1p}
shift

module load python
source torch210_new/bin/activate
export PYTHONUNBUFFERED=1
python src/rerun_grid_search.py "$ROOT" "$@" | grep --line-buffered -E '^\[[0-9]+/[0-9]+]|^Updated|^Not updated|^[0-9]+ anomaly score files|Error|Traceback'
deactivate
