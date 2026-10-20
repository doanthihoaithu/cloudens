#!/bin/bash
#SBATCH --job-name=AD_merge
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=4 # reading CSVs only, no GPU needed
#SBATCH --time=01:00:00 # time limits: 1 hour
#SBATCH --mem=8G # the merge is streamed: one grid-search file in memory at a time
#SBATCH --error=merge.err # standard error file
#SBATCH --output=merge.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# Merge the grid-search CSVs of a results folder into merged_results/<folder>_<timestamp>.csv.gz
# Usage (from the project root):
#   sbatch slurm_script_merge_results.sh [results folder] [output file] [--exclude-combined-subsets]
#   sbatch slurm_script_merge_results.sh
#   sbatch slurm_script_merge_results.sh trained_models
#   sbatch slurm_script_merge_results.sh trained_models_log1p merged_results/hpc.csv.gz
#   sbatch slurm_script_merge_results.sh trained_models_log1p merged_results/hpc.csv.gz --exclude-combined-subsets
# --exclude-combined-subsets leaves out the subsets combining two http codes (no_group_5xx_4xx_<aggregation>),
# merged apart by slurm_script_merge_combined_subsets.sh. Other options of grid_results.py are passed on too.
POSITIONAL=()
OPTIONS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then OPTIONS+=("$arg"); else POSITIONAL+=("$arg"); fi
done
ROOT=${POSITIONAL[0]:-trained_models_log1p}

module load python
source torch210_new/bin/activate
if [ -n "${POSITIONAL[1]}" ]; then
    python streamlit_result_visualization/grid_results.py "$ROOT" -o "${POSITIONAL[1]}" "${OPTIONS[@]}"
else
    python streamlit_result_visualization/grid_results.py "$ROOT" "${OPTIONS[@]}"
fi
deactivate
