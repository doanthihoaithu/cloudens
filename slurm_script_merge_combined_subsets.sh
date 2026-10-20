#!/bin/bash
#SBATCH --job-name=AD_merge_4xx5xx
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=4 # reading CSVs only, no GPU needed
#SBATCH --time=01:00:00 # time limits: 1 hour
#SBATCH --mem=8G # the merge is streamed: one grid-search file in memory at a time
#SBATCH --error=merge_combined_%j.err # standard error file
#SBATCH --output=merge_combined_%j.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# Merge the grid-search CSVs of the subsets combining two http codes (no_group_5xx_4xx_<aggregation>, run by
# slurm_script_train_combined_subset.sh) into one file, apart from the other subsets, together with those of their
# single subsets (no_group_5xx_<aggregation>, no_group_4xx_<aggregation>) of the same configurations (window,
# fill NaN, model folder, score normalization) only, to compare the combined subsets with them.
# Usage (from the project root): sbatch slurm_script_merge_combined_subsets.sh [results folder] [output file]
#   sbatch slurm_script_merge_combined_subsets.sh
#   sbatch slurm_script_merge_combined_subsets.sh trained_models_log1p merged_results/hpc_combined.csv.gz
ROOT=${1:-trained_models_log1p}
OUTPUT=${2:-merged_results/hpc_combined.csv.gz}

module load python
source torch210_new/bin/activate
python streamlit_result_visualization/grid_results.py "$ROOT" -o "$OUTPUT" --subsets 'no_group_*xx_*xx_*' \
    --with-matching-single-subsets   # 'no_group_*xx_*xx_*': grid_results.COMBINED_SUBSETS
deactivate
