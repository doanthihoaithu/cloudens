#!/bin/bash
#SBATCH --job-name=AD_IBM_4xx5xx
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00 # time limit: 48 hours
#SBATCH --mem=64G # the 4xx + 5xx subset has ~6,500 nodes: its windows need ~15-20 GB of RAM
#SBATCH --error=combined_subset_%j.err # standard error file
#SBATCH --output=combined_subset_%j.out # standard output file
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# Train and score one model on the subset combining the 4xx and 5xx series of one aggregation
# (data/massaged/extracted/no_group_5xx_4xx_<aggregation>/, results in .../no_group_5xx_4xx_<aggregation>/).
# Usage (from the project root):
#   sbatch slurm_script_train_combined_subset.sh <model> <window> <fill_nan> <null_padding_feature> <aggregation> [Hydra overrides...]
#   sbatch slurm_script_train_combined_subset.sh GRU 6 zero false count
#   sbatch slurm_script_train_combined_subset.sh BigST 6 zero true count model_configs.BigST.use_spatial=true
#   sbatch slurm_script_train_combined_subset.sh A3TGCN 6 zero true count evaluation.retrain=true
MODEL=${1:-GRU}
WINDOW=${2:-6}
FILL_NAN=${3:-zero}
NULL_PADDING_FEATURE=${4:-false}
AGGREGATION=${5:-count}

module load python
source torch210_new/bin/activate
export PYTHONUNBUFFERED=1
python src/run_training_single_model.py \
    evaluation.use_model="$MODEL" \
    evaluation.slide_win="$WINDOW" \
    evaluation.fill_nan="$FILL_NAN" \
    evaluation.null_padding_feature="$NULL_PADDING_FEATURE" \
    "data_preparation_pipeline.features_prep.filter.http_codes=[5xx,4xx]" \
    "data_preparation_pipeline.features_prep.filter.aggregations=[$AGGREGATION]" \
    "${@:6}"
deactivate
