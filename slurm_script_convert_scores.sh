#!/bin/bash
#SBATCH --job-name=AD_convert
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=4 # reading / writing files only, no GPU needed
#SBATCH --time=04:00:00 # time limits: 4 hours
#SBATCH --error=convert.err # standard error file
#SBATCH --output=convert.out # standard output file
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

# Convert the anomaly score CSVs of a results folder to parquet (scores rounded to 5 decimals), no retraining
# Usage (from the project root): sbatch slurm_script_convert_scores.sh [results folder]
ROOT=$(realpath "${1:-trained_models_log1p}")

module load python
source torch210/bin/activate
export PYTHONUNBUFFERED=1
cd src
python convert_anomaly_scores.py "$ROOT"
deactivate
