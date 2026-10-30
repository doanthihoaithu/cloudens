#!/bin/bash
# Gather the likelihood score files of GRU, ClouDens-A3T* and ClouDens-BigST-ctx* into one folder (same
# window_*/no_group_5xx_count/fill_nan_with_zero/<model>/ layout as the results root) and pack it as a .tar.gz.
# Run from the cloudens root on the HPC (login node is fine, it only copies files):
#   bash collect_likelihood_scores.sh [results_root] [out_name]
# then on the local machine:
#   scp <hpc>:<cloudens>/<out_name>.tar.gz . && tar -xzf <out_name>.tar.gz
set -euo pipefail

RESULTS_ROOT=${1:-trained_models_log1p}
OUT_NAME=${2:-likelihood_scores_$(date +%Y%m%d)}
SUBSET=no_group_5xx_count   # only the 5xx count subset
FILL_NAN=fill_nan_with_zero # ... with NaNs filled with zero

# result folder : file prefix
MODELS=(
  "GRU:GRU"                                   # GRU
  "A3TGCN_null_padding_feature:A3TGCN"        # ClouDens-A3T*
  "BigST_null_padding_feature_spatial:BigST"  # ClouDens-BigST-ctx*
)
NORMALIZATIONS=(per_node global)

[ -d "$RESULTS_ROOT" ] || { echo "No results root: $RESULTS_ROOT" >&2; exit 1; }
rm -rf "$OUT_NAME"
mkdir -p "$OUT_NAME"

copied=0
for entry in "${MODELS[@]}"; do
  folder=${entry%%:*}
  prefix=${entry##*:}
  while IFS= read -r model_dir; do
    rel=${model_dir#"$RESULTS_ROOT"/}
    files=()
    for norm in "${NORMALIZATIONS[@]}"; do
      # parquet score file, else the earlier CSV one
      if [ -f "$model_dir/${prefix}_anomaly_scores_${norm}.parquet" ]; then
        files+=("${prefix}_anomaly_scores_${norm}.parquet")
      elif [ -f "$model_dir/${prefix}_anomaly_scores_${norm}.csv" ]; then
        files+=("${prefix}_anomaly_scores_${norm}.csv")
      fi
      for f in "${prefix}_anomaly_scores_${norm}_settings.json" \
               "${prefix}_grid_search_${norm}.csv" \
               "likelihood_top_k_contribution_${norm}.csv"; do
        [ -f "$model_dir/$f" ] && files+=("$f")
      done
    done
    if [ ${#files[@]} -eq 0 ]; then
      echo "skip (no score files): $rel"
      continue
    fi
    mkdir -p "$OUT_NAME/$rel"
    for f in "${files[@]}"; do
      cp -p "$model_dir/$f" "$OUT_NAME/$rel/"
      copied=$((copied + 1))
    done
    echo "ok: $rel (${#files[@]} files)"
  done < <(find "$RESULTS_ROOT" -mindepth 4 -maxdepth 4 -type d -path "*/$SUBSET/$FILL_NAN/*" -name "$folder" | sort)
done

echo "$copied files copied to $OUT_NAME/ ($(du -sh "$OUT_NAME" | cut -f1))"
tar -czf "$OUT_NAME.tar.gz" "$OUT_NAME"
echo "Archive: $(pwd)/$OUT_NAME.tar.gz ($(du -sh "$OUT_NAME.tar.gz" | cut -f1))"
