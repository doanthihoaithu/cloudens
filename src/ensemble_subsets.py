"""
Ensembles of the subsets (<http_code>_<aggregation>) of the ClouDens models with null padding, from the merged
grid-search table (merged_results/hpc.csv.gz): for every model folder, sliding window and score normalisation,
each subset contributes the test predictions of one grid-search setting, and a combination of subsets flags a
timestamp when any of them does (OR vote), evaluated as the single models (NAB standard / reward_fn, point metrics).

Two steps:
  select    (hpc.csv.gz only, no other file needed): the setting of every subset, chosen
            - 'train_valid': on the NAB reward_fn score of the train + valid splits (no test label), or
            - 'test': on the test NAB reward_fn score (oracle, as run_ensemble_approach_new.py does),
            over the imputations (zero only for the count subsets, FIXED_IMPUTATION) and the settings of a
            scoring strategy and threshold type (absolute / percentile), shared by all the members of an ensemble;
            written to selected_configs.csv, with window_union_preview.csv: per combination, the ground-truth
            windows detected by the OR ensemble (exact: union of the detected window ids of its subsets) and
            the range of its NAB false positives (max .. sum of the subsets' counts).
  evaluate  (on the HPC, needs the results folders the file paths of hpc.csv.gz point to): the exact metrics of
            every combination from the saved test predictions (<model>_predictions_for_assembles_<norm>.parquet,
            its columns in the order of the rows of the grid-search CSV next to it); ensemble_results.csv
            and ensemble_summary.csv. Combinations of size 1 are the subsets themselves: their recomputed
            reward_fn score is checked against the grid-search one (check_diff column).

From the project root:
    python src/ensemble_subsets.py --select-only                         # selection + preview, locally
    python src/ensemble_subsets.py --workers 8                           # selection + exact evaluation (HPC)
    python src/ensemble_subsets.py --sizes 2 3 8 --windows 12 --normalizations global
"""
import argparse
import ast
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ClouDens-A3T* and ClouDens-BigST-ctx* (context graph as the prior of the learned graph)
DEFAULT_MODELS = ('A3TGCN_null_padding_feature', 'BigST_null_padding_feature_spatial')
DEFAULT_SIZES = (1, 2, 3, 8)
STRATEGIES = ('likelihood', 'likelihood_mahalanobis', 'mean_reconstruction_errors')
THRESHOLD_TYPES = ('absolute', 'percentile')
# The members of an ensemble share the scoring strategy and the threshold type; each subset has its own
# threshold value, top-k, likelihood windows and imputation within them
ENSEMBLE_KEY = ['post_processing_strategy', 'threshold_type']
# Imputation fixed for the subsets of an aggregation (a missing request count is no request: zero); the
# other subsets take the best one
FIXED_IMPUTATION = {'count': 'zero'}
# Columns ranking the settings of a subset, in order (ties broken by the next ones, then by the row order)
SELECTION_CRITERIA = {
    'train_valid': ['train_valid_reward_fn_normalized', 'train_valid_standard_normalized', 'train_valid_f1'],
    'test': ['reward_fn_normalized', 'standard_normalized', 'f1'],
}
# Identify a setting within a grid-search CSV (unique there), i.e. its column of the predictions file
SETTING_KEY = ['post_processing_strategy', 'threshold_type', 'anomaly_threshold', 'topk', 'long_window', 'short_window']
GROUP_KEY = ['model_folder', 'window', 'score_normalization']
METRICS = ['standard_normalized', 'reward_fn_normalized', 'precision', 'recall', 'f1']
HPC_COLUMNS = (GROUP_KEY + ['subset', 'fill_nan', 'model', 'file', 'detection_counters', 'confusion_matrix']
               + SETTING_KEY + METRICS + ['standard_raw', 'reward_fn_raw']
               + sorted({c for criteria in SELECTION_CRITERIA.values() for c in criteria}))


def short_subset(subset):
    """'no_group_5xx_count' -> '5xx_count'."""
    return subset.removeprefix('no_group_')


def load_hpc_rows(path, models, windows=None, normalizations=None):
    """Rows of the merged grid-search table for the model folders (and windows / normalisations), read in chunks."""
    parts = []
    for chunk in pd.read_csv(path, usecols=HPC_COLUMNS, chunksize=500_000):
        keep = chunk['model_folder'].isin(models)
        if windows:
            keep &= chunk['window'].isin(windows)
        if normalizations:
            keep &= chunk['score_normalization'].isin(normalizations)
        parts.append(chunk[keep])
    rows = pd.concat(parts, ignore_index=True)
    missing = set(models) - set(rows['model_folder'])
    if missing:
        raise ValueError(f'no rows of {sorted(missing)} in {path}')
    return rows


def select_configs(rows, selections, strategies, threshold_types):
    """One setting per (group, selection, strategy, threshold type, subset): the best on the selection criteria."""
    rows = rows[rows['post_processing_strategy'].isin(strategies) & rows['threshold_type'].isin(threshold_types)]
    for aggregation, imputation in FIXED_IMPUTATION.items():
        rows = rows[~rows['subset'].str.endswith(f'_{aggregation}') | (rows['fill_nan'] == imputation)]
    selected = []
    for selection in selections:
        criteria = SELECTION_CRITERIA[selection]
        # Stable sort: equal criteria keep the row order of the table, so the choice is reproducible
        ranked = rows.sort_values(criteria, ascending=False, kind='mergesort')
        best = ranked.groupby(GROUP_KEY + ENSEMBLE_KEY + ['subset'], sort=True).head(1)
        selected.append(best.assign(selection=selection))
    selected = pd.concat(selected, ignore_index=True)
    counters = selected['detection_counters'].map(ast.literal_eval)
    selected['detected_window_ids'] = counters.map(
        lambda c: sorted(c['issue_detected_ids'] + c['im_detected_ids'] + c['TestLog_detected_ids']))
    selected['num_windows'] = counters.map(lambda c: len(c['gt_issue_ids'] + c['gt_im_ids'] + c['gt_TestLog_ids']))
    selected['nab_fp'] = counters.map(lambda c: c['fp'])
    front = (['selection'] + ENSEMBLE_KEY + GROUP_KEY + ['subset', 'fill_nan']
             + [c for c in SETTING_KEY if c not in ENSEMBLE_KEY])
    return selected[front + [c for c in selected.columns if c not in front]]


def combinations_of(subsets, sizes):
    """Sorted subset combinations of the given sizes (sizes larger than the number of subsets skipped)."""
    subsets = sorted(subsets)
    return [combo for size in sizes if size <= len(subsets) for combo in itertools.combinations(subsets, size)]


def union_preview(selected, sizes):
    """Per combination: windows detected by the OR ensemble (exact) and the range of its NAB false positives."""
    records = []
    for key, part in selected.groupby(['selection'] + ENSEMBLE_KEY + GROUP_KEY, sort=True):
        by_subset = part.set_index('subset')
        for combo in combinations_of(by_subset.index, sizes):
            members = by_subset.loc[list(combo)]
            detected = set().union(*members['detected_window_ids'])
            records.append(dict(zip(['selection'] + ENSEMBLE_KEY + GROUP_KEY, key),
                                size=len(combo), combination='+'.join(map(short_subset, combo)),
                                windows_detected=len(detected), num_windows=int(members['num_windows'].iloc[0]),
                                detected_window_ids=sorted(detected),
                                nab_fp_min=int(members['nab_fp'].max()), nab_fp_max=int(members['nab_fp'].sum()),
                                best_member_reward_fn=members['reward_fn_normalized'].max()))
    return pd.DataFrame(records)


def _group_inputs(group_selected):
    """Test predictions of the selected setting of each subset, the test timestamps and labels of the group."""
    from model_outputs import existing_anomaly_scores_path, read_anomaly_scores_file, read_predictions_for_assembles

    grid_cache, prediction_cache = {}, {}
    predictions, timeline = {}, None
    for row in group_selected.itertuples(index=False):
        model_dir = os.path.join(PROJECT_ROOT, os.path.dirname(row.file))
        grid_file = os.path.join(PROJECT_ROOT, row.file)
        if grid_file not in grid_cache:
            grid_cache[grid_file] = pd.read_csv(grid_file, low_memory=False)
        grid = grid_cache[grid_file]
        match = np.ones(len(grid), dtype=bool)
        for column in SETTING_KEY:
            value = getattr(row, column)
            if isinstance(value, str):
                match &= (grid[column] == value).to_numpy()
            else:   # NaN topk = mean over the sensors
                match &= np.isclose(grid[column].astype(float), float(value), equal_nan=True)
        if match.sum() != 1:
            raise RuntimeError(f'{match.sum()} rows of {grid_file} match the setting {row}')
        column_index = int(np.flatnonzero(match)[0])
        if not np.isclose(grid['reward_fn_normalized'].iloc[column_index], row.reward_fn_normalized):
            print(f'Warning: {grid_file} differs from the merged table (reward_fn '
                  f'{grid["reward_fn_normalized"].iloc[column_index]} vs {row.reward_fn_normalized})', flush=True)

        if model_dir not in prediction_cache:
            frame = read_predictions_for_assembles(model_dir, row.model, row.score_normalization)
            if frame is None:
                raise FileNotFoundError(f'no predictions_for_assembles file in {model_dir}')
            if frame.shape[1] != len(grid):
                raise RuntimeError(f'{model_dir}: {frame.shape[1]} prediction columns, {len(grid)} grid-search rows')
            prediction_cache[model_dir] = frame.to_numpy(dtype=np.int8)
        member = (row.selection, row.post_processing_strategy, row.threshold_type, row.subset)
        predictions[member] = prediction_cache[model_dir][:, column_index]

        if timeline is None:   # the test timestamps and labels, the same for every subset of a window
            scores = read_anomaly_scores_file(existing_anomaly_scores_path(model_dir, row.model, row.score_normalization))
            test = scores[scores['split'] == 'test']
            timeline = (pd.DatetimeIndex(test['timestamp']), test['is_anomaly'].to_numpy().astype(int))
        if len(predictions[member]) != len(timeline[0]):
            raise RuntimeError(f'{model_dir}: predictions and test timestamps differ in length')
    return predictions, timeline


def evaluate_prediction(prediction, test_index, labels, anomaly_windows_test):
    """NAB standard / reward_fn and point metrics of a 0 / 1 prediction, as for the single models."""
    from nab_scoring import calculate_nab_score_with_window_based_tp_fn
    from run_training_single_model import evaluate_performance

    frame = pd.DataFrame({'true_anomaly': labels, 'predicted_anomaly': prediction}, index=test_index)
    result = {}
    for profile in ('standard', 'reward_fn'):
        raw, normalized, fp, fn, counters = calculate_nab_score_with_window_based_tp_fn(
            frame, anomaly_windows_test, profile, true_col='true_anomaly', pred_col='predicted_anomaly')
        result.update({f'{profile}_raw': raw, f'{profile}_normalized': normalized})
    precision, recall, f1, _, conf_matrix, _ = evaluate_performance(labels, prediction)
    result.update(precision=precision, recall=recall, f1=f1, confusion_matrix=conf_matrix,
                  nab_fp=counters['fp'], windows_detected=counters['tp'],
                  detected_window_ids=sorted(counters['issue_detected_ids'] + counters['im_detected_ids']
                                             + counters['TestLog_detected_ids']))
    return result


def evaluate_group(task):
    """Metrics of every combination of one (model folder, window, normalisation)."""
    group_key, group_selected, sizes, anomaly_windows_test = task
    predictions, (test_index, labels) = _group_inputs(group_selected)
    records = []
    for (selection, strategy, threshold_type), part in group_selected.groupby(['selection'] + ENSEMBLE_KEY, sort=True):
        by_subset = part.set_index('subset')
        for combo in combinations_of(by_subset.index, sizes):
            ensemble = np.zeros(len(labels), dtype=np.int8)
            for subset in combo:
                ensemble |= predictions[(selection, strategy, threshold_type, subset)]
            record = dict(selection=selection, post_processing_strategy=strategy, threshold_type=threshold_type,
                          **dict(zip(GROUP_KEY, group_key)),
                          size=len(combo), combination='+'.join(map(short_subset, combo)),
                          **evaluate_prediction(ensemble, test_index, labels, anomaly_windows_test))
            if len(combo) == 1:   # the subset itself: must reproduce its grid-search score
                record['check_diff'] = record['reward_fn_normalized'] - by_subset.loc[combo[0], 'reward_fn_normalized']
            records.append(record)
    print(f'Evaluated {group_key}: {len(records)} combinations', flush=True)
    return records


def summarize(results):
    """Per group, selection, strategy group and size: the best / median combination against the best single subset."""
    keys = ['selection'] + ENSEMBLE_KEY + GROUP_KEY
    singles = results[results['size'] == 1].groupby(keys)['reward_fn_normalized'].max().rename('best_single_reward_fn')
    records = []
    for key, part in results.groupby(keys + ['size'], sort=True):
        best = part.loc[part['reward_fn_normalized'].idxmax()]
        records.append(dict(zip(keys + ['size'], key), num_combinations=len(part),
                            best_combination=best['combination'], best_reward_fn=best['reward_fn_normalized'],
                            best_standard=best['standard_normalized'], best_f1=best['f1'],
                            best_windows_detected=best['windows_detected'], best_nab_fp=best['nab_fp'],
                            median_reward_fn=part['reward_fn_normalized'].median(),
                            mean_reward_fn=part['reward_fn_normalized'].mean()))
    summary = pd.DataFrame(records).join(singles, on=keys)
    beating = results.join(singles, on=keys)
    beating = beating[beating['reward_fn_normalized'] > beating['best_single_reward_fn']]
    summary = summary.join(beating.groupby(keys + ['size']).size().rename('num_beating_best_single'), on=keys + ['size'])
    summary['num_beating_best_single'] = summary['num_beating_best_single'].fillna(0).astype(int)
    return summary


def main():
    parser = argparse.ArgumentParser(description='OR ensembles of the subsets of the ClouDens models with null padding.')
    parser.add_argument('--hpc-csv', default=os.path.join(PROJECT_ROOT, 'merged_results', 'hpc.csv.gz'))
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS), help='Model folders')
    parser.add_argument('--sizes', nargs='+', type=int, default=list(DEFAULT_SIZES),
                        help='Numbers of subsets per combination (1: the subsets alone, as the reference)')
    parser.add_argument('--selections', nargs='+', default=list(SELECTION_CRITERIA), choices=list(SELECTION_CRITERIA))
    parser.add_argument('--strategies', nargs='+', default=list(STRATEGIES), choices=list(STRATEGIES),
                        help='Scoring strategies, each ensembled on its own (shared by the members)')
    parser.add_argument('--threshold-types', nargs='+', default=list(THRESHOLD_TYPES), choices=list(THRESHOLD_TYPES),
                        help='Threshold types, each ensembled on its own (shared by the members)')
    parser.add_argument('--windows', nargs='+', type=int, help='Sliding windows (default: all)')
    parser.add_argument('--normalizations', nargs='+', choices=['global', 'per_node'], help='Default: both')
    parser.add_argument('--out-dir', default=os.path.join(PROJECT_ROOT, 'merged_results', 'ensemble_subsets'))
    parser.add_argument('--select-only', action='store_true', help='Selection and window-union preview only')
    parser.add_argument('--workers', type=int, default=1, help='Processes evaluating the groups in parallel')
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'conf', 'config.yaml'))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_hpc_rows(args.hpc_csv, args.models, args.windows, args.normalizations)
    print(f'{len(rows)} grid-search rows of {", ".join(args.models)}', flush=True)
    selected = select_configs(rows, args.selections, args.strategies, args.threshold_types)
    selected_file = os.path.join(args.out_dir, 'selected_configs.csv')
    selected.to_csv(selected_file, index=False)
    print(f'Selected settings ({len(selected)}) saved to {selected_file}', flush=True)
    preview = union_preview(selected, args.sizes)
    preview_file = os.path.join(args.out_dir, 'window_union_preview.csv')
    preview.to_csv(preview_file, index=False)
    print(f'Window-union preview ({len(preview)} combinations) saved to {preview_file}', flush=True)
    if args.select_only:
        return

    from omegaconf import OmegaConf
    from rerun_grid_search import load_anomaly_windows
    anomaly_windows_test = load_anomaly_windows(OmegaConf.load(args.config))[1]
    tasks = [(key, part, args.sizes, anomaly_windows_test) for key, part in selected.groupby(GROUP_KEY, sort=True)]
    with Pool(args.workers) as pool:
        results = pd.DataFrame([r for records in pool.imap(evaluate_group, tasks) for r in records])
    results_file = os.path.join(args.out_dir, 'ensemble_results.csv')
    results.to_csv(results_file, index=False)
    print(f'Ensemble results ({len(results)} combinations) saved to {results_file}', flush=True)
    checks = results['check_diff'].dropna().abs() if 'check_diff' in results else pd.Series(dtype=float)
    if len(checks):
        print(f'Single subsets reproduced: {int((checks < 1e-6).sum())} / {len(checks)} '
              f'(max |difference| {checks.max():.3g})', flush=True)
    summary_file = os.path.join(args.out_dir, 'ensemble_summary.csv')
    summarize(results).to_csv(summary_file, index=False)
    print(f'Summary saved to {summary_file}', flush=True)


if __name__ == '__main__':
    main()
