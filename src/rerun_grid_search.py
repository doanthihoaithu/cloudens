"""
Grid search of some post-processing strategies (default: mahalanobis) from the saved anomaly scores of a results
folder (<model>_anomaly_scores_<normalization>.parquet, or the earlier CSVs), without retraining nor predicting
again: their settings are evaluated as in run_training_single_model.grid_search_new (same thresholds, metrics,
NAB scores, conf/config.yaml evaluation settings) and replace the rows of these strategies in the grid-search CSV
next to the scores; the rows of the other strategies are kept, the NAB ranks recomputed over all of them.

With --recompute-mahalanobis, the Mahalanobis distances are first computed again from the saved forecasting
errors (reconstruction_error.parquet) with the current mahalanobis_reference / mahalanobis_covariance of the
config, and the Mahalanobis columns of the score files (distance and its likelihoods) replaced; the grid search
then covers the strategies on them (mahalanobis and likelihood_mahalanobis).

From the project root:
    python src/rerun_grid_search.py trained_models_log1p --dry-run                 # files that would be updated
    python src/rerun_grid_search.py trained_models_log1p                           # mahalanobis everywhere
    python src/rerun_grid_search.py trained_models_log1p --models BigST A3TGCN     # only these model folders
    python src/rerun_grid_search.py trained_models_log1p --strategies mahalanobis mean_reconstruction_errors
    python src/rerun_grid_search.py trained_models_log1p --recompute-mahalanobis      # new Mahalanobis settings
"""
import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ibm_dataset_loader import IBMDatasetLoader
from model_outputs import (IS_NAN_FILES, MAHALANOBIS_COLUMN, MEAN_ERROR_COLUMN, PREDICTION_FILES, SCORE_INFO_COLUMNS,
                           SPLITS, compute_mahalanobis_scores, existing_anomaly_scores_path, likelihood_window_pairs,
                           load_predictions, mahalanobis_likelihood_column, mahalanobis_score_columns,
                           read_anomaly_scores_file, save_anomaly_scores)
from run_training_single_model import EVALUATION_SPLITS, SCORE_STRATEGIES, grid_search_new
from utils import get_project_root, scoring_result_file_name

SCORE_FILE_PATTERN = re.compile(r'^(?P<model>.+)_anomaly_scores_(?P<normalization>global|per_node)\.(parquet|csv)$')
STRATEGY_COLUMNS = {'mahalanobis': MAHALANOBIS_COLUMN, 'mean_reconstruction_errors': MEAN_ERROR_COLUMN,
                    'likelihood_mahalanobis': mahalanobis_likelihood_column(30, 2, True)}
# Strategies scored on the Mahalanobis distances, whose grid-search rows change with them
MAHALANOBIS_STRATEGIES = ('mahalanobis', 'likelihood_mahalanobis')
NULL_PADDING = re.compile(r'_null_padding_(feature|target|both)')


def find_score_files(root, models=None):
    """(model folder, model, normalization) of every anomaly score file under root, the parquet preferred."""
    found = {}
    for path in sorted(Path(root).glob('window_*/*/fill_nan_with_*/*/*_anomaly_scores_*')):
        match = SCORE_FILE_PATTERN.match(path.name)
        if match and (not models or any(path.parent.name.startswith(m) for m in models)):
            found[(path.parent, match['model'], match['normalization'])] = True
    return list(found)


def load_anomaly_windows(cfg):
    """(all, test) ground-truth anomaly windows, as the data loader reads them (IBMDatasetLoader)."""
    data_config = cfg.data_preparation_pipeline
    loader = SimpleNamespace(
        anomaly_windows_file=os.path.join(get_project_root(), data_config.ground_truth_labels.path,
                                          data_config.ground_truth_labels.filename),
        data_preparation_config=data_config)
    return IBMDatasetLoader._load_anomaly_windows(loader)


def experiment_config_of(cfg, model_dir, model, strategies):
    """The evaluation settings of the run that wrote the scores in model_dir, with only the given strategies."""
    config = OmegaConf.to_container(cfg.evaluation, resolve=True)
    padding = NULL_PADDING.search(model_dir.name)
    config.update(use_model=model, fill_nan=model_dir.parent.name.removeprefix('fill_nan_with_'),
                  null_padding_feature=bool(padding) and padding[1] in ('feature', 'both'),
                  null_padding_target=bool(padding) and padding[1] in ('target', 'both'),
                  post_processing_strategies=list(strategies))
    return OmegaConf.create(config)


def scorer_args_of(experiment_config):
    """MahalanobisScorer settings of the config, as run_training_single_model builds them."""
    return dict(covariance=experiment_config.get('mahalanobis_covariance', 'empirical'),
                min_variance=experiment_config.get('mahalanobis_min_variance', 0.0),
                pca_variance=experiment_config.get('mahalanobis_pca_variance', 0.95),
                pca_quantile=experiment_config.get('mahalanobis_pca_quantile', 0.99))


def recompute_mahalanobis(model_dir, model, normalizations, experiment_config):
    """
    Mahalanobis distances computed again from the saved forecasting errors of model_dir, with the config's
    reference split and scorer, written into the score files of the normalizations (their other columns kept).
    The splits and the windows without anomalies are those of the saved scores (timestamp, split,
    is_clean_window columns), as the data loader made them.
    """
    if not all((model_dir / f).exists() for f in PREDICTION_FILES.values()):
        raise FileNotFoundError(f'no saved predictions ({", ".join(PREDICTION_FILES.values())})')
    scores = read_anomaly_scores_file(existing_anomaly_scores_path(str(model_dir), model, normalizations[0]))
    split_of = {s: scores[scores['split'] == s] for s in SPLITS}
    num_columns = pd.read_parquet(model_dir / PREDICTION_FILES['error']).shape[1]
    is_nan_file = model_dir / IS_NAN_FILES['predicted']
    num_nodes = pd.read_parquet(is_nan_file).shape[1] if is_nan_file.exists() else num_columns
    # What load_predictions / compute_mahalanobis_scores read from the data loader
    data_loader = SimpleNamespace(
        split_index={s: pd.DatetimeIndex(part['timestamp']) for s, part in split_of.items()},
        train_clean_mask=split_of['train']['is_clean_window'].to_numpy() == 1,
        valid_clean_mask=split_of['valid']['is_clean_window'].to_numpy() == 1,
        num_nodes=num_nodes, num_node_features=num_columns // num_nodes)
    outputs = load_predictions(data_loader, str(model_dir))
    scorer_args = scorer_args_of(experiment_config)
    distances, reference_distances, _ = compute_mahalanobis_scores(outputs, data_loader, experiment_config, scorer_args)
    window_pairs, with_windows = likelihood_window_pairs(experiment_config)
    columns = mahalanobis_score_columns(distances, reference_distances, window_pairs, with_windows)
    for normalization in normalizations:
        scores = read_anomaly_scores_file(existing_anomaly_scores_path(str(model_dir), model, normalization))
        updated = scores[list(SCORE_INFO_COLUMNS)].assign(**columns)
        save_anomaly_scores(updated, str(model_dir), model, experiment_config, scorer_args,
                            predictions_recomputed=False, normalization=normalization)


def rerun(model_dir, model, normalization, strategies, cfg, anomaly_windows):
    """Grid search of the strategies on the saved scores; returns the updated grid-search table."""
    scores = read_anomaly_scores_file(existing_anomaly_scores_path(str(model_dir), model, normalization))
    missing = [STRATEGY_COLUMNS[s] for s in strategies if STRATEGY_COLUMNS[s] not in scores]
    if missing:
        raise ValueError(f'scores without the columns {missing}')
    test_scores = scores[scores['split'] == 'test'].reset_index(drop=True)
    split_scores = {name: scores[scores['split'].isin(splits)].reset_index(drop=True)
                    for name, splits in EVALUATION_SPLITS.items()}
    # What grid_search_new reads from the data loader, rebuilt from the scores (the same test timestamps and labels)
    test_index = pd.DatetimeIndex(test_scores['timestamp'])
    data_loader = SimpleNamespace(test_index=test_index,
                                  test_labels=pd.Series(test_scores['is_anomaly'].to_numpy(), index=test_index),
                                  anomaly_windows=anomaly_windows[0], anomaly_windows_test=anomaly_windows[1],
                                  count_5xx=np.zeros(len(test_index)))   # only shown in a debug print
    result_df, _ = grid_search_new(data_loader, test_scores, experiment_config_of(cfg, model_dir, model, strategies),
                                   split_scores=split_scores, score_normalization=normalization)

    grid_file = model_dir / scoring_result_file_name(model, 'grid_search', normalization)
    if grid_file.exists():
        previous = pd.read_csv(grid_file, low_memory=False)
        kept = previous[~previous['post_processing_strategy'].isin(strategies)].drop(
            columns=['NAB_standard_rank', 'NAB_reward_fn_rank'], errors='ignore')
        result_df = pd.concat([kept, result_df], ignore_index=True)
    result_df.insert(1, 'NAB_standard_rank', result_df['standard_normalized'].rank(ascending=False))
    result_df.insert(2, 'NAB_reward_fn_rank', result_df['reward_fn_normalized'].rank(ascending=False))
    return grid_file, result_df


def main():
    parser = argparse.ArgumentParser(description='Grid search of post-processing strategies from saved anomaly scores.')
    parser.add_argument('root', help='Results folder, e.g. trained_models_log1p')
    parser.add_argument('--strategies', nargs='+', default=['mahalanobis'], choices=list(STRATEGY_COLUMNS),
                        help='Strategies whose grid-search rows are (re)computed (default: mahalanobis)')
    parser.add_argument('--recompute-mahalanobis', action='store_true',
                        help='Compute the Mahalanobis distances again from the saved forecasting errors with the '
                             'mahalanobis_reference / mahalanobis_covariance of the config (then grid search of '
                             'mahalanobis and likelihood_mahalanobis)')
    parser.add_argument('--models', nargs='+', help='Only the model folders starting with these names')
    parser.add_argument('--config', default=os.path.join(get_project_root(), 'conf', 'config.yaml'))
    parser.add_argument('--dry-run', action='store_true', help='Only list the score files')
    args = parser.parse_args()

    files = find_score_files(args.root, args.models)
    strategies = list(dict.fromkeys(args.strategies + (list(MAHALANOBIS_STRATEGIES) if args.recompute_mahalanobis else [])))
    print(f'{len(files)} anomaly score files under {args.root}, strategies: {", ".join(strategies)}' +
          (', Mahalanobis distances recomputed' if args.recompute_mahalanobis else ''))
    if args.dry_run:
        for model_dir, model, normalization in files:
            print(f'  {model_dir} ({model}, {normalization})')
        return
    cfg = OmegaConf.load(args.config)
    anomaly_windows = load_anomaly_windows(cfg)

    failed, total = [], len(files)
    if args.recompute_mahalanobis:
        # Once per model folder for all its score normalisations (the distances do not depend on them)
        folders = {}
        for model_dir, model, normalization in files:
            folders.setdefault((model_dir, model), []).append(normalization)
        for i, ((model_dir, model), normalizations) in enumerate(folders.items(), 1):
            try:
                recompute_mahalanobis(model_dir, model, normalizations, experiment_config_of(cfg, model_dir, model, strategies))
                print(f'[{i}/{len(folders)}] Mahalanobis recomputed: {model_dir}', flush=True)
            except Exception as e:
                print(f'[{i}/{len(folders)}] FAILED Mahalanobis {model_dir}: {type(e).__name__}: {e}', flush=True)
                failed += [(model_dir, n, f'Mahalanobis not recomputed: {type(e).__name__}: {e}') for n in normalizations]
        not_recomputed = {(d, n) for d, n, _ in failed}
        files = [f for f in files if (f[0], f[2]) not in not_recomputed]   # no grid search on the old distances
    for i, (model_dir, model, normalization) in enumerate(files, 1):
        try:
            grid_file, result_df = rerun(model_dir, model, normalization, strategies, cfg, anomaly_windows)
            tmp_file = grid_file.with_name(grid_file.name + '.tmp')
            result_df.to_csv(tmp_file, index=False)
            os.replace(tmp_file, grid_file)   # a run stopped midway leaves the previous grid search intact
            print(f'[{i}/{len(files)}] {grid_file} ({len(result_df)} settings)', flush=True)
        except Exception as e:   # e.g. a truncated score file
            failed.append((model_dir, normalization, f'{type(e).__name__}: {e}'))
            print(f'[{i}/{len(files)}] FAILED {model_dir} ({normalization}): {failed[-1][2]}', flush=True)
    print(f'Updated {total - len(failed)} / {total} grid-search files')
    for model_dir, normalization, error in failed:
        print(f'Not updated: {model_dir} ({normalization}): {error}')


if __name__ == '__main__':
    main()
