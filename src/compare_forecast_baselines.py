"""
Do the forecasting errors of the models say more about the anomalies than trivial forecasts? For every experiment
folder (<root>/window_<w>/<subset>/fill_nan_with_<fill>/) the saved errors of each model folder are compared with
those of two naive forecasts of the same scaled data: 'persistence' (the value at t-1) and 'zero' (predicting 0,
so the error is the data itself). Every error series is turned into the anomaly scores of the pipeline (mean error,
top-k of the median/IQR-normalised errors, Mahalanobis distance fitted on the clean train/valid windows) and
scored on the test split: AUC-PR and best F1 (over all thresholds), plus the forecasting MAE / MSE.

From the project root:
    python src/compare_forecast_baselines.py trained_models_log1p                     # every experiment folder
    python src/compare_forecast_baselines.py trained_models_log1p --subsets no_group_5xx_count --windows 6
Writes <root>/forecast_baseline_comparison.csv and prints the table.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_outputs import PREDICTION_FILES, aggregate_sensor_scores, existing_anomaly_scores_path, read_anomaly_scores_file
from run_training_single_model import _best_f1_threshold
from utils import MahalanobisScorer, get_full_err_scores

NAIVE_FORECASTS = ('persistence', 'zero')
TOPKS = (1, 5, None)


def scores_of(errors, train_reference, mahalanobis_reference):
    """Anomaly scores [T] of the errors [T, D]: name -> series, as the pipeline's strategies build them."""
    scores = {'mean error': errors.mean(axis=1)}
    for normalization in ('per_node', 'global'):
        z = get_full_err_scores(errors[:, :, None], normalization, reference_errors=train_reference[:, :, None])[:, :, 0]
        for topk in TOPKS:
            scores[f'{normalization} top-{topk or "mean"}'] = aggregate_sensor_scores(z, topk)
    scorer = MahalanobisScorer(covariance='ledoit_wolf', min_variance=1e-6).fit(mahalanobis_reference)
    scores['mahalanobis (train_valid)'] = scorer.score(errors)[0]
    return scores


def compare_folder(experiment_dir):
    """Rows (one per forecast and score) of an experiment folder; [] when it has no saved predictions."""
    model_dirs = sorted(d for d in glob.glob(os.path.join(experiment_dir, '*/'))
                        if all(os.path.exists(os.path.join(d, f)) for f in PREDICTION_FILES.values()))
    if not model_dirs:
        return []
    # Splits, labels and clean windows from the saved scores of the first model (the same for all of them)
    first = model_dirs[0]
    model_name = os.path.basename(glob.glob(os.path.join(first, '*_anomaly_scores_*'))[0]).split('_anomaly_scores_')[0]
    info = read_anomaly_scores_file(existing_anomaly_scores_path(first, model_name, 'global'))
    split = info['split'].to_numpy()
    test = split == 'test'
    labels = info.loc[test, 'is_anomaly'].to_numpy().astype(int)
    clean_train = (split == 'train') & (info['is_clean_window'].to_numpy() == 1)
    clean_train_valid = np.isin(split, ('train', 'valid')) & (info['is_clean_window'].to_numpy() == 1)
    timestamps = pd.DatetimeIndex(info['timestamp'])

    actual = pd.read_parquet(os.path.join(first, PREDICTION_FILES['actual'])).loc[timestamps].to_numpy(np.float32)
    nonzero_test = actual[test] != 0

    def errors_of():
        """(forecast, errors [T, D]) one at a time: all of them at once do not fit in memory."""
        yield 'persistence', np.abs(actual - np.vstack([actual[:1], actual[:-1]]))
        yield 'zero', np.abs(actual)
        for model_dir in model_dirs:
            predicted = pd.read_parquet(os.path.join(model_dir, PREDICTION_FILES['predicted'])).loc[timestamps]
            yield os.path.basename(os.path.normpath(model_dir)), np.abs(actual - predicted.to_numpy(np.float32))

    rows = []
    for forecast, errors in errors_of():
        test_errors = errors[test]
        base = {'experiment': os.path.relpath(experiment_dir), 'forecast': forecast,
                'naive': forecast in NAIVE_FORECASTS, 'test_mae': test_errors.mean(),
                'test_mse': (test_errors.astype(np.float64) ** 2).mean(), 'test_mae_nonzero': test_errors[nonzero_test].mean()}
        for score_name, score in scores_of(errors, errors[clean_train], errors[clean_train_valid]).items():
            best_f1, _ = _best_f1_threshold(score[test], labels)
            rows.append({**base, 'score': score_name, 'auc_pr': average_precision_score(labels, score[test]),
                         'best_f1': best_f1})
        print(f'  {forecast}: done', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Compare the forecasting errors of the models with naive forecasts.')
    parser.add_argument('root', help='Results folder, e.g. trained_models_log1p')
    parser.add_argument('--subsets', nargs='+', help='e.g. no_group_5xx_count (default: all)')
    parser.add_argument('--windows', nargs='+', type=int, help='e.g. 6 12 (default: all)')
    parser.add_argument('--fill-nan', nargs='+', default=['zero'], help='fill_nan values (default: zero)')
    args = parser.parse_args()

    experiments = sorted(d for d in glob.glob(os.path.join(args.root, 'window_*/*/fill_nan_with_*/'))
                         if (not args.subsets or d.split(os.sep)[-3] in args.subsets)
                         and (not args.windows or int(d.split(os.sep)[-4].removeprefix('window_')) in args.windows)
                         and d.split(os.sep)[-2].removeprefix('fill_nan_with_') in args.fill_nan)
    rows = []
    for experiment_dir in experiments:
        print(experiment_dir, flush=True)
        rows += compare_folder(experiment_dir)
    if not rows:
        raise SystemExit('No experiment folder with saved predictions')
    table = pd.DataFrame(rows)
    out = os.path.join(args.root, 'forecast_baseline_comparison.csv')
    table.to_csv(out, index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        for experiment, part in table.groupby('experiment'):
            print(f'\n{experiment}')
            print(part.pivot_table(index='forecast', columns='score', values='auc_pr', sort=False).round(3))
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
