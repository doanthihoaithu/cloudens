"""
Per-model outputs on all data splits (train / valid / test):

  • predictions saved as parquet files of shape [T, N·F] (one row per timestamp of the
    three splits, one column per node / feature):
        scaled_actual_data.parquet, predicted_data.parquet, reconstruction_error.parquet
    plus predicted_is_nan.parquet / actual_is_nan.parquet [T, N] when the target is null-padded;
  • anomaly scores saved as a CSV with one row per timestamp:
        timestamp, split, is_anomaly, is_clean_window,
        anomaly_score_postprocessed_by_mean_reconstruction_errors,
        anomaly_score_postprocessed_by_likelihood_with_top_{k},
        anomaly_score_postprocessed_by_likelihood_with_mahalanobis,
        anomaly_score_postprocessed_by_mahalanobis
    The CSV is updated in place: re-running with other scoring settings adds / overwrites
    columns while keeping the others (unless the predictions themselves were recomputed).

Scoring statistics are fitted on windows without anomalies and applied to every split: the
median / IQR of the errors on the train windows, the Mahalanobis mean / covariance on the
train, valid or train + valid windows (mahalanobis_reference; 'test' uses all its windows). The anomaly
likelihood is computed separately on each split.
"""
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

from anomaly_likelihood import compute_anomaly_likelihood
from utils import (get_full_err_scores, has_is_nan_mask, apply_is_nan_mask, MahalanobisScorer, score_normalizations,
                   scoring_result_file_name)

log = logging.getLogger(__name__)

SPLITS = ('train', 'valid', 'test')
# Splits on which the anomaly likelihood runs as one series: valid directly follows train in time,
# so its likelihood continues the train history; test is a separate period with its own history
LIKELIHOOD_SEGMENTS = (('train', 'valid'), ('test',))
PREDICTION_FILES = {
    'actual': 'scaled_actual_data.parquet',
    'predicted': 'predicted_data.parquet',
    'error': 'reconstruction_error.parquet',
}
IS_NAN_FILES = {'predicted': 'predicted_is_nan.parquet', 'actual': 'actual_is_nan.parquet'}
# Full-precision float32 predictions barely compress (every value is distinct). Rounding to 4
# decimals (max error 5e-5, ~1% of a typical reconstruction error) makes values repeat, so the
# default dictionary encoding + zstd is ~2.5x smaller than raw float32
PREDICTION_DECIMALS = 4
PARQUET_OPTIONS = dict(compression='zstd')

MEAN_ERROR_COLUMN = 'anomaly_score_postprocessed_by_mean_reconstruction_errors'
# Columns of score CSVs written by earlier versions: {old name: current name, or None if dropped}
LEGACY_SCORE_COLUMNS = {
    'is_clean_train_window': None,   # replaced by is_clean_window
    'anomaly_score_postprocessed_by_mean_reconstruction_error': MEAN_ERROR_COLUMN,
}
MAHALANOBIS_COLUMN = 'anomaly_score_postprocessed_by_mahalanobis'


def parse_topk(value):
    """topk from the config or a grid-search CSV: None / NaN (mean over all sensors) or an int.
    The strings 'None' / 'null' are accepted too, since YAML reads `None` as a string."""
    if isinstance(value, str) and value.strip().lower() in ('none', 'null', ''):
        return None
    return None if value is None or pd.isna(value) else int(value)


def aggregate_sensor_scores(scores, topk):
    """[T, D] sensor scores → [T]: mean of the top-k largest per timestamp, or of all sensors when topk is None."""
    if topk is None:
        return scores.mean(axis=-1)
    topk = min(topk, scores.shape[1])
    # Partial sort: only the top-k values are needed, not the order of all sensors
    return np.partition(scores, -topk, axis=1)[:, -topk:].mean(axis=-1)


def top_k_indices(scores, topk):
    """[T, D] → [T, k] indices of the top-k largest scores per timestamp, in ascending order of
    score (as scores.argsort(axis=1)[:, -topk:], without sorting all D sensors)."""
    topk = min(topk, scores.shape[1])
    idx = np.argpartition(scores, -topk, axis=1)[:, -topk:]
    order = np.argsort(np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def likelihood_column(topk, long_window, short_window, with_windows=False):
    """Score column of the likelihood post-processing; the windows are part of the name only
    when several (long, short) window pairs are configured."""
    name = f'anomaly_score_postprocessed_by_likelihood_with_top_{"mean" if topk is None else topk}'
    if with_windows:
        name += f'_long_{long_window}_short_{short_window}'
    return name


def mahalanobis_likelihood_column(long_window, short_window, with_windows=False):
    """Score column of the likelihood of the min-max scaled Mahalanobis distances."""
    name = 'anomaly_score_postprocessed_by_likelihood_with_mahalanobis'
    if with_windows:
        name += f'_long_{long_window}_short_{short_window}'
    return name


def likelihood_window_pairs(experiment_config):
    pairs = [(int(lw), int(sw)) for lw in experiment_config.long_windows for sw in experiment_config.short_windows]
    return pairs, len(pairs) > 1


# ── Predictions ───────────────────────────────────────────────────────────────

def predict_all_splits(model_wrapper, data_loader, batch_size, device):
    """
    Predict every window of the three splits (train windows with anomalies included).
    Returns {split: {'actual', 'predicted', 'error': [L, N, F], 'is_nan': [2, L, N] or None}}.
    """
    N, F = data_loader.num_nodes, data_loader.num_node_features
    outputs = {}
    for split in SPLITS:
        loader = data_loader.get_split_loader(split, batch_size=batch_size, device=device)
        t0 = time.time()
        predicted, is_nan_results, errors, _ = model_wrapper.predict(loader, mode=split)
        if split == 'test':
            model_wrapper.inference_time = time.time() - t0
        outputs[split] = {
            'actual': data_loader.split_scaled[split].reshape(-1, N, F),
            'predicted': np.asarray(predicted).reshape(-1, N, F),
            'error': np.asarray(errors).reshape(-1, N, F),
            'is_nan': is_nan_results if has_is_nan_mask(is_nan_results) else None,
        }
        assert outputs[split]['error'].shape[0] == len(data_loader.split_index[split]), split
    return outputs


def _all_timestamps(data_loader):
    return data_loader.split_index['train'].append([data_loader.split_index['valid'], data_loader.split_index['test']])


def save_predictions(outputs, data_loader, model_dir):
    """Parquet files [T, N·F] over the concatenated train / valid / test timestamps."""
    index = pd.Index(_all_timestamps(data_loader), name='timestamp')
    for kind, file_name in PREDICTION_FILES.items():
        values = np.concatenate([outputs[s][kind].reshape(len(outputs[s][kind]), -1) for s in SPLITS])
        if kind != 'actual':  # the actual data is kept exact; it compresses well anyway
            values = np.round(values, PREDICTION_DECIMALS)
        values = values.astype(np.float32)
        pd.DataFrame(values, index=index, columns=data_loader.feature_columns).to_parquet(
            os.path.join(model_dir, file_name), **PARQUET_OPTIONS)
    if outputs['test']['is_nan'] is not None:
        node_columns = [str(n) for n in data_loader.meta_data['node_ids']]
        for row, kind in enumerate(('predicted', 'actual')):
            values = np.concatenate([outputs[s]['is_nan'][row] for s in SPLITS]).astype(np.float32)
            pd.DataFrame(values, index=index, columns=node_columns).to_parquet(
                os.path.join(model_dir, IS_NAN_FILES[kind]), **PARQUET_OPTIONS)
    log.info(f'Predictions of all splits saved as parquet in {model_dir}')


def predictions_exist(model_dir):
    return all(os.path.exists(os.path.join(model_dir, f)) for f in PREDICTION_FILES.values())


def load_predictions(data_loader, model_dir):
    """Inverse of save_predictions, split back into train / valid / test."""
    N, F = data_loader.num_nodes, data_loader.num_node_features
    frames = {kind: pd.read_parquet(os.path.join(model_dir, f)) for kind, f in PREDICTION_FILES.items()}
    is_nan_frames = {kind: pd.read_parquet(os.path.join(model_dir, f)) for kind, f in IS_NAN_FILES.items()
                     if os.path.exists(os.path.join(model_dir, f))}
    assert frames['error'].index.equals(pd.Index(_all_timestamps(data_loader))), \
        f'Saved predictions in {model_dir} do not match the current train / valid / test split'
    outputs = {}
    for split in SPLITS:
        idx = data_loader.split_index[split]
        outputs[split] = {kind: frames[kind].loc[idx].values.reshape(-1, N, F) for kind in PREDICTION_FILES}
        outputs[split]['is_nan'] = (np.array([is_nan_frames['predicted'].loc[idx].values, is_nan_frames['actual'].loc[idx].values])
                                    if len(is_nan_frames) == 2 else None)
    log.info(f'Predictions of all splits loaded from {model_dir}')
    return outputs


# ── Anomaly scores ────────────────────────────────────────────────────────────

def _clean_mask(data_loader, split):
    """Boolean mask of the train / valid windows without anomalies; None for the test split."""
    return {'train': data_loader.train_clean_mask, 'valid': data_loader.valid_clean_mask}.get(split)


def _clean_window_mask(data_loader, split):
    """1 / 0 for the train and valid windows without / with anomalies, NaN on the test split."""
    mask = _clean_mask(data_loader, split)
    return np.nan if mask is None else mask.astype(float)


def _clean_split(outputs, data_loader, split, key):
    """Values (errors or is_nan results) of a split restricted to its windows without anomalies;
    the test split is returned whole, since cleaning it would use its labels."""
    value = outputs[split][key]
    mask = _clean_mask(data_loader, split)
    if mask is None:
        return value
    if value is None:
        return None
    return value[:, mask] if key == 'is_nan' else value[mask]


MAHALANOBIS_REFERENCES = {'train': ('train',), 'valid': ('valid',), 'train_valid': ('train', 'valid'), 'test': ('test',)}


def _reference_values(outputs, data_loader, reference, key):
    """Values of the reference split(s) restricted to their windows without anomalies, concatenated
    in time (is_nan results [2, L, N] along their second axis); None when a split has none."""
    values = [_clean_split(outputs, data_loader, split, key) for split in MAHALANOBIS_REFERENCES[reference]]
    if any(v is None for v in values):
        return None
    return np.concatenate(values, axis=1 if key == 'is_nan' else 0)


def _mahalanobis_scores(outputs, data_loader, reference, scorer_args, top_k):
    """
    Distances of every split, fitted on the reference errors (the windows without anomalies
    of 'train' / 'valid' / both for 'train_valid', all windows of 'test'), and the top-k
    contributing dimensions on the test split (after the is_nan mask).
    """
    ref_errors, ref_is_nan = _reference_values(outputs, data_loader, reference, 'error'), _reference_values(outputs, data_loader, reference, 'is_nan')
    scorer = MahalanobisScorer(**scorer_args).fit(ref_errors)
    # Contributions on the test split, with predicted-missing positions masked out: without
    # is_nan masks they come from the same pass as the test distances
    masked = has_is_nan_mask(outputs['test']['is_nan']) or has_is_nan_mask(ref_is_nan)
    distances = {}
    for split in SPLITS:
        with_contributions = split == 'test' and not masked
        distances[split], contributions = scorer.score(outputs[split]['error'], top_k=top_k if with_contributions else None)
        if with_contributions:
            test_contributions = contributions
    # The reference timestamps are (clean) timestamps of the splits: no need to score them again
    reference_distances = np.concatenate([
        distances[split] if _clean_mask(data_loader, split) is None else distances[split][_clean_mask(data_loader, split)]
        for split in MAHALANOBIS_REFERENCES[reference]])

    if masked:
        masked_scorer = MahalanobisScorer(**scorer_args).fit(apply_is_nan_mask(ref_errors, ref_is_nan))
        _, test_contributions = masked_scorer.score(apply_is_nan_mask(outputs['test']['error'], outputs['test']['is_nan']), top_k=top_k)
    return distances, reference_distances, test_contributions


def _likelihood_series(scores, long_window, short_window, chunk_size=8192):
    """
    Anomaly likelihood of every timestamp of a split, from its past scores only: vectorised
    compute_anomaly_likelihood(scores[:i + 1], long_window, short_window) for every i, i.e.
    0.5 for the first long_window timestamps, then 0.5 + 0.5 * Φ((mean of the last short_window
    scores − mean of the last long_window) / (std of the last long_window + ε)).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not 0 < short_window <= long_window:
        # Unusual windows (the short one reaches past the long one): reference implementation
        return np.array([compute_anomaly_likelihood(scores[:i + 1], long_window, short_window) for i in range(len(scores))])
    likelihood = np.full(len(scores), 0.5)
    if len(scores) <= long_window:
        return likelihood
    # Row j holds scores[j:j + long_window], the long window ending at timestamp j + long_window - 1;
    # timestamps long_window … T-1 use rows 1 … T-long_window. Chunked to bound the temporaries.
    windows = sliding_window_view(scores, long_window)[1:]
    for start in range(0, len(windows), chunk_size):
        w = windows[start:start + chunk_size]
        z = (w[:, -short_window:].mean(axis=1) - w.mean(axis=1)) / (w.std(axis=1) + 1e-10)
        likelihood[long_window + start:long_window + start + len(w)] = 0.5 + 0.5 * (1 - norm.sf(z))
    return likelihood


def _segment_likelihoods(series, window_pairs):
    """Anomaly likelihood of the scores {split: [T_split]} of every split, computed on each segment
    of LIKELIHOOD_SEGMENTS as one series (valid continues the history of train).
    Returns {split: {(lw, sw): likelihood}}."""
    likelihoods = {s: {} for s in SPLITS}
    for segment in LIKELIHOOD_SEGMENTS:
        joined = np.concatenate([series[s] for s in segment])
        bounds = np.cumsum([len(series[s]) for s in segment])[:-1]
        for lw, sw in window_pairs:
            for s, part in zip(segment, np.split(_likelihood_series(joined, lw, sw), bounds)):
                likelihoods[s][(lw, sw)] = part
    return likelihoods


def compute_mahalanobis_scores(outputs, data_loader, experiment_config, scorer_args):
    """Mahalanobis distances of every split, of the reference timestamps and test contributions: they do not
    depend on the score normalisation, so they can be computed once for all of them."""
    return _mahalanobis_scores(outputs, data_loader, experiment_config.get('mahalanobis_reference', 'train'),
                               scorer_args, experiment_config.top_k_contribution)


def compute_anomaly_scores(outputs, data_loader, experiment_config, scorer_args, normalization=None, mahalanobis=None):
    """
    Score DataFrame (one row per timestamp of every split) and the test-split top-k
    contributions {'likelihood': {f'top{k}_{i}': ...}, 'mahalanobis': [T_test, k]}.
    normalization: 'per_node' or 'global' (default: the first of the config's score normalisations);
    mahalanobis: result of compute_mahalanobis_scores, computed here when not given.
    """
    normalization = normalization or score_normalizations(experiment_config)[0]
    topks = [parse_topk(t) for t in experiment_config.topks]
    window_pairs, with_windows = likelihood_window_pairs(experiment_config)
    reference_errors = _clean_split(outputs, data_loader, 'train', 'error')

    frames = []
    for split in SPLITS:
        frames.append(pd.DataFrame({
            'timestamp': data_loader.split_index[split],
            'split': split,
            'is_anomaly': data_loader.split_labels[split],
            'is_clean_window': _clean_window_mask(data_loader, split),
            MEAN_ERROR_COLUMN: outputs[split]['error'].reshape(len(outputs[split]['error']), -1).mean(axis=1),
        }))
    scores = pd.concat(frames, ignore_index=True)

    # Per-node error normalisation (statistics of the reference errors), once per split for every top-k
    z = {s: get_full_err_scores(outputs[s]['error'], normalization, reference_errors=reference_errors)
            .reshape(len(outputs[s]['error']), -1) for s in SPLITS}
    # Score columns are collected and joined at once: adding them one by one fragments the DataFrame
    score_columns = {}
    likelihood_contributions = {}
    for topk in topks:
        per_split = _segment_likelihoods({s: aggregate_sensor_scores(z[s], topk) for s in SPLITS}, window_pairs)
        for lw, sw in window_pairs:
            score_columns[likelihood_column(topk, lw, sw, with_windows)] = np.concatenate([per_split[s][(lw, sw)] for s in SPLITS])
        if topk is not None:   # contributing nodes, on the test split only
            test_top_nodes = top_k_indices(z['test'], topk)
            for i in range(test_top_nodes.shape[1]):
                likelihood_contributions[f'top{topk}_{i}'] = test_top_nodes[:, i]

    distances, reference_distances, mahalanobis_contributions = (
        mahalanobis or compute_mahalanobis_scores(outputs, data_loader, experiment_config, scorer_args))

    # Distances min-max scaled with the min / max of the reference distances: the reference lies
    # in [0, 1] and other splits may exceed it (> 1 = farther than any reference timestamp)
    d_min, d_max = reference_distances.min(), reference_distances.max()
    scaled = {s: (distances[s] - d_min) / max(d_max - d_min, 1e-12) for s in SPLITS}
    score_columns[MAHALANOBIS_COLUMN] = np.concatenate([scaled[s] for s in SPLITS])

    # Likelihood of the scaled distances, computed on each segment of LIKELIHOOD_SEGMENTS
    scaled_likelihoods = _segment_likelihoods(scaled, window_pairs)
    for lw, sw in window_pairs:
        score_columns[mahalanobis_likelihood_column(lw, sw, with_windows)] = np.concatenate(
            [scaled_likelihoods[s][(lw, sw)] for s in SPLITS])
    scores = pd.concat([scores, pd.DataFrame(score_columns, index=scores.index)], axis=1)
    return scores, {'likelihood': likelihood_contributions, 'mahalanobis': mahalanobis_contributions}


def save_anomaly_scores(scores, model_dir, model, experiment_config, scorer_args, predictions_recomputed,
                        normalization=None):
    """
    Write the score CSV of one score normalisation, updating an existing one: columns computed now
    replace theirs, other columns are kept. A fresh file is written when the predictions were recomputed.
    """
    normalization = normalization or score_normalizations(experiment_config)[0]
    path = os.path.join(model_dir, scoring_result_file_name(model, 'anomaly_scores', normalization))
    if os.path.exists(path) and not predictions_recomputed:
        existing = pd.read_csv(path, parse_dates=['timestamp'])
        same_rows = (len(existing) == len(scores)
                     and existing['timestamp'].equals(scores['timestamp'])
                     and existing['split'].equals(scores['split']))
        if same_rows:
            kept = existing.columns.difference(scores.columns).drop(list(LEGACY_SCORE_COLUMNS), errors='ignore')
            scores = pd.concat([scores, existing[kept].set_axis(scores.index)], axis=1)
        else:
            log.warning(f'{path} does not match the current splits and is rewritten')
    # Fixed column order: base columns, mean error, likelihood columns, Mahalanobis
    likelihood_columns = sorted(c for c in scores.columns if c.startswith('anomaly_score_postprocessed_by_likelihood'))
    other_columns = [c for c in scores.columns if c not in likelihood_columns and c != MAHALANOBIS_COLUMN]
    scores = scores[other_columns + likelihood_columns + [MAHALANOBIS_COLUMN]]
    scores.to_csv(path, index=False)

    settings = {
        'score_normalization': normalization,
        'normalization_statistics': 'train windows without anomalies',
        'topks': [parse_topk(t) for t in experiment_config.topks],
        'likelihood_windows': likelihood_window_pairs(experiment_config)[0],
        'mahalanobis_reference': experiment_config.get('mahalanobis_reference', 'train'),
        'mahalanobis_scorer': MahalanobisScorer(**scorer_args).settings(),
    }
    with open(path.replace('.csv', '_settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    log.info(f'Anomaly scores saved to {path}')
    return path


def load_anomaly_scores(model_dir, model, score_normalization):
    path = os.path.join(model_dir, scoring_result_file_name(model, 'anomaly_scores', score_normalization))
    if not os.path.exists(path):
        return None
    scores = pd.read_csv(path, parse_dates=['timestamp'])
    renamed = {old: new for old, new in LEGACY_SCORE_COLUMNS.items() if new and old in scores and new not in scores}
    return scores.rename(columns=renamed)


# How the likelihood strategies are thresholded: 'absolute' = likelihood > anomaly_threshold (a value
# in [0.5, 1], past scores only); 'percentile' = above its anomaly_threshold-th percentile on the
# split (uses the whole split, as the mahalanobis / mean_reconstruction_errors strategies)
LIKELIHOOD_THRESHOLD_MODES = ('absolute', 'percentile')
LIKELIHOOD_STRATEGIES = ('likelihood', 'likelihood_mahalanobis')


def threshold_type(strategy, likelihood_threshold_mode):
    """'absolute' or 'percentile': how anomaly_threshold is applied for this strategy."""
    return likelihood_threshold_mode if strategy in LIKELIHOOD_STRATEGIES else 'percentile'


def label_from_scores(test_scores, strategy, topk, anomaly_threshold, long_window, short_window, with_windows,
                      likelihood_threshold_mode='absolute'):
    """
    Binary predictions on the test split from the saved scores:
      • likelihood            : likelihood > anomaly_threshold, or above its anomaly_threshold-th
                                percentile on the split with likelihood_threshold_mode='percentile'
      • likelihood_mahalanobis: same, on the likelihood of the scaled Mahalanobis distance
      • mahalanobis           : distance above its anomaly_threshold-th percentile on the test split
      • mean_reconstruction_errors: mean error over all sensors above its anomaly_threshold-th
                                    percentile on the test split
    Returns (is_anomalies, score series).
    """
    assert likelihood_threshold_mode in LIKELIHOOD_THRESHOLD_MODES, \
        f'Unknown likelihood_threshold_mode: {likelihood_threshold_mode}'
    if strategy in LIKELIHOOD_STRATEGIES:
        if strategy == 'likelihood':
            score = test_scores[likelihood_column(topk, long_window, short_window, with_windows)].values
        else:
            score = test_scores[mahalanobis_likelihood_column(long_window, short_window, with_windows)].values
        if likelihood_threshold_mode == 'percentile':
            return (score > np.percentile(score, anomaly_threshold)).astype(int), score
        return (score > anomaly_threshold).astype(int), score
    if strategy == 'mahalanobis':
        score = test_scores[MAHALANOBIS_COLUMN].values
        return (score > np.percentile(score, anomaly_threshold)).astype(int), score
    if strategy == 'mean_reconstruction_errors':
        score = test_scores[MEAN_ERROR_COLUMN].values
        return (score > np.percentile(score, anomaly_threshold)).astype(int), score
    raise ValueError(f'Unsupported post-processing strategy: {strategy}')
