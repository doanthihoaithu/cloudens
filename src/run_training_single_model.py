import random
import sys
import os
import time

import hydra
from omegaconf import DictConfig
import logging
import itertools
from tqdm import tqdm

import math

import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import mahalanobis
from numpy.linalg import pinv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

from ibm_dataset_loader import IBMDatasetLoader
from metrics.ffvus.ffvus_metrics import FFVUS
from model_wrappers.A3TGCNWrapper import A3TGCNWrapper
from model_wrappers.GRUWrapper import GRUWrapper
from model_wrappers.GDNWrapper import GDNWrapper
from model_wrappers.TranADWrapper import TranADWrapper
from model_wrappers.OmniAnomalyWrapper import OmniAnomalyWrapper
from model_wrappers.USADWrapper import USADWrapper
from model_wrappers.MTADGATWrapper import MTADGATWrapper
from model_wrappers.GSTPROWrapper import GSTPROWrapper
from model_wrappers.AnomalyTransformerWrapper import AnomalyTransformerWrapper
from model_wrappers.STformerWrapper import STformerWrapper
from model_wrappers.STGformerWrapper import STGformerWrapper
from plotting_module import plot_training_history, plot_reconstruction_and_mahalanobis
from anomaly_likelihood import compute_anomaly_likelihood
from model_outputs import (predict_all_splits, save_predictions, predictions_exist, load_predictions,
                           compute_anomaly_scores, compute_mahalanobis_scores, save_anomaly_scores, label_from_scores,
                           likelihood_window_pairs, parse_topk, MAHALANOBIS_COLUMN, MAHALANOBIS_REFERENCES,
                           LIKELIHOOD_THRESHOLD_MODES, LIKELIHOOD_STRATEGIES, threshold_type)
from nab_scoring import calculate_nab_score_with_window_based_tp_fn, merge_overlapping_windows
from utils import clear_folder, get_project_root, get_full_err_scores, set_random_seed, calculate_mahalanobis_distance, has_is_nan_mask, MahalanobisScorer, scoring_result_file_name, results_root_dir, score_normalizations, \
    calculate_mahalanobis_distance_with_is_nan_mask, refine_reconstruction_error_with_is_nan_mask

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_wrapper(model_name, config, static_edge_index, static_edge_weight):
    node_features = config['node_features']
    periods = config['slide_win']
    batch_size = config['batch_size']
    device = config['device']
    hidden_units = config['hidden_units']
    num_nodes = config['num_nodes']
    null_padding_target = config['null_padding_target']
    null_padding_feature = config['null_padding_feature']

    if model_name == 'A3TGCN':
        return A3TGCNWrapper(node_features, null_padding_feature, null_padding_target, periods, static_edge_index, static_edge_weight, batch_size=batch_size, device=device)
    elif model_name == 'GRU':
        return GRUWrapper(num_nodes, node_features, hidden_units, layer_dim=config['layer_dim'],
                          dropout=config['dropout'], lr=config['learning_rate'], batch_size=batch_size, device=device)
    elif model_name == 'GDN':
        return GDNWrapper(num_nodes, node_features, periods,
                          embed_dim=64, hidden_dim=hidden_units * 2, topk=20,
                          null_padding_feature=null_padding_feature,
                          null_padding_target=null_padding_target,
                          batch_size=batch_size, device=device)
    elif model_name == 'TranAD':
        return TranADWrapper(num_nodes, node_features, periods,
                             d_model=64, nhead=4, n_layers=1,
                             batch_size=batch_size, device=device)
    elif model_name == 'OmniAnomaly':
        return OmniAnomalyWrapper(num_nodes, node_features, periods,
                                  hidden_dim=hidden_units * 2, latent_dim=16,
                                  n_layers=2, beta=0.1,
                                  batch_size=batch_size, device=device)
    elif model_name == 'USAD':
        return USADWrapper(num_nodes, node_features, periods,
                           hidden_dim=hidden_units * 2, latent_dim=32,
                           n_layers=1, alpha=0.5,
                           batch_size=batch_size, device=device)
    elif model_name == 'MTAD_GAT':
        return MTADGATWrapper(num_nodes, node_features, periods,
                              d_model=32, nhead=2, gru_hidden=hidden_units * 2,
                              batch_size=batch_size, device=device)
    elif model_name == 'GST_PRO':
        return GSTPROWrapper(num_nodes, node_features, periods,
                             embed_dim=32, hidden_dim=hidden_units * 2, K=2,
                             null_padding_feature=null_padding_feature,
                             null_padding_target=null_padding_target,
                             batch_size=batch_size, device=device)
    elif model_name == 'AnomalyTransformer':
        return AnomalyTransformerWrapper(num_nodes, node_features, periods,
                                         d_model=hidden_units * 2, nhead=4, n_layers=3,
                                         lambda_=0.1,
                                         score_mode=config.get('score_mode', 'forecast'),
                                         temperature=config.get('score_temperature', 1.0),
                                         batch_size=batch_size, device=device)
    elif model_name == 'STformer':
        return STformerWrapper(num_nodes, node_features, periods,
                               d_model=hidden_units * 2, nhead=4, n_layers=2,
                               batch_size=batch_size, device=device)
    elif model_name == 'STGformer':
        return STGformerWrapper(num_nodes, node_features, periods,
                                input_embedding_dim=24, adaptive_embedding_dim=hidden_units + 8,
                                num_heads=4, num_layers=3, order=2,
                                temporal_kernel_size=config.get('temporal_kernel_size', 1),
                                null_padding_feature=null_padding_feature,
                                null_padding_target=null_padding_target,
                                batch_size=batch_size, device=device)
    # if model_name == 'ASTGCN':
    #     return ASTGCNWrapper(num_nodes, node_features, periods, static_edge_index, batch_size=batch_size, device=device)
    # if model_name == 'MTGNN':
    #     return MTGNNWrapper(num_nodes, node_features, periods, static_edge_index, batch_size=batch_size, device=device)
    # elif model_name == 'TGCN':
    #     return TGCNWrapper(num_nodes, node_features, hidden_units, static_edge_index, static_edge_weights=None, device=device)
    else:
        raise Exception("Model type not supported")

@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to coordinate anomaly detection using autoencoders.

    Steps:
    1. Load and preprocess the data.
    2. Extract training and testing windows.
    3. Prepare ground truth anomaly windows for evaluation.
    4. Perform grid search to optimize anomaly detection parameters.

    Parameters:
    - cfg: DictConfig, configuration object containing paths, parameters, and settings.
    """
    log.info("Starting main function")

    random_seed = cfg.modeling.random_seed
    set_random_seed(random_seed)

    experiment_config = cfg.evaluation
    model_configs = cfg.model_configs

    # Extract experiment parameters
    # start_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.start_date)
    # train_end_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.train_end_date)
    # test_start_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.test_start_date)
    # end_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.end_date) + timedelta(days=1)

    # data_loader_config = dict({
    #     "pivoted_raw_data_dir": os.path.join(data_dir, 'massaged'),
    #     'anomaly_windows_dir': os.path.join(data_dir, 'labels'),
    #     'start_date': start_date,  # Actual Satrt 26 Jan 2024
    #     'train_end_date': train_end_date,
    #     'test_start_date': test_start_date,
    #     'minutes_before': cfg.train_test_config.anomaly_window.minutes_before,
    #     #    end_date: '2024-03-02'
    #     'end_date': end_date,
    #     'train_test_config': cfg.train_test_config,
    # })

    data_preparation_config = cfg.data_preparation_pipeline
    # Null padding only applies to graph-based models
    if experiment_config.use_model not in cfg.graph_models:
        data_preparation_config.null_padding_feature = False
        data_preparation_config.null_padding_target = False

        experiment_config.null_padding_feature = False
        experiment_config.null_padding_target = False


    # task_id = sys.argv[0]
    # print(f"Task ID: {task_id}")
    # model = experiment_config.use_models[task_id]
    # print(f'Using model: {model}')

    ibm_dataset_loader = IBMDatasetLoader(data_preparation_config)

    selected_group_mode = ibm_dataset_loader.selected_group_mode

    analyze_reconstruction_errors(ibm_dataset_loader, selected_group_mode, model_configs=model_configs, experiment_config=experiment_config, random_seed=random_seed)

def _compute_pr_metrics(reconstruction_error_raw: np.ndarray,
                        test_labels,
                        sliding_window: int = 64) -> dict:
    """
    Compute AUC-PR and VUS-PR (FFVUS) from raw per-timestep reconstruction errors.

    Parameters
    ----------
    reconstruction_error_raw : np.ndarray, shape [total, N, F]
        Absolute reconstruction error from model.predict().
    test_labels : array-like, shape [total]
        Binary ground-truth anomaly labels (0 = normal, 1 = anomaly).
    sliding_window : int
        Temporal tolerance window for VUS (the FFVUS "slope" parameter).
        VUS averages AUC-PR over window sizes 0 … sliding_window.

    Returns
    -------
    dict with keys 'AUC_PR' and 'VUS_PR' (float, NaN on failure).
    """
    # Aggregate to a 1-D anomaly score: mean over N and F
    scores = reconstruction_error_raw.mean(axis=-1).mean(axis=-1)   # [total]
    return _pr_metrics_from_scores(scores, test_labels, sliding_window)


# Keys of _pr_metrics_from_scores
PR_METRIC_KEYS = ('AUC_PR', 'VUS_PR', 'BEST_F1', 'F1_OPTIMAL_THRESHOLD')


def _f1_thresholds(scores, labels):
    """
    F1 of every threshold of a 1-D anomaly score (one per distinct score value cutting it, from the PR
    curve) and that threshold, in the units of the score. Each threshold lies halfway between the lowest
    score predicted anomalous and the next lower score, so that score > threshold (as label_from_scores)
    and score >= threshold give the same F1. Returns (f1, thresholds, detected), thresholds in increasing
    order, detected the number of timestamps above each threshold.
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, lowest_detected = precision_recall_curve(labels, scores)   # score >= lowest_detected[i]
    precision, recall = precision[:-1], recall[:-1]   # last point: recall 0, no threshold
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    # Next lower score of every cut: the previous distinct score, below the first one any lower score
    lower = scores[scores < lowest_detected[0]]
    first_lower = lower.max() if len(lower) else np.nextafter(lowest_detected[0], -np.inf)
    next_lower = np.concatenate([[first_lower], lowest_detected[:-1]])
    thresholds = (lowest_detected + next_lower) / 2
    rounded = ~((next_lower <= thresholds) & (thresholds < lowest_detected))   # midpoint on lowest_detected (adjacent floats)
    thresholds[rounded] = next_lower[rounded]
    detected = len(scores) - np.searchsorted(np.sort(scores), lowest_detected, side='left')
    return f1, thresholds, detected


def _best_f1_threshold(scores, labels):
    """Best F1 over every threshold of a 1-D anomaly score and the threshold reaching it (see _f1_thresholds)."""
    f1, thresholds, _ = _f1_thresholds(scores, labels)
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])


def _top_f1_thresholds(scores, labels, k, min_gap=0.0):
    """
    The k thresholds of a 1-D anomaly score with the highest F1, best first (see _f1_thresholds), at least
    min_gap apart: the numbers of timestamps they flag differ by at least min_gap (a share of the larger
    one) from those of every threshold kept before, so that they do not all flag nearly the same timestamps.
    """
    f1, thresholds, detected = _f1_thresholds(scores, labels)
    kept = []
    for i in np.argsort(-f1, kind='stable'):
        if all(abs(detected[i] - detected[j]) >= min_gap * max(detected[i], detected[j]) for j in kept):
            kept.append(i)
            if len(kept) == k:
                break
    return [float(thresholds[i]) for i in kept]


def _pr_metrics_from_scores(scores, labels, sliding_window: int = 64) -> dict:
    """AUC-PR and VUS-PR (FFVUS, slope = sliding_window) of a 1-D anomaly score [total], and its
    best F1 over all thresholds with the threshold reaching it (in the units of the score);
    {'AUC_PR', 'VUS_PR', 'BEST_F1', 'F1_OPTIMAL_THRESHOLD'}, NaN when undefined (no positive label) or on failure."""
    from sklearn.metrics import average_precision_score
    scores = np.asarray(scores, dtype=np.float64)

    # Min-max normalise to [0, 1] (required by VUS internals)
    s_min, s_max = scores.min(), scores.max()
    scores_norm = (scores - s_min) / (s_max - s_min + 1e-12)

    # Align ground-truth labels; drop NaN positions
    labels = np.array(labels).ravel().astype(float)
    valid = ~np.isnan(labels)
    labels = labels[valid].astype(int)
    scores_norm = scores_norm[valid]

    results = dict.fromkeys(PR_METRIC_KEYS, float('nan'))

    if labels.sum() == 0:
        log.warning('No positive labels found — AUC_PR, VUS_PR and the best F1 are undefined.')
        return results

    # Best F1 and its threshold, on the score itself (not min-max normalised)
    try:
        results['BEST_F1'], results['F1_OPTIMAL_THRESHOLD'] = _best_f1_threshold(scores[valid], labels)
    except Exception as e:
        log.warning(f'Best F1 computation failed: {e}')

    # AUC-PR
    try:
        results['AUC_PR'] = float(average_precision_score(labels, scores_norm))
    except Exception as e:
        log.warning(f'AUC-PR computation failed: {e}')

    # VUS-PR  (FFVUS, slope = sliding_window)
    try:
        log.info(f'Computing VUS-PR with sliding_window={sliding_window} — this may take a moment …')
        vus_pr = FFVUS(slope=sliding_window).score(labels, scores_norm)['value']
        results['VUS_PR'] = float(vus_pr)
    except Exception as e:
        log.warning(f'VUS-PR computation failed: {e}')

    return results


def analyze_reconstruction_errors(data_loader, selected_group_mode, model_configs, experiment_config, random_seed):

    mode = experiment_config.get('mode', 'single')
    if mode == 'single':
        models = [experiment_config.get('use_model', 'A3TGCN')]
    else:
        models = experiment_config.get('use_models', ['A3TGCN'])

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = experiment_config.train_batch_size

    train_loader, valid_loader, test_loader, edge_index = data_loader.get_index_dataset(
        window_size=experiment_config.slide_win,
        null_padding_feature=experiment_config.null_padding_feature,
        null_padding_target=experiment_config.null_padding_target,
        batch_size=batch_size,
        device=DEVICE)
    print("Training dataset batches", len(train_loader))
    print("Validation dataset batches", len(valid_loader))
    print("Testing dataset batches", len(test_loader))

    # Set on which the Mahalanobis mean/covariance are fitted: 'train', 'valid', 'train_valid' or 'test'
    mahalanobis_reference = experiment_config.get('mahalanobis_reference', 'train')
    assert mahalanobis_reference in MAHALANOBIS_REFERENCES, \
        f'Unknown mahalanobis_reference: {mahalanobis_reference}'
    # Covariance estimator of the Mahalanobis score (see utils.MahalanobisScorer)
    mahalanobis_scorer_args = dict(
        covariance=experiment_config.get('mahalanobis_covariance', 'empirical'),
        min_variance=experiment_config.get('mahalanobis_min_variance', 0.0),
        pca_variance=experiment_config.get('mahalanobis_pca_variance', 0.95),
        pca_quantile=experiment_config.get('mahalanobis_pca_quantile', 0.99),
    )

    # Train the autoencoder based on model type
    # Define the path to the trained models directory
    project_root_dir = get_project_root()
    trained_models_dir = os.path.join(project_root_dir,
                                      results_root_dir(experiment_config.model_save_path, data_loader.log_transform))
    os.makedirs(trained_models_dir, exist_ok=True)  # Ensure the directory exists

    # Define the model filename based on the model type
    # model_filename = f"{trained_models_dir}{model_type}_autoencoder.h5"

    slide_win = experiment_config.slide_win

    fill_nan = experiment_config.fill_nan

    for model in models:
        print("Running model", model)
        model_config = model_configs[model]
        # model_filename_extension = 'h5' if model_type in tf_models else 'pt'

        # model_dir = os.path.join(trained_models_dir, selected_group_mode, model)
        if (experiment_config.null_padding_feature == False) and (experiment_config.null_padding_target == False):
            model_dir = os.path.join(trained_models_dir, f'window_{slide_win}',
                                     selected_group_mode,
                                     f'fill_nan_with_{fill_nan}',
                                     model)
        elif experiment_config.null_padding_target and (not experiment_config.null_padding_feature):
            model_dir = os.path.join(trained_models_dir, f'window_{slide_win}',
                                     selected_group_mode,
                                     f'fill_nan_with_{fill_nan}',
                                     f'{model}_null_padding_target')
        elif (not experiment_config.null_padding_target) and experiment_config.null_padding_feature:
            model_dir = os.path.join(trained_models_dir, f'window_{slide_win}',
                                     selected_group_mode,
                                     f'fill_nan_with_{fill_nan}',
                                     f'{model}_null_padding_feature')
        else:
            model_dir = os.path.join(trained_models_dir, f'window_{slide_win}',
                                     selected_group_mode,
                                     f'fill_nan_with_{fill_nan}',
                                     f'{model}_null_padding_both')

        # Models with a configurable anomaly score keep each mode's results apart,
        # e.g. AnomalyTransformer_forecast / AnomalyTransformer_association
        score_mode = model_config.get('score_mode', None)
        if score_mode is not None:
            model_dir = f'{model_dir}_{score_mode}'
        # STGformer with its graphs pooled over time (temporal_kernel_size > 1): results kept apart,
        # e.g. STGformer_pool6, since the trained weights differ in shape
        temporal_kernel_size = model_config.get('temporal_kernel_size', 1)
        if temporal_kernel_size > 1:
            model_dir = f'{model_dir}_pool{temporal_kernel_size}'

        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, model_config['model_filename'])
        # Predictions are recomputed when asked or when this model is (re)trained; kept local so that
        # training one model does not force the re-prediction of the next ones (shared experiment_config)
        retest = experiment_config.retest
        # Check if the model file exists
        if os.path.exists(model_filename) and experiment_config.retrain == False:
            print(f"Loading trained model: {model_filename}")
            graph_config = dict({"node_features": data_loader.get_num_node_features(),
                                 'slide_win': experiment_config.slide_win,
                                 'batch_size': batch_size,
                                 'hidden_units': model_config.get('hidden_units', 32),
                                 'layer_dim': model_config.get('layer_dim', 1),
                                 'dropout': model_config.get('dropout', 0.3),
                                 'learning_rate': model_config.get('learning_rate', 0.001),
                                 'num_nodes': data_loader.num_nodes,
                                 'null_padding_feature': experiment_config.null_padding_feature,
                                 'null_padding_target': experiment_config.null_padding_target,
                                 'score_mode': score_mode or 'forecast',
                                 'score_temperature': model_config.get('score_temperature', 1.0),
                                 'temporal_kernel_size': temporal_kernel_size,
                                 'device': DEVICE})
            model_wrapper = load_wrapper(model_name=model, config=graph_config,
                                         static_edge_index=data_loader.get_edges_as_tensor(device=DEVICE),
                                         static_edge_weight=data_loader.get_edge_weights_as_tensor(device=DEVICE))
            model_wrapper.load(model_filename)
        else:
            if experiment_config.retrain:
                print(f"Using model: {model}, null padding: {experiment_config.null_padding_target}. Re-training...")
            else:
                print(f"No trained model found for {model}, null padding {experiment_config.null_padding_target}. Training a new model...")

            clear_folder(model_dir)

            # Re-seed right before constructing the model so every model's initial
            # weights are deterministic from modeling.random_seed, independent of
            # how much RNG state prior data loading/training in this process consumed.
            set_random_seed(random_seed)

            graph_config = dict({"node_features": data_loader.get_num_node_features(),
                                 'slide_win': experiment_config.slide_win,
                                 'batch_size': batch_size,
                                 'hidden_units': model_config.get('hidden_units', 32),
                                 'layer_dim': model_config.get('layer_dim', 1),
                                 'dropout': model_config.get('dropout', 0.3),
                                 'learning_rate': model_config.get('learning_rate', 0.001),
                                 'num_nodes': data_loader.num_nodes,
                                 'null_padding_target': experiment_config.null_padding_target,
                                 'null_padding_feature': experiment_config.null_padding_feature,
                                 'score_mode': score_mode or 'forecast',
                                 'score_temperature': model_config.get('score_temperature', 1.0),
                                 'temporal_kernel_size': temporal_kernel_size,
                                 'device': DEVICE})
            model_wrapper = load_wrapper(model_name=model, config=graph_config,
                                         static_edge_index=data_loader.get_edges_as_tensor(device=DEVICE),
                                         static_edge_weight=data_loader.get_edge_weights_as_tensor(device=DEVICE))
            history = model_wrapper.train(train_loader, valid_loader, epochs=experiment_config.get('epochs'))
            model_wrapper.save(model_filename)
            plot_training_history(model_name=model, training_history=history,
                                  model_save_dir=os.path.dirname(model_filename))

            retest = True

        # Predict every window of the three splits (train windows with anomalies included)
        predictions_recomputed = not predictions_exist(model_dir) or retest
        if predictions_recomputed:
            outputs = predict_all_splits(model_wrapper, data_loader, batch_size, DEVICE)
            save_predictions(outputs, data_loader, model_dir)
            pd.DataFrame(data={'inference_time': [model_wrapper.inference_time]}).to_csv(os.path.join(model_dir, 'inference_time.csv'))
        else:
            outputs = load_predictions(data_loader, model_dir)
        test_errors = outputs['test']['error']

        mse_reconstruction_error_file = os.path.join(model_dir, 'mse_error.txt')
        pr_metrics = _compute_pr_metrics(test_errors, data_loader.test_labels)
        with open(mse_reconstruction_error_file, 'w') as mse_f:
            mse_f.write(f"MSE={str(test_errors.mean())}\n")
            mse_f.write(f"AUC_PR={pr_metrics['AUC_PR']:.6f}\n")
            mse_f.write(f"VUS_PR={pr_metrics['VUS_PR']:.6f}\n")
            log.info(f"MSE, AUC_PR, VUS_PR (FFVUS slope=64) saved to {mse_reconstruction_error_file}")

        # Mahalanobis distances do not depend on the score normalisation: computed once for all of them
        mahalanobis = compute_mahalanobis_scores(outputs, data_loader, experiment_config, mahalanobis_scorer_args)
        mahalanobis_top_k_contribution_df = pd.DataFrame(mahalanobis[2], index=data_loader.test_index)
        mahalanobis_top_k_contribution_df.to_csv(os.path.join(model_dir, 'mahalanobis_top_k_contribution.csv'))
        # The plot min-max scales the test distances itself
        plot_path = plot_reconstruction_and_mahalanobis(
            reconstruction_error_raw=test_errors,
            mahalanobis_distances=mahalanobis[0]['test'],
            test_index=data_loader.test_index,
            test_labels=data_loader.test_labels,
            model_dir=model_dir,
            model_name=model,
        )
        log.info(f'Reconstruction & Mahalanobis plot saved to {plot_path}')

        # Scores and grid search for every score normalisation; their files are tagged with it
        for score_normalization in score_normalizations(experiment_config):
            log.info(f'Anomaly scores and grid search with {score_normalization} score normalization')
            # Anomaly scores of every split (statistics fitted on the train windows without anomalies)
            scores, contributions = compute_anomaly_scores(outputs, data_loader, experiment_config, mahalanobis_scorer_args,
                                                           normalization=score_normalization, mahalanobis=mahalanobis)
            save_anomaly_scores(scores, model_dir, model, experiment_config, mahalanobis_scorer_args, predictions_recomputed,
                                normalization=score_normalization)
            test_scores = scores[scores['split'] == 'test'].reset_index(drop=True)

            log.info("Starting Grid Search for best parameters...")
            split_scores = {name: scores[scores['split'].isin(splits)].reset_index(drop=True)
                            for name, splits in EVALUATION_SPLITS.items()}
            result_df, is_anomalies_df = grid_search_new(data_loader, test_scores, experiment_config=experiment_config,
                                                         split_scores=split_scores, score_normalization=score_normalization)

            result_df.insert(1,'NAB_standard_rank', result_df['standard_normalized'].rank(ascending=False))
            result_df.insert(2, 'NAB_reward_fn_rank', result_df['reward_fn_normalized'].rank(ascending=False))
            assert result_df.shape[0] == is_anomalies_df.shape[1]

            grid_search_file = os.path.join(model_dir, scoring_result_file_name(model, 'grid_search', score_normalization))
            result_df.to_csv(grid_search_file, index=False)
            log.info('Grid Search results saved to {}'.format(grid_search_file))

            likelihood_top_k_contribution_file = os.path.join(
                model_dir, scoring_result_file_name(model, 'likelihood_top_k_contribution', score_normalization))
            pd.DataFrame(data=contributions['likelihood'], index=data_loader.test_index).to_csv(likelihood_top_k_contribution_file)

            max_NAB_standard_profile_index = result_df['standard_normalized'].idxmax()
            log.info(
                f"{model} models's best NAB score with standard profile ({score_normalization}): {result_df['standard_normalized'].max()}"
                f" with params {result_df.loc[max_NAB_standard_profile_index].values}")
            max_NAB_reward_fn_profile_index = result_df['reward_fn_normalized'].idxmax()
            log.info(
                f"{model} models's best NAB score with reward_fn profile ({score_normalization}): {result_df['reward_fn_normalized'].max()}"
                f" with params {result_df.loc[max_NAB_reward_fn_profile_index].values}")

            predictions_for_assembles_file = os.path.join(
                model_dir, scoring_result_file_name(model, 'predictions_for_assembles', score_normalization))
            is_anomalies_df.to_csv(predictions_for_assembles_file, index=False)
            log.info('Prediction results saved to {}'.format(predictions_for_assembles_file))

# Splits on which every grid-search setting is evaluated with the same metrics, e.g. to choose the
# optimal setting / model on train / valid without the test labels and compare with the test split
# (test_precision ... test_accuracy repeat the precision ... accuracy columns, also on the test split)
EVALUATION_SPLITS = {'train': ('train',), 'valid': ('valid',), 'train_valid': ('train', 'valid'),
                     'test': ('test',)}   # name: splits concatenated
# best_f1 / f1_optimal_threshold: best F1 of the setting's score over all thresholds on the split and the
# threshold reaching it (in the units of the score; depends on the score only, not on anomaly_threshold).
# auc_pr / vus_pr / best_f1 / f1_optimal_threshold are empty (NaN) for the likelihood strategies: a
# precision-recall curve over the thresholds of a likelihood does not apply, only for the scores
# themselves (mean reconstruction errors, Mahalanobis distance)
SPLIT_METRICS = ('precision', 'recall', 'f1', 'accuracy', 'auc_pr', 'vus_pr', 'best_f1', 'f1_optimal_threshold')
# Score strategies (mahalanobis, mean_reconstruction_errors) with score_threshold_mode 'absolute': their
# thresholds are the score_absolute_num_thresholds thresholds with the highest F1 on F1_OPTIMAL_SPLIT (no test
# label), at least score_absolute_min_gap apart (see _top_f1_thresholds), score values applied as is to every
# split; with 'percentile', the distribution_anomaly_thresholds
# percentiles of each split
F1_OPTIMAL_SPLIT = 'train_valid'
SCORE_STRATEGIES = ('mean_reconstruction_errors', 'mahalanobis')
# Splits also scored with NAB ({split}_{profile}_raw / _normalized columns), e.g. to choose the optimal
# setting on train / valid by the NAB score, as on the test split, without the test labels
NAB_SPLITS = ('train_valid',)
NAB_PROFILES = ('standard', 'reward_fn')
NAB_SPLIT_METRICS = tuple(f'{profile}_{kind}' for profile in NAB_PROFILES for kind in ('raw', 'normalized'))


def _split_anomaly_windows(data_loader, split_scores):
    """Ground-truth anomaly windows lying within the timestamps of a split, as anomaly_windows_test
    for the test split, the overlapping ones merged (e.g. three overlapping windows on 2024-02-12 in
    train_valid): otherwise each would be a TP / FN of its own while the normalisation counts one."""
    windows = data_loader.anomaly_windows
    timestamps = split_scores['timestamp']
    return merge_overlapping_windows(windows[(windows['anomaly_window_start'] >= timestamps.min()) &
                                             (windows['anomaly_window_end'] <= timestamps.max())])


def _split_metrics(split_scores, strategy, topk, anomaly_threshold, long_window, short_window, with_windows, pr_cache,
                   likelihood_threshold_mode='absolute', anomaly_windows=None, score_threshold_mode='percentile'):
    """
    Point metrics of one post-processing setting on one split (all its windows, labels included);
    percentile thresholds use the percentiles of this split's own scores, as on the test split.
    AUC-PR / VUS-PR depend on the score only, not on the threshold: cached in pr_cache.
    With anomaly_windows (the split's ground-truth windows), the NAB scores of the same predictions too.
    """
    labels = split_scores['is_anomaly'].to_numpy().astype(int)
    is_anomalies, score = label_from_scores(split_scores, strategy, topk, anomaly_threshold, long_window, short_window,
                                            with_windows, likelihood_threshold_mode, score_threshold_mode)
    precision, recall, f1, accuracy, _, _ = evaluate_performance(labels, is_anomalies)
    key = (strategy, topk, long_window, short_window)
    if key not in pr_cache:
        pr_cache[key] = (dict.fromkeys(PR_METRIC_KEYS, float('nan')) if strategy in LIKELIHOOD_STRATEGIES
                         else _pr_metrics_from_scores(score, labels))
    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
               'auc_pr': pr_cache[key]['AUC_PR'], 'vus_pr': pr_cache[key]['VUS_PR'],
               'best_f1': pr_cache[key]['BEST_F1'], 'f1_optimal_threshold': pr_cache[key]['F1_OPTIMAL_THRESHOLD']}
    if anomaly_windows is not None:
        nab_df = pd.DataFrame({'true_anomaly': labels, 'predicted_anomaly': is_anomalies},
                              index=pd.DatetimeIndex(split_scores['timestamp']))
        for profile in NAB_PROFILES:
            raw, normalized, _, _, _ = calculate_nab_score_with_window_based_tp_fn(
                nab_df, anomaly_windows, profile, true_col='true_anomaly', pred_col='predicted_anomaly')
            metrics.update({f'{profile}_raw': raw, f'{profile}_normalized': normalized})
    return metrics


def grid_search_new(data_loader, test_scores, experiment_config, split_scores=None, score_normalization=None):
    """
    NAB / point metrics of every post-processing setting, thresholding the precomputed test scores.
    With split_scores ({split: score DataFrame} of EVALUATION_SPLITS), the point metrics and
    AUC-PR / VUS-PR on those splits are added as {split}_{metric} columns, and the NAB scores
    on the NAB_SPLITS among them as {split}_{profile}_raw / _normalized columns.
    """
    split_scores = split_scores or {}
    pr_caches = {split: {} for split in split_scores}
    split_anomaly_windows = {split: _split_anomaly_windows(data_loader, split_scores[split])
                             for split in NAB_SPLITS if split in split_scores}
    # best_params_unweighted, best_unweighted_score, best_results_unweighted, result_csv_filepath
    columns = [
            'standard_normalized',
            'reward_fn_normalized',
            'detection_counters',
            'confusion_matrix',
            'post_processing_strategy',
            'anomaly_threshold',
            'topk',
            'long_window',
            'short_window',
            'precision',
            'recall',
            'f1',
            'accuracy',
            'standard_raw',
            'reward_fn_raw',
            'model',
            'fill_nan_value',
            'null_padding_feature',
            'null_padding_target',
            'score_normalization',
            'threshold_type',   # 'absolute' (likelihood / score value) or 'percentile' of the split's scores
            ] + [f'{split}_{metric}' for split in EVALUATION_SPLITS for metric in SPLIT_METRICS] \
              + [f'{split}_{metric}' for split in NAB_SPLITS for metric in NAB_SPLIT_METRICS]

    model = experiment_config.use_model
    null_padding_feature = experiment_config.null_padding_feature
    null_padding_target = experiment_config.null_padding_target
    fill_nan_value = experiment_config.fill_nan
    result_rows = []   # one dict per setting, turned into result_df at the end
    post_processing_strategies = experiment_config.post_processing_strategies
    topks = [parse_topk(topk) for topk in experiment_config.topks]   # None = mean over all sensors
    anomaly_thresholds = experiment_config.anomaly_thresholds
    # Likelihood strategies: absolute thresholds (anomaly_thresholds) or percentiles (distribution_anomaly_thresholds)
    likelihood_threshold_mode = experiment_config.get('likelihood_threshold_mode', 'absolute')
    # Score strategies: percentile thresholds (distribution_anomaly_thresholds) or the best-F1 threshold on F1_OPTIMAL_SPLIT
    score_threshold_mode = experiment_config.get('score_threshold_mode', 'percentile')
    assert score_threshold_mode in LIKELIHOOD_THRESHOLD_MODES, f'Unknown score_threshold_mode: {score_threshold_mode}'
    assert likelihood_threshold_mode in LIKELIHOOD_THRESHOLD_MODES, \
        f'Unknown likelihood_threshold_mode: {likelihood_threshold_mode}'
    distribution_anomaly_thresholds = experiment_config.distribution_anomaly_thresholds
    long_window_values = experiment_config.long_windows
    short_window_values = experiment_config.short_windows
    # Normalisation of the per-node errors before the top-k aggregation ('global' or 'per_node') of test_scores
    score_normalization = score_normalization or score_normalizations(experiment_config)[0]

    # post_processing = experiment_config.post_processing if 'post_processing' in experiment_config else None


    # if post_processing:
    #     is_anomalies_df = pd.DataFrame()
    #     anomaly_thresholds = post_processing.anomaly_thresholds
    #     for index, anomaly_threshold in enumerate(anomaly_thresholds):
    #         is_anomalies, likelihoods, reconstruction_error = label_reconstruction_errors_with_mahalanobis(data_loader.test_index,
    #                                                                                       post_processing,
    #                                                                                       mahalanobis_distances, anomaly_threshold)
    #
    #
    #         visualization_df = pd.DataFrame({
    #             '5XX_count': data_loader.count_5xx,  # Adjust as needed for your data
    #             'true_anomaly': data_loader.test_labels,  # This is what the function expects
    #             'predicted_anomaly': is_anomalies.values,  # The output of the model
    #             'anomaly_likelihood': likelihoods,
    #             'reconstruction_error': reconstruction_error
    #         })
    #         visualization_df.index = data_loader.test_index
    #         is_anomalies_df[f'is_anomaly_{index}'] = is_anomalies.values
    #
    #         print("SAMPLE RESULT DF: ", visualization_df.head())
    #
    #         # model_dir
    #         # visualization_file = os.path.join(model_dir, f'{model}_visualization.png')
    #         # plot_results(visualization_df, visualization_df['predicted_anomaly'], data_loader.anomaly_windows_test,
    #         #              result_directory=visualization_file, \
    #         #              model=model)
    #
    #         # plot_results(visualization_df, is_anomalies.values, data_loader.anomaly_windows_test, result_directory=?, model=?)
    #
    #         # Evaluate performance based on ground truth
    #         precision, recall, f1, accuracy, conf_matrix, mcc = evaluate_performance(data_loader.test_labels,
    #                                                                                  is_anomalies.values)
    #
    #         log.info(
    #             f"Experiment results - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, ConfusionMatrix: {conf_matrix}")
    #
    #         # Calculate the weighted NAB score and normalized NAB score
    #         raw_nab_score_standard, normalized_nab_score_standard, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
    #             visualization_df, data_loader.anomaly_windows_test, 'standard', true_col='true_anomaly',
    #             pred_col='predicted_anomaly'
    #         )
    #
    #         # Calculate the weighted NAB score and normalized NAB score
    #         raw_nab_score_reward_fn, normalized_nab_score_reward_fn, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
    #             visualization_df, data_loader.anomaly_windows_test, 'reward_fn', true_col='true_anomaly',
    #             pred_col='predicted_anomaly'
    #         )
    #
    #         # Return all values needed for grid_search
    #
    #         standard_score = raw_nab_score_standard
    #         reward_fn_score = raw_nab_score_reward_fn
    #         standard_score_normalized = normalized_nab_score_standard
    #         reward_fn_score_normalized = normalized_nab_score_reward_fn
    #
    #         new_row = {
    #             'confusion_matrix': conf_matrix,
    #             'scale_prediction': False,
    #             'topk': 1,
    #             'anomaly_threshold': anomaly_threshold,
    #             'long_window': 0,
    #             'short_window': 0,
    #             'standard_raw': standard_score,
    #             'reward_fn_raw': reward_fn_score,
    #             'standard_normalized': standard_score_normalized,
    #             'reward_fn_normalized': reward_fn_score_normalized,
    #             'precision': precision,
    #             'recall': recall,
    #             'f1': f1,
    #             'detection_counters': detection_counters,
    #             'accuracy': accuracy,
    #             # conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score,
    #         }
    #         result_df.loc[len(result_df)] = new_row
    #
    #     return result_df, is_anomalies_df

    likelihood_thresholds = (anomaly_thresholds if likelihood_threshold_mode == 'absolute'
                             else experiment_config.distribution_anomaly_thresholds)
    _, with_windows = likelihood_window_pairs(experiment_config)
    # Absolute thresholds of the score strategies: the num_absolute thresholds with the highest F1 on
    # F1_OPTIMAL_SPLIT, best first, {strategy: [value, ...]}
    f1_optimal_thresholds = {}
    if score_threshold_mode == 'absolute':
        num_absolute = experiment_config.get('score_absolute_num_thresholds', 5)
        min_gap = experiment_config.get('score_absolute_min_gap', 0.1)
        assert F1_OPTIMAL_SPLIT in split_scores, f'Absolute score thresholds need the {F1_OPTIMAL_SPLIT} split scores'
        reference = split_scores[F1_OPTIMAL_SPLIT]
        reference_labels = reference['is_anomaly'].to_numpy().astype(int)
        for strategy in set(post_processing_strategies) & set(SCORE_STRATEGIES):
            if reference_labels.sum() == 0:
                log.warning(f'No anomaly in {F1_OPTIMAL_SPLIT}: no best-F1 threshold, no setting for {strategy}')
                continue
            _, score = label_from_scores(reference, strategy, None, 50, 0, 0, with_windows)
            f1_optimal_thresholds[strategy] = _top_f1_thresholds(score, reference_labels, num_absolute, min_gap)
            log.info(f'{strategy}: {len(f1_optimal_thresholds[strategy])} highest-F1 thresholds on {F1_OPTIMAL_SPLIT}: '
                     f'{[f"{t:.6g}" for t in f1_optimal_thresholds[strategy]]}')
    params_combinations = []
    for post_processing_strategy in post_processing_strategies:
        if post_processing_strategy == 'likelihood':
            params_combinations_new = itertools.product([post_processing_strategy],
                                                    topks,
                                                    likelihood_thresholds,
                                                    long_window_values,
                                                    short_window_values)
            params_combinations.extend(list(params_combinations_new))
        elif post_processing_strategy == 'likelihood_mahalanobis':
            # The Mahalanobis distance covers all sensors: no top-k
            params_combinations_new = itertools.product([post_processing_strategy],
                                                        [None],
                                                        likelihood_thresholds,
                                                        long_window_values,
                                                        short_window_values)
            params_combinations.extend(list(params_combinations_new))
        elif post_processing_strategy in SCORE_STRATEGIES:
            # Scores over all sensors (mean error, Mahalanobis distance): no top-k, no windows. Percentile
            # thresholds from the config, or the highest-F1 thresholds on F1_OPTIMAL_SPLIT as absolute values
            if score_threshold_mode == 'percentile':
                params_combinations.extend((post_processing_strategy, None, threshold, 0, 0)
                                           for threshold in distribution_anomaly_thresholds)
            else:
                params_combinations.extend((post_processing_strategy, None, threshold, 0, 0)
                                           for threshold in f1_optimal_thresholds.get(post_processing_strategy, []))
        else:
            raise ValueError(f'Unsupported post-processing strategy: {post_processing_strategy}')
    num_combinations = len(params_combinations)
    is_anomalies_columns = {}   # joined at the end: inserting columns one by one fragments a DataFrame
    for index, (post_processing_strategy, topk, anomaly_threshold, long_window, short_window) in tqdm(enumerate(params_combinations), desc='running grid search', total=num_combinations):
        print(f'post_processing_strategy: {post_processing_strategy}')
        print(f'topk: {topk} and anomaly_threshold: {anomaly_threshold} long window: {long_window} short window: {short_window}')
        is_anomalies, scores = label_from_scores(test_scores, post_processing_strategy, topk, anomaly_threshold,
                                                 long_window, short_window, with_windows, likelihood_threshold_mode,
                                                 score_threshold_mode)
        is_anomalies = pd.Series(is_anomalies, index=data_loader.test_index)
        visualization_df = pd.DataFrame({
            '5XX_count': data_loader.count_5xx,  # Adjust as needed for your data
            'true_anomaly': data_loader.test_labels,  # This is what the function expects
            'predicted_anomaly': is_anomalies.values,  # The output of the model
            'anomaly_likelihood': scores,
        })

        visualization_df.index = data_loader.test_index

        print("SAMPLE RESULT DF: ", visualization_df.head())
        is_anomalies_columns[f'is_anomaly_{index}'] = is_anomalies.values

        # model_dir
        # visualization_file = os.path.join(model_dir, f'{model}_visualization.png')
        # plot_results(visualization_df, visualization_df['predicted_anomaly'], data_loader.anomaly_windows_test,
        #              result_directory=visualization_file, \
        #              model=model)

        # plot_results(visualization_df, is_anomalies.values, data_loader.anomaly_windows_test, result_directory=?, model=?)


        # Evaluate performance based on ground truth
        precision, recall, f1, accuracy, conf_matrix, mcc = evaluate_performance(data_loader.test_labels, is_anomalies.values)

        log.info(
            f"Experiment results - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, ConfusionMatrix: {conf_matrix}")

        # Calculate the weighted NAB score and normalized NAB score
        raw_nab_score_standard, normalized_nab_score_standard, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
            visualization_df, data_loader.anomaly_windows_test, 'standard', true_col='true_anomaly', pred_col='predicted_anomaly'
        )

        # Calculate the weighted NAB score and normalized NAB score
        raw_nab_score_reward_fn, normalized_nab_score_reward_fn, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
            visualization_df, data_loader.anomaly_windows_test, 'reward_fn', true_col='true_anomaly',
            pred_col='predicted_anomaly'
        )

        # Return all values needed for grid_search


        standard_score =raw_nab_score_standard
        reward_fn_score = raw_nab_score_reward_fn
        standard_score_normalized = normalized_nab_score_standard
        reward_fn_score_normalized = normalized_nab_score_reward_fn

        new_row = {
            'confusion_matrix': conf_matrix,
            'post_processing_strategy': post_processing_strategy,
            'topk': topk,
            'anomaly_threshold': anomaly_threshold,
            'long_window': long_window,
            'short_window': short_window,
            'standard_raw': standard_score,
            'reward_fn_raw': reward_fn_score,
            'standard_normalized': standard_score_normalized,
            'reward_fn_normalized': reward_fn_score_normalized,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_counters': detection_counters,
            'accuracy': accuracy,
            'model' : model,
            'fill_nan_value': fill_nan_value,
            'null_padding_feature': null_padding_feature,
            'null_padding_target': null_padding_target,
            'score_normalization': score_normalization,
            'threshold_type': threshold_type(post_processing_strategy, likelihood_threshold_mode, score_threshold_mode),
            # conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score,
        }
        for split, scores_of_split in split_scores.items():
            metrics = _split_metrics(scores_of_split, post_processing_strategy, topk, anomaly_threshold,
                                     long_window, short_window, with_windows, pr_caches[split],
                                     likelihood_threshold_mode, split_anomaly_windows.get(split), score_threshold_mode)
            new_row.update({f'{split}_{metric}': value for metric, value in metrics.items()})
        result_rows.append(new_row)

    result_df = pd.DataFrame(result_rows, columns=columns)
    # topk mixes ints and None (no top-k): kept as objects so the CSV shows 1, not 1.0, as before
    result_df['topk'] = pd.Series([row['topk'] for row in result_rows], index=result_df.index, dtype=object)
    is_anomalies_df = pd.DataFrame(is_anomalies_columns, index=data_loader.test_index)
    return result_df, is_anomalies_df

def label_reconstruction_errors_with_mahalanobis(index, post_processing, mahalanobis_distances, anomaly_threshold):
    print('Post processing', post_processing)
    if post_processing.distance == 'mahala':
        # num_timestamps, num_nodes, num_feats = reconstruction_errors.shape
        # reconstruction_error_raw = reconstruction_errors
        #
        # # mahala_file = './mahala.csv'
        # # if not os.path.exists(mahala_file) or post_processing.recreate == False:
        #     # num_samples, num_nodes, num_feats = reconstruction_errors.shape
        #     # reconstruction_errors = reconstruction_errors.reshape((num_samples, num_nodes * num_feats))
        #     # if post_processing.scale_prediction:
        #     #     reconstruction_errors = get_full_err_scores(reconstruction_errors)
        #     # reconstruction_errors_normalized = MinMaxScaler().fit_transform(reconstruction_errors)
        #     # reconstruction_errors = reconstruction_errors.reshape((num_samples, num_nodes, num_feats))
        # flattened = reconstruction_error_raw.reshape(num_timestamps, -1)
        # mean_vec = np.mean(flattened, axis=0)
        # cov_matrix = np.cov(flattened, rowvar=False)
        # inv_cov_matrix = pinv(cov_matrix)
        #
        # # Compute Mahalanobis distance at each timestamp
        # mahalanobis_distances = np.array([
        #     mahalanobis(flattened[t], mean_vec, inv_cov_matrix)
        #     for t in range(num_timestamps)
        # ])
        #
        # #     print('Saving mahala.csv...')
        # #     pd.DataFrame(mahalanobis_distances).to_csv(mahala_file, index=False)
        # # else:
        # #     mahalanobis_distances = pd.read_csv(mahala_file).values

        threshold = np.percentile(mahalanobis_distances, anomaly_threshold)
        is_anomalies = (mahalanobis_distances > threshold).astype(int)

        reconstruction_error_full = mahalanobis_distances

        # reconstruction_error_raw = reconstruction_error_raw.max(axis=2)
        # reconstruction_error_full = reconstruction_error_raw.max(axis=-1)
        # reconstruction_error_full = np.sort(reconstruction_error_raw, axis=1)[:, -topk:].sum(axis=-1)
        # reconstruction_error_full = rankdata(reconstruction_error_full, method="ordinal")

        # reconstruction_error_full = MinMaxScaler().fit_transform(reconstruction_error_raw.reshape(-1,1)).reshape(-1)
        # threshold = np.percentile(reconstruction_error_full, anomaly_threshold)
        # is_anomalies = (reconstruction_error_full > threshold).astype(int)
        likelihoods = MinMaxScaler().fit_transform(mahalanobis_distances.reshape(-1, 1)).reshape(-1)
        return pd.Series(is_anomalies, index=index), likelihoods, reconstruction_error_full

def evaluate_performance(y_true, y_pred):
    """
    Evaluates the performance of the anomaly detection using various classification metrics.

    Parameters:
    - y_true: Array, true labels.
    - y_pred: Array, predicted labels.

    Returns:
    - precision: Float, precision of the model.
    - recall: Float, recall of the model.
    - f1: Float, F1-score of the model.
    - accuracy: Float, accuracy of the model.
    - conf_matrix: Array, confusion matrix.
    - mcc: Float, Matthews correlation coefficient.
    """
    # Binary 0 / 1 labels: the counts give the same values as sklearn's precision_score, recall_score,
    # f1_score (0 when undefined), accuracy_score, confusion_matrix and matthews_corrcoef, without
    # their input validation, which made up most of the grid-search time (~17 ms per call)
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    true_pos, pred_pos = y_true == 1, y_pred == 1
    if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred, (0, 1)).all():
        raise ValueError('evaluate_performance expects binary 0 / 1 labels and predictions')
    tp = int(np.count_nonzero(true_pos & pred_pos))
    fp = int(np.count_nonzero(~true_pos & pred_pos))
    fn = int(np.count_nonzero(true_pos & ~pred_pos))
    tn = int(np.count_nonzero(~true_pos & ~pred_pos))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    accuracy = (tp + tn) / len(y_true)
    # sklearn's matrix covers the classes present in y_true or y_pred only
    classes = np.union1d(y_true, y_pred)
    conf_matrix = [tn, fp, fn, tp] if len(classes) == 2 else [len(y_true)]
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0

    return precision, recall, f1, accuracy, conf_matrix, mcc
if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()

