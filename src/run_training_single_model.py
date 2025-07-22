import random
import sys
import os
import hydra
from omegaconf import DictConfig
import logging
import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import mahalanobis
from numpy.linalg import pinv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

from ibm_dataset_loader import IBMDatasetLoader
from model_wrappers.A3TGCNWrapper import A3TGCNWrapper
from model_wrappers.GRUWrapper import GRUWrapper
from plotting_module import plot_training_history
from anomaly_likelihood import compute_anomaly_likelihood
from nab_scoring import calculate_nab_score_with_window_based_tp_fn
from utils import clear_folder, get_project_root, get_full_err_scores, set_random_seed, calculate_mahalanobis_distance

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

    if model_name == 'A3TGCN':
        return A3TGCNWrapper(node_features, periods, static_edge_index, static_edge_weight, batch_size=batch_size, device=device)
    elif model_name == 'GRU':
        return GRUWrapper(num_nodes, node_features, hidden_units, layer_dim=1, batch_size=batch_size, device=device)
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


    # task_id = sys.argv[0]
    # print(f"Task ID: {task_id}")
    # model = experiment_config.use_models[task_id]
    # print(f'Using model: {model}')

    ibm_dataset_loader = IBMDatasetLoader(data_preparation_config)

    selected_group_mode = ibm_dataset_loader.selected_group_mode

    analyze_reconstruction_errors(ibm_dataset_loader, selected_group_mode, model_configs=model_configs, experiment_config=experiment_config)

def analyze_reconstruction_errors(data_loader, selected_group_mode, model_configs, experiment_config):

    mode = experiment_config.get('mode', 'single')
    if mode == 'single':
        models = [experiment_config.get('use_model', 'A3TGCN')]
    else:
        models = experiment_config.get('use_models', ['A3TGCN'])

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = experiment_config.train_batch_size

    train_loader, valid_loader, test_loader, edge_index = data_loader.get_index_dataset(
        window_size=experiment_config.slide_win,
        batch_size=batch_size,
        device=DEVICE)
    print("Training dataset batches", len(train_loader))
    print("Validation dataset batches", len(valid_loader))
    print("Testing dataset batches", len(test_loader))

    # Train the autoencoder based on model type
    # Define the path to the trained models directory
    project_root_dir = get_project_root()
    trained_models_dir = os.path.join(project_root_dir, experiment_config.model_save_path)
    os.makedirs(trained_models_dir, exist_ok=True)  # Ensure the directory exists

    # Define the model filename based on the model type
    # model_filename = f"{trained_models_dir}{model_type}_autoencoder.h5"


    for model in models:
        print("Running model", model)
        model_config = model_configs[model]
        # model_filename_extension = 'h5' if model_type in tf_models else 'pt'

        model_dir = os.path.join(trained_models_dir, selected_group_mode, model)
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, model_config['model_filename'])
        # Check if the model file exists
        if os.path.exists(model_filename) and experiment_config.retrain == False:
            print(f"Loading trained model: {model_filename}")
            graph_config = dict({"node_features": data_loader.get_num_node_features(),
                                 'slide_win': experiment_config.slide_win,
                                 'batch_size': batch_size,
                                 'hidden_units': 32,
                                 'num_nodes': data_loader.num_nodes,
                                 'device': DEVICE})
            model_wrapper = load_wrapper(model_name=model, config=graph_config,
                                         static_edge_index=data_loader.get_edges_as_tensor(device=DEVICE),
                                         static_edge_weight=data_loader.get_edge_weights_as_tensor(device=DEVICE))
            model_wrapper.load(model_filename)
        else:
            if experiment_config.retrain:
                print(f"Using model: {model}. Re-training...")
            else:
                print(f"No trained model found for {model}. Training a new model...")

            clear_folder(model_dir)

            graph_config = dict({"node_features": data_loader.get_num_node_features(),
                                 'slide_win': experiment_config.slide_win,
                                 'batch_size': batch_size,
                                 'hidden_units': 32,
                                 'num_nodes': data_loader.num_nodes,
                                 'device': DEVICE})
            model_wrapper = load_wrapper(model_name=model, config=graph_config,
                                         static_edge_index=data_loader.get_edges_as_tensor(device=DEVICE),
                                         static_edge_weight=data_loader.get_edge_weights_as_tensor(device=DEVICE))
            history = model_wrapper.train(train_loader, valid_loader, epochs=experiment_config.get('epochs'))
            model_wrapper.save(model_filename)
            plot_training_history(model_name='A3TGCN', training_history=history,
                                  model_save_dir=os.path.dirname(model_filename))

        predictions_file = os.path.join(model_dir,'reconstruction_errors.npy')
        if not os.path.exists(predictions_file):
            X_test_predictions, reconstruction_error_raw, test_loss = model_wrapper.predict(test_loader)
            with open(predictions_file, 'wb') as f:
                np.save(f, reconstruction_error_raw)
                log.info(f"Reconstruction errors saved to {predictions_file}")
                mse_reconstruction_error_file = os.path.join(os.path.dirname(predictions_file), 'mse_error.txt')
                mse_reconstruction_error_raw = reconstruction_error_raw.mean(axis=-1).mean(axis=-1).mean(axis=-1)
                with open(mse_reconstruction_error_file, 'w') as f:
                    f.write(str(mse_reconstruction_error_raw))
                    log.info(f"MSE reconstruction errors saved to {mse_reconstruction_error_file}")
                mahalanobis_distances = calculate_mahalanobis_distance(reconstruction_error_raw)
                mahalanobis_distances_file = os.path.join(os.path.dirname(predictions_file), 'mahalanobis.npy')
                with open(mahalanobis_distances_file, 'wb') as f:
                    np.save(f, mahalanobis_distances)
                    log.info(f"Mahalanobis_distances saved to {mahalanobis_distances_file}")
        else:
            with open(predictions_file, 'rb') as f:
                reconstruction_error_raw = np.load(f)
                log.info(f"Reconstruction errors loaded from {predictions_file}")
                mse_reconstruction_error_file = os.path.join(os.path.dirname(predictions_file), 'mse_error.txt')
                mse_reconstruction_error_raw = reconstruction_error_raw.mean(axis=-1).mean(axis=-1).mean(axis=-1)
                with open(mse_reconstruction_error_file, 'w') as f:
                    f.write(str(mse_reconstruction_error_raw))
                    log.info(f"MSE reconstruction errors saved to {mse_reconstruction_error_file}")
                mahalanobis_distances_file = os.path.join(os.path.dirname(predictions_file), 'mahalanobis.npy')
                with open(mahalanobis_distances_file, 'rb') as mahala_f:
                    mahalanobis_distances = np.load(mahala_f)
                    log.info(f'Mahalanobis_distances loaded from {mahalanobis_distances_file}, having shape: {mahalanobis_distances.shape}')

        assert reconstruction_error_raw.shape[0] == mahalanobis_distances.shape[0]
        assert reconstruction_error_raw.shape[0] == len(data_loader.test_index)

        # is_anomalies, likelihoods, reconstruction_error = label_reconstruction_errors(reconstruction_errors, )
        # Call grid search or other functions
        log.info("Starting Grid Search for best parameters...")
        result_df, is_anomalies_df = grid_search_new(
            data_loader, reconstruction_error_raw, experiment_config=experiment_config, mahalanobis_distances=mahalanobis_distances)

        result_df.insert(1,'NAB_standard_rank', result_df['standard_normalized'].rank(ascending=False))
        result_df.insert(2, 'NAB_reward_fn_rank', result_df['reward_fn_normalized'].rank(ascending=False))
        assert result_df.shape[0] == is_anomalies_df.shape[1]

        grid_search_file = os.path.join(model_dir, f'{model}_grid_search.csv')

        result_df.to_csv(grid_search_file, index=False)
        log.info('Grid Search results saved to {}'.format(grid_search_file))

        max_NAB_standard_profile_index = result_df['standard_normalized'].idxmax()
        log.info(
            f"{model} models's best NAB score with standard profile: {result_df['standard_normalized'].max()}"
            f" with params {result_df.loc[max_NAB_standard_profile_index].values}")
        # is_anomalies_df[f'is_anomaly_max_NAB_standard_profile_{max_NAB_standard_profile_index}'] = is_anomalies_df[f'is_anomaly_{max_NAB_standard_profile_index}']

        max_NAB_reward_fn_profile_index = result_df['reward_fn_normalized'].idxmax()
        log.info(
            f"{model} models's best NAB score with reward_fn profile: {result_df['reward_fn_normalized'].max()}"
            f" with params {result_df.loc[max_NAB_reward_fn_profile_index].values}")
        # is_anomalies_df[f'is_anomaly_max_NAB_low_positive_profile_{max_NAB_low_positive_profile_index}'] = is_anomalies_df[f'is_anomaly_{max_NAB_low_positive_profile_index}']

        predictions_for_assembles_file = os.path.join(model_dir, f'{model}_predictions_for_assembles.csv')
        is_anomalies_df.to_csv(predictions_for_assembles_file, index=False)
        log.info('Prediction results saved to {}'.format(predictions_for_assembles_file))

        # # Train the autoencoder
        # X_train = pd.DataFrame(scaled_train_data, columns=cleaned_training_data.columns)
        # autoencoder, history = train_autoencoder(X_train, model_type= model_type, model_cfg=model_configs[model_type], experiment_config=experiment_config)
        #
        # # Save the trained model
        # autoencoder.save(model_filename)
        # print(f"Model saved: {model_filename}")

def grid_search_new(data_loader, reconstruction_errors, experiment_config, mahalanobis_distances=None):
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
            ]
    result_df = pd.DataFrame(columns=columns)
    post_processing_strategies = experiment_config.post_processing_strategies
    topks = experiment_config.topks
    anomaly_thresholds = experiment_config.anomaly_thresholds
    distribution_anomaly_thresholds = experiment_config.distribution_anomaly_thresholds
    long_window_values = experiment_config.long_windows
    short_window_values = experiment_config.short_windows

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

    params_combinations = []
    for post_processing_strategy in post_processing_strategies:
        if post_processing_strategy == 'likelihood':
            params_combinations_new = itertools.product([post_processing_strategy],
                                                    topks,
                                                    anomaly_thresholds,
                                                    long_window_values,
                                                    short_window_values)
            params_combinations.extend(list(params_combinations_new))
        else:
            params_combinations_new = itertools.product([post_processing_strategy],
                                                        topks,
                                                        distribution_anomaly_thresholds,
                                                        [0],
                                                        [0])
            params_combinations.extend(list(params_combinations_new))
    num_combinations = len(params_combinations)
    is_anomalies_df = pd.DataFrame()
    for index, (post_processing_strategy, topk, anomaly_threshold, long_window, short_window) in tqdm(enumerate(params_combinations), desc='running grid search', total=num_combinations):
        print(f'post_processing_strategy: {post_processing_strategy}')
        print(f'topk: {topk} and anomaly_threshold: {anomaly_threshold} long window: {long_window} short window: {short_window}')
        is_anomalies, likelihoods, reconstruction_error = label_reconstruction_errors(data_loader.test_index, reconstruction_errors, mahalanobis_distances, post_processing_strategy, topk, anomaly_threshold, long_window, short_window)

        # Create results DataFrame for evaluation
        visualization_df = pd.DataFrame({
            '5XX_count': data_loader.count_5xx,  # Adjust as needed for your data
            'true_anomaly': data_loader.test_labels,  # This is what the function expects
            'predicted_anomaly': is_anomalies.values,  # The output of the model
            'anomaly_likelihood': likelihoods,
            'reconstruction_error': reconstruction_error
        })
        visualization_df.index = data_loader.test_index

        print("SAMPLE RESULT DF: ", visualization_df.head())

        is_anomalies_df[f'is_anomaly_{index}'] = is_anomalies.values

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
            # conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score,
        }
        result_df.loc[len(result_df)] = new_row

    is_anomalies_df.index = data_loader.test_index
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

def label_reconstruction_errors(index, reconstruction_errors, mahalanobis_distances, post_processing_strategy, topk, anomaly_threshold, long_window, short_window):
    # reconstruction_error_full = np.mean(
    #     np.power(X_test - X_test_predictions, 2), axis=1
    # )
    # print('Post processing', post_processing)
    is_anomalies_layer_1 = None
    num_timestamps, num_nodes, num_feats = reconstruction_errors.shape
    if post_processing_strategy == 'mahalanobis':
        # num_samples, num_nodes, num_feats = reconstruction_errors.shape
        # reconstruction_errors = reconstruction_errors.reshape((num_samples, num_nodes * num_feats))
        # reconstruction_errors = get_full_err_scores(reconstruction_errors)
        # reconstruction_errors_normalized = MinMaxScaler().fit_transform(reconstruction_errors)
        # reconstruction_errors = reconstruction_errors.reshape((num_samples, num_nodes, num_feats))

        reconstruction_error_raw = mahalanobis_distances
        reconstruction_error_raw = reconstruction_error_raw.reshape(num_timestamps, -1)
        # reconstruction_error_full = np.sort(reconstruction_error_raw, axis=1)[:, -topk:].mean(axis=-1)
        reconstruction_error_full = MinMaxScaler().fit_transform(np.sort(reconstruction_error_raw, axis=1)[:, -topk:].mean(axis=-1, keepdims=True)).reshape(-1)
        # reconstruction_error_full = MinMaxScaler().fit_transform(reconstruction_error_raw.mean(axis=-1, keepdims=True)).reshape(-1)
        threshold = np.percentile(reconstruction_error_full, anomaly_threshold)
        is_anomalies = (reconstruction_error_full > threshold).astype(int)
        likelihoods = reconstruction_error_full
        return pd.Series(is_anomalies, index=index), likelihoods, reconstruction_error_full
    if post_processing_strategy == 'max':
        # num_samples, num_nodes, num_feats = reconstruction_errors.shape
        # reconstruction_errors = reconstruction_errors.reshape((num_samples, num_nodes * num_feats))
        reconstruction_errors = get_full_err_scores(reconstruction_errors)
        # reconstruction_errors_normalized = MinMaxScaler().fit_transform(reconstruction_errors)
        # reconstruction_errors = reconstruction_errors.reshape((num_samples, num_nodes, num_feats))

        reconstruction_error_raw = reconstruction_errors
        reconstruction_error_raw = reconstruction_error_raw.reshape(num_timestamps, -1)
        # reconstruction_error_full = np.sort(reconstruction_error_raw, axis=1)[:, -topk:].mean(axis=-1)
        reconstruction_error_full = MinMaxScaler().fit_transform(np.sort(reconstruction_error_raw, axis=1)[:, -topk:].mean(axis=-1, keepdims=True)).reshape(-1)
        # reconstruction_error_full = MinMaxScaler().fit_transform(reconstruction_error_raw.mean(axis=-1, keepdims=True)).reshape(-1)
        threshold = np.percentile(reconstruction_error_full, anomaly_threshold)
        is_anomalies = (reconstruction_error_full > threshold).astype(int)
        likelihoods = reconstruction_error_full
        return pd.Series(is_anomalies, index=index), likelihoods, reconstruction_error_full

    # reconstruction_errors = reconstruction_errors.reshape(num_timestamps, -1)
    reconstruction_errors = get_full_err_scores(reconstruction_errors)
    reconstruction_error_raw = reconstruction_errors
    reconstruction_error_raw = reconstruction_error_raw.reshape(num_timestamps, -1)
    reconstruction_error_full = MinMaxScaler().fit_transform(np.sort(reconstruction_error_raw, axis=1)[:, -topk:].mean(axis=-1, keepdims=True)).reshape(-1)
    # reconstruction_error_raw = reconstruction_error_raw.mean(axis=-1)
    # reconstruction_error_full = MinMaxScaler().fit_transform(reconstruction_error_raw.mean(axis=-1, keepdims=True)).reshape(-1)
    is_anomalies_layer_1 = None
    # threshold = np.percentile(reconstruction_error_full, 99.8)
    # is_anomalies_layer_1 = (reconstruction_error_full > threshold).astype(int)
    reconstruction_error_full = np.power(reconstruction_error_full, 2)
    likelihoods = []
    for i in range(len(reconstruction_error_full)):
        likelihood = compute_anomaly_likelihood(
            reconstruction_error_full[:i + 1], long_window, short_window
        )
        likelihoods.append(likelihood)

    likelihoods = np.array(likelihoods)
    is_anomalies = (likelihoods > anomaly_threshold).astype(int)
    if is_anomalies_layer_1 is not None:
        is_anomalies = ((is_anomalies_layer_1+is_anomalies)>=1).astype(int)

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
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return precision, recall, f1, accuracy, conf_matrix, mcc
if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()

