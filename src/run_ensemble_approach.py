import hydra
import sys
from omegaconf import DictConfig
import os
import itertools

import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
# Configure logging
import logging

from ibm_dataset_loader import IBMDatasetLoader
from run_preprocessing import set_random_seed
from run_training_single_model import evaluate_performance
from nab_scoring import calculate_nab_score_with_window_based_tp_fn
from utils import get_project_root, NumpyEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def analyze_reconstruction_errors_essembles(ensembles_config, data_preparation_config, model_configs, experiment_config):
    http_codes = ensembles_config.http_codes
    aggregations = ensembles_config.aggregations

    # Train the autoencoder based on model type
    # Define the path to the trained models directory
    project_root_dir = get_project_root()
    trained_models_dir = os.path.join(project_root_dir, experiment_config.model_save_path)
    os.makedirs(trained_models_dir, exist_ok=True)  # Ensure the directory exists

    ensemble_model_dir = os.path.join(trained_models_dir, 'ensemble')
    os.makedirs(ensemble_model_dir, exist_ok=True)

    selected_models_list = list(itertools.product(http_codes, aggregations))
    print('Selected_model_list length', len(selected_models_list))
    for i in range(1,2):
        if i > len(selected_models_list):
            return

        selected_models_combinations = list(itertools.combinations(selected_models_list, i))

        for selected_models in tqdm(selected_models_combinations, desc='Running for each combination...', total=len(selected_models_combinations)):
            selected_models_id = '+'.join([f'{code}_{agg}' for code, agg in selected_models])
            predicted_label_all_file = os.path.join(ensemble_model_dir, f'predicted_labels_of_all_{selected_models_id}.npy')
            ensemble_meta_data_file = os.path.join(ensemble_model_dir, f'ensemble_meta_data_{selected_models_id}.json')
            data_loader_first = None
            is_duplicated = False

            if os.path.exists(predicted_label_all_file) and os.path.exists(ensemble_meta_data_file):
                predicted_label_matrix = np.load(predicted_label_all_file)
                print('Loaded saved aggregated label matrix with shape', predicted_label_matrix.shape)
                ensemble_meta_data = json.load(open(ensemble_meta_data_file))
                print('Loaded saved ensemble meta data', ensemble_meta_data.keys())
                chosen_models = ensemble_meta_data['models']

                model_collections = list(itertools.product(http_codes, aggregations))
                if len(model_collections) == len(chosen_models):
                    for http_code, aggregation in model_collections:
                        exist = False
                        for m in chosen_models:
                            if http_code in m and aggregation in m:
                                exist = True
                                break
                        if exist == False:
                            is_duplicated = exist
                            break
                        is_duplicated = exist

                    # is_duplicated = np.any([(http_code in m and aggregation in m) for m in chosen_models for http_code, aggregation in model_collections])
                if is_duplicated:
                    selected_model = list(itertools.product(http_codes, aggregations))[0]
                    # num_ensembles = len(selected_models)

                    http_code, aggregation = selected_model
                    data_preparation_config.features_prep.filter.http_codes = [http_code]
                    data_preparation_config.features_prep.filter.aggregations = [aggregation]
                    data_loader = IBMDatasetLoader(data_preparation_config)

                    data_loader_first = data_loader

            if is_duplicated == False:
                # num_ensembles = len(selected_models)
                label_dictionary = dict()
                grid_search_params = dict()

                # data_identification = []
                for (http_code, aggregation) in tqdm(selected_models, desc='Collect predicted labels for ensemble models', total=len(selected_models)):
                    data_preparation_config.features_prep.filter.http_codes = [http_code]
                    data_preparation_config.features_prep.filter.aggregations = [aggregation]
                    data_loader = IBMDatasetLoader(data_preparation_config)

                    selected_group_mode = data_loader.selected_group_mode
                    # data_identification.append(selected_group_mode)
                    # if selected_group_mode not in label_dictionary:
                    #     label_dictionary[selected_group_mode]
                    if data_loader_first is None:
                        data_loader_first = data_loader


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

                    # Define the model filename based on the model type
                    # model_filename = f"{trained_models_dir}{model_type}_autoencoder.h5"


                    model = models[0]
                    print("Running model", model)
                    model_config = model_configs[model]
                    # model_filename_extension = 'h5' if model_type in tf_models else 'pt'

                    model_dir = os.path.join(trained_models_dir, selected_group_mode, model)
                    os.makedirs(model_dir, exist_ok=True)
                    # model_filename = os.path.join(model_dir, model_config['model_filename'])

                    # Check if the model file exists
                    # if os.path.exists(model_filename) and experiment_config.retrain == False:
                    #     print(f"Loading trained model: {model_filename}")
                    #     graph_config = dict({"node_features": data_loader.get_num_node_features(),
                    #                          'slide_win': experiment_config.slide_win,
                    #                          'batch_size': batch_size,
                    #                          'hidden_units': 32,
                    #                          'num_nodes': data_loader.num_nodes,
                    #                          'device': DEVICE})
                    #     model_wrapper = load_wrapper(model_name=model, config=graph_config,
                    #                                  static_edge_index=data_loader.get_edges_as_tensor(device=DEVICE))
                    #     model_wrapper.load(model_filename)
                    # else:
                    #     if experiment_config.retrain:
                    #         print(f"Using model: {model}. Re-training...")
                    #     else:
                    #         print(f"No trained model found for {model}. Training a new model...")
                    #
                    #     clear_folder(model_dir)
                    #
                    #     graph_config = dict({"node_features": data_loader.get_num_node_features(),
                    #                          'slide_win': experiment_config.slide_win,
                    #                          'batch_size': batch_size,
                    #                          'hidden_units': 32,
                    #                          'num_nodes': data_loader.num_nodes,
                    #                          'device': DEVICE})
                    #     model_wrapper = load_wrapper(model_name=model, config=graph_config,
                    #                                  static_edge_index=data_loader.get_edges_as_tensor(device=DEVICE))
                    #     history = model_wrapper.train(train_loader, valid_loader, epochs=experiment_config.get('epochs'))
                    #     model_wrapper.save(model_filename)
                    #     plot_training_history(model_name='A3TGCN', training_history=history,
                    #                           model_save_dir=os.path.dirname(model_filename))

                    # predictions_file = os.path.join(model_dir,'reconstruction_errors.npy')
                    # if not os.path.exists(predictions_file):
                    #     X_test_predictions, reconstruction_error_raw, test_loss = model_wrapper.predict(test_loader)
                    #     with open(predictions_file, 'wb') as f:
                    #         numpy.save(f, reconstruction_error_raw)
                    #         log.info(f"Reconstruction errors saved to {predictions_file}")
                    # else:
                    #     with open(predictions_file, 'rb') as f:
                    #         reconstruction_error_raw = np.load(f)
                    #         log.info(f"Reconstruction errors loaded from {predictions_file}")
                    #
                    # assert reconstruction_error_raw.shape[0] == len(data_loader.test_index)

                    label_ensemble_file = os.path.join(model_dir, f'{model}_predictions_for_assembles.csv')
                    if not os.path.exists(label_ensemble_file):
                        raise RuntimeError('Label Ensembles file not found')
                    else:
                        label_essemble_df = pd.read_csv(label_ensemble_file)

                    grid_search_file = os.path.join(model_dir, f'{model}_grid_search.csv')
                    if not os.path.exists(grid_search_file):
                        raise RuntimeError('Grid Search file not found')
                    else:
                        grid_search_df = pd.read_csv(grid_search_file)

                        top1_NAB_standard_index = grid_search_df['NAB_standard_rank'].idxmin()
                        top1_NAB_reward_fn_index = grid_search_df['NAB_reward_fn_rank'].idxmin()

                        # if grid_search_params == None:
                        #     grid_search_params = grid_search_df.to_dict(orient='records')

                    assert label_essemble_df.shape[0] == len(data_loader.test_index)
                    assert label_essemble_df.shape[1] == grid_search_df.shape[0]

                    # label_dictionary[selected_group_mode] = label_essemble_df.values
                    grid_search_params[selected_group_mode] = grid_search_df.iloc[[top1_NAB_standard_index, top1_NAB_reward_fn_index],:].to_dict(orient='records')
                    label_dictionary[selected_group_mode] = label_essemble_df.values[:,[top1_NAB_standard_index, top1_NAB_reward_fn_index]]

                    # is_anomalies, likelihoods, reconstruction_error = label_reconstruction_errors(reconstruction_errors, )
                    # Call grid search or other functions
                log.info("Starting ensembles analysis")
                print(label_dictionary.keys())
                for feature_group in label_dictionary.keys():
                    print('Grid search result shape', label_dictionary[feature_group].shape)

                predicted_label_matrix = np.stack(list(label_dictionary.values()), axis=0)
                print('Predicted labels of all models with shape', predicted_label_matrix.shape)
                np.save(predicted_label_all_file, predicted_label_matrix)
                print('Saved predicted label matrix of all models to', predicted_label_all_file)

                ensemble_meta_data = dict()
                ensemble_meta_data['models'] = list(label_dictionary.keys())
                ensemble_meta_data['grid_params'] = grid_search_params
                with open(ensemble_meta_data_file, 'w') as f:
                    json.dump(ensemble_meta_data, f, cls=NumpyEncoder, indent=4)
                    print('Saved ensemble meta data to ', ensemble_meta_data_file)

                assert ensemble_meta_data != None

                aggregated_label_over_models = (predicted_label_matrix.sum(axis=0) >= 1)
                print('Aggregated label matrix shape', aggregated_label_over_models.shape)

                columns = [
                    'standard_normalized',
                    'reward_fn_normalized',
                    'detection_counters',
                    'confusion_matrix',
                    'scale_prediction',
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
                ensemble_result_df = pd.DataFrame(columns=columns)

                # Calculate the weighted NAB score and normalized NAB score
                num_timestamps , num_grid_search = aggregated_label_over_models.shape
                grid_search_params = ensemble_meta_data['grid_params']
                for i in range(num_grid_search):
                    is_anomalies = aggregated_label_over_models[:, i]
                    visualization_df = pd.DataFrame({
                        'true_anomaly': data_loader_first.test_labels,  # This is what the function expects
                        'predicted_anomaly': is_anomalies,  # The output of the model
                    })
                    visualization_df.index = data_loader_first.test_index

                    raw_nab_score_standard, normalized_nab_score_standard, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
                        visualization_df, data_loader_first.anomaly_windows_test, 'standard', true_col='true_anomaly',
                        pred_col='predicted_anomaly'
                    )

                    # Calculate the weighted NAB score and normalized NAB score
                    raw_nab_score_reward_fn, normalized_nab_score_reward_fn, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
                        visualization_df, data_loader_first.anomaly_windows_test, 'reward_fn', true_col='true_anomaly',
                        pred_col='predicted_anomaly'
                    )

                    # Evaluate performance based on ground truth
                    precision, recall, f1, accuracy, conf_matrix, mcc = evaluate_performance(data_loader_first.test_labels,
                                                                                             is_anomalies)

                    # Return all values needed for grid_search
                    standard_score = raw_nab_score_standard
                    reward_fn_score = raw_nab_score_reward_fn
                    standard_score_normalized = normalized_nab_score_standard
                    reward_fn_score_normalized = normalized_nab_score_reward_fn


                    new_row = {
                        'confusion_matrix': conf_matrix,
                        'scale_prediction': None,
                        'topk': None,
                        'anomaly_threshold': None,
                        'long_window': None,
                        'short_window': None,
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
                    ensemble_result_df.loc[len(ensemble_result_df)] = new_row

                    for model_id, grid_search_param in grid_search_params.items():
                        grid_search_param = grid_search_param[i]
                        new_row = {
                            'confusion_matrix': grid_search_param['confusion_matrix'],
                            'scale_prediction': grid_search_param['scale_prediction'],
                            'topk': grid_search_param['topk'],
                            'anomaly_threshold': grid_search_param['anomaly_threshold'],
                            'long_window': grid_search_param['long_window'],
                            'short_window': grid_search_param['short_window'],
                            'standard_raw': grid_search_param['standard_raw'],
                            'reward_fn_raw': grid_search_param['reward_fn_raw'],
                            'standard_normalized': standard_score_normalized,
                            'reward_fn_normalized': reward_fn_score_normalized,
                            'precision': grid_search_param['precision'],
                            'recall': grid_search_param['recall'],
                            'f1': grid_search_param['f1'],
                            'detection_counters': grid_search_param['detection_counters'],
                            'accuracy': grid_search_param['accuracy'],
                            # conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score,
                        }
                        ensemble_result_df.loc[len(ensemble_result_df)] = new_row

                # print(data_identification)
                ensemble_meta_data_file = os.path.join(ensemble_model_dir, f'ensemble_final_result_{"+".join(ensemble_meta_data["models"])}.csv')
                print('Save final ensemble result in', ensemble_meta_data_file)
                ensemble_result_df.to_csv(ensemble_meta_data_file, index=False)
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

    ensembles_config = experiment_config.ensembles if 'ensembles' in experiment_config else None
    if ensembles_config:
        analyze_reconstruction_errors_essembles(ensembles_config, data_preparation_config=data_preparation_config, model_configs=model_configs,
                                          experiment_config=experiment_config)

if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()