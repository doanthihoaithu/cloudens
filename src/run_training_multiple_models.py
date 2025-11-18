import random
import sys
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf
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
from run_training_single_model import analyze_reconstruction_errors
from utils import clear_folder, get_project_root, get_full_err_scores, set_random_seed, calculate_mahalanobis_distance, \
    calculate_mahalanobis_distance_with_is_nan_mask, refine_reconstruction_error_with_is_nan_mask

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

    multiple_evaluation_config = cfg.multiple_evaluation
    models = multiple_evaluation_config.models
    slide_wins = multiple_evaluation_config.slide_wins
    http_codes = multiple_evaluation_config.subsets.http_codes
    aggregations = multiple_evaluation_config.subsets.aggregations
    fill_nan_values = multiple_evaluation_config.fill_nan_values
    null_padding_feature_list = multiple_evaluation_config.null_padding_feature
    null_padding_target_list = multiple_evaluation_config.null_padding_target

    combination_list = []
    for model in models:
        if 'GRU' == model:
            GRU_combinations = list(itertools.product(['GRU'],slide_wins,
                                            http_codes,
                                            aggregations,
                                            fill_nan_values,
                                            [False],
                                            [False]))
            combination_list.extend(GRU_combinations)
        else:
            graph_combinations = list(itertools.product([model], slide_wins,
                                            http_codes,
                                            aggregations,
                                            fill_nan_values,
                                            null_padding_target_list,
                                            null_padding_feature_list))
            combination_list.extend(graph_combinations)

    for config in tqdm(combination_list, total=len(combination_list), desc='Training multiple models....'):
        model, slide_win, http_code, aggregation, fill_nan, null_padding_feature, null_padding_target = config
        print(config)
        OmegaConf.update(cfg, 'evaluation.use_model', model)
        OmegaConf.update(cfg, 'evaluation.slide_win', slide_win)
        OmegaConf.update(cfg, 'evaluation.fill_nan', fill_nan)
        OmegaConf.update(cfg, 'evaluation.null_padding_feature', null_padding_feature)
        OmegaConf.update(cfg, 'evaluation.null_padding_target', null_padding_target)

        OmegaConf.update(cfg, 'data_preparation_pipeline.features_prep.filter.http_codes', [http_code])
        OmegaConf.update(cfg, 'data_preparation_pipeline.features_prep.filter.aggregations', [aggregation])

        data_preparation_config = cfg.data_preparation_pipeline
        experiment_config = cfg.evaluation
        model_configs = cfg.model_configs

        if experiment_config.use_model == 'GRU':
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

        analyze_reconstruction_errors(ibm_dataset_loader, selected_group_mode, model_configs=model_configs,
                                      experiment_config=experiment_config)

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

    # data_preparation_config = cfg.data_preparation_pipeline
    # if experiment_config.use_model == 'GRU':
    #     data_preparation_config.null_padding_feature = False
    #     data_preparation_config.null_padding_target = False
    #
    #     experiment_config.null_padding_feature = False
    #     experiment_config.null_padding_target = False
    #
    #
    # # task_id = sys.argv[0]
    # # print(f"Task ID: {task_id}")
    # # model = experiment_config.use_models[task_id]
    # # print(f'Using model: {model}')
    #
    # ibm_dataset_loader = IBMDatasetLoader(data_preparation_config)
    #
    # selected_group_mode = ibm_dataset_loader.selected_group_mode
    #
    # analyze_reconstruction_errors(ibm_dataset_loader, selected_group_mode, model_configs=model_configs, experiment_config=experiment_config)


if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()

