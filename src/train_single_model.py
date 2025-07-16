import random
import sys
import hydra
from omegaconf import DictConfig
import logging

import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # ibm_dataset_loader = IBMDatasetLoader(data_preparation_config)

    # selected_group_mode = ibm_dataset_loader.selected_group_mode

    # analyze_reconstruction_errors(ibm_dataset_loader, selected_group_mode, model_configs=model_configs, experiment_config=experiment_config)


if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()