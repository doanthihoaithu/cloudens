import itertools
import os
import random
import sys
import hydra
from omegaconf import DictConfig
import logging

import numpy as np
import torch
from tqdm import tqdm

from ibm_dataset_loader import IBMDatasetLoader
from plotting_module import generate_latex_training_inference_time, generate_latex_full_table, \
    generate_latex_ensemble_table
from utils import set_random_seed, get_project_root

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to preprocess the pivoted dataset, dividing into different feature subsets.

    Steps:
    1. Load and preprocess the data.
    2. Save processed data to disk.

    Parameters:
    - cfg: DictConfig, configuration object containing paths, parameters, and settings.
    """
    log.info("Starting main function")

    random_seed = cfg.modeling.random_seed
    set_random_seed(random_seed)

    data_preparation_config = cfg.data_preparation_pipeline
    print(data_preparation_config)

    # model: ['GRU', 'A3TGCN']
    # slide_win: ${evaluation.slide_win}
    # subset:
    #   http_codes: ['5xx']
    #   aggregations: ['count', 'avg','min','max']
    # null_padding_feature: ${evaluation.null_padding_feature}
    # null_padding_target: ${evaluation.null_padding_target}
    plotting_config = cfg.plotting
    window_size = plotting_config.slide_win
    combined_figure_save_dir = os.path.join(get_project_root(), plotting_config.output_dir, f'window_{window_size}')
    grid_search_combine_file_path = os.path.join(combined_figure_save_dir, f'combined_grid_search_results.csv')

    generate_latex_training_inference_time(grid_search_combine_file_path)
    generate_latex_full_table(grid_search_combine_file_path)


    experiment_config = cfg.evaluation
    ensemble_combine_dir = os.path.join(get_project_root(),
                                        experiment_config.model_save_path,
                                        f'window_{window_size}',
                                        'ensemble')
    os.makedirs(ensemble_combine_dir, exist_ok=True)
    generate_latex_ensemble_table(ensemble_combine_dir)
    # generate_figure_for_fpr(plotting_config)





if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()