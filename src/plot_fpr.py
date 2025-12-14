import itertools
import random
import sys
import hydra
from omegaconf import DictConfig
import logging

import numpy as np
import torch
from tqdm import tqdm

from ibm_dataset_loader import IBMDatasetLoader
from plotting_module import generate_figure_for_fdr
from utils import set_random_seed

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

    plotting_config = cfg.plotting

    # model: ['GRU', 'A3TGCN']
    # slide_win: ${evaluation.slide_win}
    # subset:
    #   http_codes: ['5xx']
    #   aggregations: ['count', 'avg','min','max']
    # null_padding_feature: ${evaluation.null_padding_feature}
    # null_padding_target: ${evaluation.null_padding_target}

    generate_figure_for_fdr(plotting_config)





if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()