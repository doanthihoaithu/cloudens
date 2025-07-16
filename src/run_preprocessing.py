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
    multiple_preprocessing = data_preparation_config.multiple_preprocessing
    if not multiple_preprocessing:
        ibm_dataset_loader = IBMDatasetLoader(data_preparation_config)
    else:
        http_codes = data_preparation_config.feature_subsets.http_codes
        aggregations = data_preparation_config.feature_subsets.aggregations
        code_aggr_combinations = list(itertools.product(http_codes, aggregations))
        for index, (http_code, aggregation) in (pbar := tqdm(enumerate(code_aggr_combinations),
                                                    total=len(code_aggr_combinations))):
            pbar.set_description(f'Preprocessing on feature subset {index}: {http_code}-{aggregation}')
            copy_config = data_preparation_config.copy()
            copy_config.features_prep.filter.http_codes = [http_code]
            copy_config.features_prep.filter.aggregations = [aggregation]
            ibm_dataset_loader = IBMDatasetLoader(copy_config)

if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')
    main()