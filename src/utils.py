import json
import random
from pathlib import Path

import torch
from scipy.linalg import pinv
from scipy.spatial.distance import mahalanobis
from scipy.stats import iqr
from tqdm import tqdm
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def get_project_root() -> Path:
    return Path(__file__).parent.parent

class ProgressBar:
  def __init__(self):
      self.progress_bar = None

  def __call__(self, current_bytes, total_bytes, width):
      current_mb = round(current_bytes / 1024 ** 2, 1)
      total_mb = round(total_bytes / 1024 ** 2, 1)
      if self.progress_bar is None:
          self.progress_bar = tqdm(total=total_mb, desc="MB")
      delta_mb = current_mb - self.progress_bar.n
      self.progress_bar.update(delta_mb)

# def build_adjacency_matrix_no_group(node_ids):
#     matrix = []
#     for index_i, i in enumerate(node_ids):
#         i_distance = np.zeros(len(node_ids))
#         i_tokens = np.array(i.split('_'))
#         i_center, i_communication_type, i_component, i_method, i_endpoint = i_tokens[0],i_tokens[1], i_tokens[2], i_tokens[3], i_tokens[5]
#         for index_j, j in enumerate(node_ids):
#             j_tokens = np.array(j.split('_'))
#             j_center,j_communication_type, j_component, j_method, j_endpoint = j_tokens[0], j_tokens[1], j_tokens[2], j_tokens[3], j_tokens[5]
#
#             # correlated = np.any([i_tokens[token_index] == j_tokens[token_index] for token_index in range(len(i_tokens))])
#             # correlated = i_tokens[-1] == j_tokens[-1]
#             # if i_component == j_component:
#             #     i_distance[index_j] = 1
#             if i_endpoint == j_endpoint :
#                 if i_component == j_component:
#                     if i_method == j_method:
#                         i_distance[index_j] = 1
#
#             if index_i == index_j:
#                 i_distance[index_j] = 0
#         matrix.append(i_distance)
#     return np.array(matrix)

def build_adjacency_matrix_no_group(node_ids):
    num_nodes = len(node_ids)
    matrix = []
    for index_i, i in enumerate(node_ids):
        i_distance = np.zeros(len(node_ids))
        i_tokens = np.array(i.split('_'))
        i_center, i_communication_type, i_component, i_method, i_endpoint = i_tokens[0],i_tokens[1], i_tokens[2], i_tokens[3], i_tokens[5]
        for index_j, j in enumerate(node_ids):
            j_tokens = np.array(j.split('_'))
            j_center,j_communication_type, j_component, j_method, j_endpoint = j_tokens[0], j_tokens[1], j_tokens[2], j_tokens[3], j_tokens[5]

            # correlated = np.any([i_tokens[token_index] == j_tokens[token_index] for token_index in range(len(i_tokens))])
            # correlated = i_tokens[-1] == j_tokens[-1]
            # if i_component == j_component:
            #     i_distance[index_j] = 1
            if (i_endpoint == j_endpoint) and (i_component == j_component):
                # if i_component == j_component:
                if i_method == j_method:
                    # if i_communication_type == j_communication_type:
                    i_distance[index_j] = 0.8
                    # else:
                    #     i_distance[index_j] = 0.6
                else:
                    if i_communication_type == j_communication_type:
                        i_distance[index_j] = 0.6
                    else:
                        i_distance[index_j] = 0.2

            if index_i == index_j:
                i_distance[index_j] = 0
        matrix.append(i_distance)

    matrix = np.array(matrix)

    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Use i < j to avoid duplicates in undirected graph
            if matrix[i, j] > 0:
                edges.append([i, j, matrix[i, j]])
                edges.append([j, i, matrix[i, j]])
    edges = np.array(edges)
    return edges
    # return edges[:,0], edges[:,1], edges[:,2]

def clear_folder(folder_dir):
    import os
    import glob

    print('Clearing folder {}'.format(folder_dir))

    if os.path.exists(folder_dir):
        files = glob.glob(folder_dir)
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
                print(f'Removed {f}')

def get_full_err_scores(reconstruction_errors_all_nodes):
    """Get stacked array of error scores for each feature by applying the
    `get_err_scores` function on every slice of the `test_result` tensor.
    """
    all_scores = [
        get_err_scores(reconstruction_errors_all_nodes[:, :, i])
        for i in range(reconstruction_errors_all_nodes.shape[-1])
    ]
    return np.stack(all_scores, axis = -1)
def get_err_scores(reconstruction_errors):
    """
    Calculate the error scores, normalised by the median and interquartile range.

    Parameters
    ----------
    test_result_list (list):
        List containing two lists of predicted and ground truth values
    smoothen_error (bool):
        A boolean value indicating whether error smoothing should be applied or not

    Returns
    -------
    err_scores (np.ndarray):
        An array of error scores
    """
    # test_predict, test_ground = test_result_list

    n_err_mid, n_err_iqr = get_err_median_and_iqr(reconstruction_errors)

    # test_delta = np.abs(
    #     np.subtract(
    #         np.array(test_predict).astype(np.float64),
    #         np.array(test_ground).astype(np.float64),
    #     )
    # )
    test_delta = reconstruction_errors.astype(np.float64)
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    return err_scores

def get_err_median_and_iqr(reconstruction_errors):
    # np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    #
    return np.median(reconstruction_errors), iqr(reconstruction_errors)

def calculate_mahalanobis_distance(reconstruction_error_raw):
    print('Calculating Mahalanobis Distance')
    num_timestamps = reconstruction_error_raw.shape[0]
    flattened = reconstruction_error_raw.reshape(num_timestamps, -1)
    mean_vec = np.mean(flattened, axis=0)
    cov_matrix = np.cov(flattened, rowvar=False)
    inv_cov_matrix = pinv(cov_matrix)

    # Compute Mahalanobis distance at each timestamp
    mahalanobis_distances = np.array([
        mahalanobis(flattened[t], mean_vec, inv_cov_matrix)
        for t in range(num_timestamps)
    ])
    return  mahalanobis_distances
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)