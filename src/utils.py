import json
import random
from pathlib import Path

import torch
from scipy.linalg import pinv, pinvh, inv, sqrtm, cho_factor, cho_solve, LinAlgError
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

def results_root_dir(model_save_path, log_transform=False):
    """Root directory of the trained models / results.  Runs on log1p-transformed data
    live in a sibling directory (e.g. trained_models_log1p/) so they never mix with
    runs on the raw counts."""
    if not log_transform:
        return model_save_path
    return model_save_path.rstrip('/') + '_log1p/'

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

SCORE_NORMALIZATIONS = ('global', 'per_node')

def score_normalizations(evaluation_config):
    """
    Score normalisations to run, from evaluation.score_normalizations (a list, e.g. ['per_node', 'global']),
    or the single evaluation.score_normalization of older configs; 'per_node' when neither is set.
    """
    if evaluation_config.get('score_normalizations', None) is not None:
        normalizations = list(evaluation_config.score_normalizations)
    else:
        normalizations = [evaluation_config.get('score_normalization', 'per_node')]
    for normalization in normalizations:
        assert normalization in SCORE_NORMALIZATIONS, f'Unknown score normalization: {normalization}'
    assert normalizations, 'score_normalizations is empty'
    return normalizations

def scoring_result_file_name(model, kind, score_normalization):
    """
    File name of a grid-search output, tagged with the score normalisation so the
    results of 'global' and 'per_node' runs do not overwrite each other.
    kind: 'grid_search', 'predictions_for_assembles' or 'likelihood_top_k_contribution'.
    """
    prefix = '' if kind == 'likelihood_top_k_contribution' else f'{model}_'
    return f'{prefix}{kind}_{score_normalization}.csv'

def get_full_err_scores(reconstruction_errors_all_nodes, normalization='per_node', reference_errors=None):
    """Get stacked array of error scores [T, N, F] by applying the `get_err_scores`
    function on the error series of the `test_result` tensor.
      • 'per_node': each node (and feature) is normalised with its own median / IQR
                    over time, as GDN does per sensor.
      • 'global'  : one median / IQR per feature over all nodes and time steps,
                    so the ranking across nodes is that of the raw errors.
    The median / IQR come from reference_errors [T_ref, N, F] when given (e.g. the
    train windows without anomalies), otherwise from the errors themselves.
    """
    assert normalization in SCORE_NORMALIZATIONS, f'Unknown score normalization: {normalization}'
    errors = reconstruction_errors_all_nodes
    reference = errors if reference_errors is None else reference_errors
    T, N, F = errors.shape
    if normalization == 'global':
        all_scores = [
            get_err_scores(errors[:, :, f].reshape(-1), reference[:, :, f].reshape(-1)).reshape(T, N)
            for f in range(F)
        ]
        return np.stack(all_scores, axis = -1)
    # Statistics over time (axis 0) give one median / IQR per (node, feature)
    return get_err_scores(errors, reference)

def get_err_scores(reconstruction_errors, reference_errors=None):
    """
    Calculate the error scores, normalised per sensor (column) by the median and
    interquartile range of its errors over time, as in GDN (Deng & Hooi, 2021).
    The statistics are taken from reference_errors when given.
    """
    reference = reconstruction_errors if reference_errors is None else reference_errors
    n_err_mid, n_err_iqr = get_err_median_and_iqr(reference)

    test_delta = reconstruction_errors.astype(np.float64)
    epsilon = 1e-6

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    return err_scores

def get_err_median_and_iqr(reconstruction_errors):
    # np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    # Statistics over time (axis 0), one median / IQR per sensor
    return np.median(reconstruction_errors, axis=0), iqr(reconstruction_errors, axis=0)

class MahalanobisScorer:
    """
    Mahalanobis-type anomaly score of per-timestamp error vectors.  The
    statistics are fitted on reference errors (e.g. train / validation) and
    applied to new errors (e.g. test).

    covariance:
      • 'empirical'   : sample covariance, pseudo-inverted.
      • 'ledoit_wolf' : Ledoit-Wolf shrinkage of the sample covariance, which
                        keeps it well-conditioned when D is large w.r.t. T.
      • 'pca'         : PCA monitoring statistics.  Hotelling T² (Mahalanobis
                        distance in the principal subspace keeping pca_variance
                        of the variance) plus SPE/Q (squared residual outside it),
                        each divided by its pca_quantile on the reference errors.
    min_variance: dimensions whose reference variance is <= this are dropped
                  first (they make Σ near-singular and dominate the distance).
    """

    COVARIANCES = ('empirical', 'ledoit_wolf', 'pca')

    def __init__(self, covariance='empirical', min_variance=0.0,
                 pca_variance=0.95, pca_quantile=0.99, batch_size=4096):
        assert covariance in self.COVARIANCES, f'Unknown covariance estimator: {covariance}'
        self.covariance = covariance
        self.min_variance = min_variance
        self.pca_variance = pca_variance
        self.pca_quantile = pca_quantile
        self.batch_size = batch_size

    def settings(self):
        """Short description of the configuration, used to tag cached results."""
        s = f'{self.covariance}|min_variance={self.min_variance}'
        if self.covariance == 'pca':
            s += f'|pca_variance={self.pca_variance}|pca_quantile={self.pca_quantile}'
        return s

    def fit(self, reference_errors):
        flattened = reference_errors.reshape(reference_errors.shape[0], -1).astype(np.float64)
        variance = flattened.var(axis=0)
        self.kept_dims = np.flatnonzero(variance > self.min_variance)
        assert self.kept_dims.size > 0, f'No dimension has variance > {self.min_variance}'
        flattened = flattened[:, self.kept_dims]
        self.mean_vec = flattened.mean(axis=0)

        if self.covariance == 'ledoit_wolf':
            from sklearn.covariance import LedoitWolf
            # store_precision=False: sklearn would otherwise pinvh Σ itself, and the result is unused
            cov_matrix = LedoitWolf(assume_centered=False, store_precision=False).fit(flattened).covariance_
        else:
            cov_matrix = np.atleast_2d(np.cov(flattened, rowvar=False))

        if self.covariance == 'pca':
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = np.clip(eigvals[order], 0.0, None), eigvecs[:, order]
            explained = np.cumsum(eigvals) / eigvals.sum()
            num_components = int(np.searchsorted(explained, self.pca_variance) + 1)
            self.components = eigvecs[:, :num_components]                 # [D, k]
            self.component_var = np.maximum(eigvals[:num_components], 1e-12)
            # Empirical control limits on the reference errors
            t2, q, _ = self._pca_terms(flattened)
            self.t2_limit = max(np.quantile(t2, self.pca_quantile), 1e-12)
            self.q_limit = max(np.quantile(q, self.pca_quantile), 1e-12)
        else:
            self.inv_cov_matrix = None
            if self.covariance == 'ledoit_wolf':
                # The shrunk Σ is positive definite: a Cholesky inverse is ~40x faster than pinvh
                try:
                    self.inv_cov_matrix = cho_solve(cho_factor(cov_matrix), np.eye(len(cov_matrix)))
                except LinAlgError:   # no shrinkage and singular Σ
                    pass
            if self.inv_cov_matrix is None:
                # Σ is symmetric: an eigendecomposition (pinvh) is ~3x faster than the SVD used by pinv
                self.inv_cov_matrix = pinvh(cov_matrix)
            self.inv_cov_diag = np.diag(self.inv_cov_matrix)
        return self

    def _pca_terms(self, rows):
        """T², Q and the per-dimension residual of rows [B, D] (kept dimensions only)."""
        delta = rows - self.mean_vec
        scores = delta @ self.components                                  # [B, k]
        t2 = (scores ** 2 / self.component_var).sum(axis=1)
        residual = delta - scores @ self.components.T
        return t2, (residual ** 2).sum(axis=1), residual

    def score(self, errors, top_k=None):
        """
        Distance of every timestamp of errors [T, ...] (sqrt of D², or of the
        combined T²/Q index for 'pca').  With top_k, also returns the indices of
        the top_k dimensions with the largest contribution, in ascending order.
        """
        flattened = errors.reshape(errors.shape[0], -1)
        num_timestamps = flattened.shape[0]
        distances = np.empty(num_timestamps)
        top_contributions = np.empty((num_timestamps, top_k), dtype=np.int64) if top_k else None
        for start in range(0, num_timestamps, self.batch_size):
            # Row-major: errors from a DataFrame are column-major, which makes the per-row
            # reductions below (einsum, argpartition along axis 1) strided and ~20x slower
            rows = flattened[start:start + self.batch_size, self.kept_dims].astype(np.float64, order='C')
            if self.covariance == 'pca':
                t2, q, residual = self._pca_terms(rows)
                D2 = t2 / self.t2_limit + q / self.q_limit
                contrib = residual ** 2 / self.q_limit                     # Q contribution per dimension
            else:
                delta = rows - self.mean_vec
                D2 = np.einsum('ij,ij->i', delta @ self.inv_cov_matrix, delta)
                contrib = (delta ** 2) * self.inv_cov_diag                 # diagonal contribution
            distances[start:start + self.batch_size] = np.sqrt(np.maximum(D2, 0.0))   # clip round-off negatives
            if top_k:
                k = min(top_k, contrib.shape[1])
                idx = np.argpartition(contrib, -k, axis=1)[:, -k:]
                order = np.argsort(np.take_along_axis(contrib, idx, axis=1), axis=1)
                top = self.kept_dims[np.take_along_axis(idx, order, axis=1)]   # map back to original dims
                top_contributions[start:start + self.batch_size] = np.pad(top, ((0, 0), (top_k - k, 0)), mode='edge')
        return distances, top_contributions

def calculate_mahalanobis_distance(reconstruction_error_raw, reference_errors=None, **scorer_args):
    """
    Mahalanobis distance of each timestamp's error vector.
    The statistics are fitted on reference_errors (e.g. the training-set errors);
    when it is None they are fitted on reconstruction_error_raw itself.
    scorer_args configure the MahalanobisScorer (covariance estimator, min_variance, ...).
    """
    print('Calculating Mahalanobis Distance')
    reference = reconstruction_error_raw if reference_errors is None else reference_errors
    scorer = MahalanobisScorer(**scorer_args).fit(reference)

    # Compute Mahalanobis distance at each timestamp
    mahalanobis_distances, _ = scorer.score(reconstruction_error_raw)
    return  mahalanobis_distances

def has_is_nan_mask(is_nan_mask):
    """True when is_nan results are available (null_padding_target) and can mask the errors."""
    return is_nan_mask is not None and is_nan_mask.ndim == 3

def apply_is_nan_mask(reconstruction_error_raw, is_nan_mask):
    """Copy of the errors with positions predicted as missing (is_nan prediction <= 0.6) set to 0."""
    masked = reconstruction_error_raw.copy()
    if has_is_nan_mask(is_nan_mask):
        not_nan_prob = is_nan_mask[0, :,:].reshape(masked.shape[0], -1)

        assert masked.shape[0] == not_nan_prob.shape[0]
        masked[not_nan_prob <= 0.6] = 0.0
    return masked

def calculate_mahalanobis_distance_with_is_nan_mask(reconstruction_error_raw, is_nan_mask, top_k,
                                                    reference_errors=None, reference_is_nan_mask=None,
                                                    **scorer_args):
    """
    Mahalanobis distance after masking predicted-missing positions, plus the
    top-k contributing dimensions per timestamp.  The inputs are not modified.
    μ and Σ are fitted on the masked reference_errors when given, otherwise on
    the masked reconstruction_error_raw itself.  scorer_args configure the MahalanobisScorer.
    """
    if has_is_nan_mask(is_nan_mask):
        print('Calculating Mahalanobis Distance with is_nan_mask')
    masked_errors = apply_is_nan_mask(reconstruction_error_raw, is_nan_mask)
    reference = masked_errors if reference_errors is None \
        else apply_is_nan_mask(reference_errors, reference_is_nan_mask)
    scorer = MahalanobisScorer(**scorer_args).fit(reference)

    k = top_k
    mahalanobis_distances, mahalanobis_top_contributions = scorer.score(masked_errors, top_k=k)

    assert mahalanobis_distances.shape[0] == mahalanobis_top_contributions.shape[0]
    assert k == mahalanobis_top_contributions.shape[1]
    return mahalanobis_distances, mahalanobis_top_contributions

def refine_reconstruction_error_with_is_nan_mask(reconstruction_error_raw, is_nan_mask):
    if is_nan_mask is not None:
        refined = reconstruction_error_raw.copy()
        not_nan_prob = is_nan_mask.reshape(refined.shape[0], -1)

        assert refined.shape[0] == not_nan_prob.shape[0]
        refined[not_nan_prob <= 0.6] = 0.0
        print('Calculating Mahalanobis Distance with is_nan_mask')
        return refined
    return reconstruction_error_raw
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)