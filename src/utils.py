import json
from pathlib import Path

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


def build_adjacency_matrix_no_group(node_ids):
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
            if i_endpoint == j_endpoint :
                if i_component == j_component:
                    if i_method == j_method:
                        i_distance[index_j] = 1

            if index_i == index_j:
                i_distance[index_j] = 0
        matrix.append(i_distance)
    return np.array(matrix)

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