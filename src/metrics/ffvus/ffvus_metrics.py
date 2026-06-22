from abc import ABC
from typing import Optional

import numpy as np
import torch

from metrics.Metric import Metric
from metrics.ffvus.vus_torch import VUSTorch


class FFVUS(Metric):
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self, slope) -> None:
        self.slope: Optional[int] = slope

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        # assert y_true.ndim == 2
        # assert y_score.ndim == 2
        # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
        # result = auc(fpr, tpr)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vus = VUSTorch(slope_size=self.slope, device=device)
        value_data, timing = vus.compute(torch.from_numpy(y_true).to(device), torch.from_numpy(np.round(y_score, decimals=2)).to(device))
        result_dict = dict(value=value_data[0].item(), computation_time=timing)
        return result_dict

    def supports_continuous_scorings(self) -> bool:
        return True
    @property
    def name(self) -> str:
        return f'FFVUS_PR'