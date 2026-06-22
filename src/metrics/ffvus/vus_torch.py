"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""
import functools
import time

import torch
import numpy as np
from tqdm import tqdm
import psutil
import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns

def time_it(func):
    """Wrapper to measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper
class VUSTorch():
    def __init__(
            self, 
            slope_size=100, 
            step=1, 
            zita=(1/torch.sqrt(torch.tensor(2))), 
            global_mask=True,
            existence=True,
            conf_matrix='dynamic_plus',
            device=None,
        ):
        """
        Initialize the torch version of the VUS metric.

        Args:
            TODO: Write the arguments description when done
        """
        if slope_size < 0 or step < 0:
            raise ValueError(f"Error with the slope_size {slope_size} or step {step}. They should be positive values.")
        if step == 0 and slope_size > 0:
            raise ValueError(f"Error with step {step} and slope_size {slope_size}. Step can't be 0 with non-zero slope_size.")
        if (step > 0 and slope_size > 0) and (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 and zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # if self.device == 'cpu':
        #     print('You are using the GPU version of VUS on a CPU. Performance will be limited!')

        self.slope_size, self.step, self.zita, self.global_mask = slope_size, step, zita, global_mask    
        self.existence = False if existence == 'None'else existence
        
        self.slope_values = torch.arange(
            start=0, 
            end=self.slope_size + 1, 
            step=self.step, 
            device=self.device, 
            dtype=torch.int16 if self.slope_size < 32000 else torch.int32 
        ) if self.slope_size > 0 else torch.tensor([0], device=self.device)
        self.n_slopes = self.slope_values.shape[0]

        conf_matrix_args = ['dynamic', 'dynamic_plus']
        if conf_matrix in conf_matrix_args:
            self.conf_matrix_mode = conf_matrix
        else:
            raise ValueError(f"Unknown argument for conf_matrix: {conf_matrix}. 'conf_matrix' should be one of {conf_matrix_args}")

        # self.warmup()

    def warmup(self):
        '''
        Torch first computation is slower than the rest because of JIT, etc. This function does a dummy computation
        to remove this overhead from the first actual computation.
        '''
        label = torch.randint(0, 2, (20,), dtype=torch.int8, device=self.device)
        score = torch.rand(20, device=self.device)
        self.compute(label, score)


    def update_max_memory_tokens(self, divider=50):
        """
        Update self.max_memory_tokens according to the currently available memory.

        Changing the divider makes a little difference
       """
        if self.device == 'cuda':
            available_memory, _ = torch.cuda.mem_get_info()
            self.max_memory_tokens = available_memory / divider
        else:
            available_memory = psutil.virtual_memory().available
            self.max_memory_tokens = available_memory / divider

    @time_it
    def compute(self, label, score):
        """
        The main computing function of the metric
        
        TODO: There is still some error on MITDB two first time series, CPU version too
        """
        self.update_max_memory_tokens()
        fp = tp = positives = anomalies_found = total_anomalies = time_sm = time_slopes = time_existence = time_confusion = time_pos = 0

        ((_), (start_with_edges, end_with_edges)), time_anomalies_coord = self.get_anomalies_coordinates(label)
        safe_mask, time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=self.global_mask)
        (thresholds), time_thresholds = self.get_unique_thresholds(score)

        # Apply global mask
        if self.global_mask:
            chunks = self.process_in_chunks(score[~safe_mask])
            extra_fp = torch.zeros((thresholds.shape[0]), device=self.device)

            # for chunk in tqdm(chunks, desc='Computing extra FPs', leave=False):
            #     sm, extra_time_sm = self.get_score_mask(chunk, thresholds)
            #     extra_fp += sm.sum(axis=1)
            #     time_sm += extra_time_sm

            for chunk in chunks:
                sm, extra_time_sm = self.get_score_mask(chunk, thresholds)
                extra_fp += sm.sum(axis=1)
                time_sm += extra_time_sm

            label, score = label[safe_mask], score[safe_mask]
            ((_), (start_with_edges, end_with_edges)), extra_time_anomalies_coord = self.get_anomalies_coordinates(label)
            safe_mask, extra_time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges)
            time_anomalies_coord += extra_time_anomalies_coord
            time_safe_mask += extra_time_safe_mask

        # Number of chunks required to fit into memory
        sloped_label_mem_size = self.n_slopes * len(label) * 4
        n_splits = np.ceil(sloped_label_mem_size / self.max_memory_tokens).astype(int)
        split_points = self.find_safe_splits(label, safe_mask, n_splits)

        # Computation in chunks
        prev_split = 0
        for curr_split in tqdm(split_points, desc='In distance from anomaly', disable=(n_splits <= 10), leave=False):
            label_c = label[prev_split:curr_split]
            score_c = score[prev_split:curr_split]
            safe_mask_c = safe_mask[prev_split:curr_split]
            
            # Preprocessing
            ((_), (start_with_edges_c, end_with_edges_c)), chunk_time_anomalies_coord = self.get_anomalies_coordinates(label_c)
            pos, chunk_time_pos = self.distance_from_anomaly(label_c, start_with_edges_c, end_with_edges_c)
            sm, chunk_time_sm = self.get_score_mask(score_c, thresholds)

            # Main computation
            labels_c, chunk_time_slope = self.add_slopes(label_c, pos)
            (anomalies_found_c, total_anomalies_c), chunk_time_existence = self.compute_existence(labels_c, sm, safe_mask_c, normalize=False)
            (fp_c, fn_c, tp_c, positives_c, neg_c, fpr_c), chunk_time_conf = self.compute_confusion_matrix(labels_c, sm)

            # Accumulate results and timing from each chunk
            tp += tp_c
            fp += fp_c
            positives += positives_c
            anomalies_found += anomalies_found_c
            total_anomalies += total_anomalies_c
            time_anomalies_coord += chunk_time_anomalies_coord
            time_pos += chunk_time_pos
            time_sm += chunk_time_sm
            time_slopes += chunk_time_slope
            time_existence += chunk_time_existence
            time_confusion += chunk_time_conf

            prev_split = curr_split

        # Combine existence of all chunks
        existence = anomalies_found / total_anomalies if self.existence else torch.ones((self.n_slopes, thresholds.shape[0]), device=self.device)

        if self.global_mask:
            fp += extra_fp

        # After chunking
        (precision, recall), time_pr_rec = self.precision_recall_curve(tp, fp, positives, existence)
        vus_pr, time_integral = self.auc(recall, precision)

        time_analysis = {
            "Anomaly coordinates time": time_anomalies_coord,
            "Safe mask time": time_safe_mask,
            "Thresholds time": time_thresholds,
            "Score mask time": time_sm,
            "Position time": time_pos,
            "Slopes time": time_slopes,
            "Existence time": time_existence,
            "Confusion matrix time": time_confusion,
            "Precision recall curve time": time_pr_rec,
            "Integral time": time_integral,
        }

        return vus_pr, time_analysis

    def process_in_chunks(self, data, divider=100):
        '''
        Divider is used to simulate the number of thresholds, however some times
        we process single dimension arrays, as in unique_thresholds
        '''
        if len(data) == 0:
            return
        data_mem_size = data.nbytes * divider  # Around 100 thresholds 
        n_splits = np.ceil(data_mem_size / self.max_memory_tokens).astype(int)
        chunk_size = len(data) // n_splits

        for i in range(n_splits):
            start = i * chunk_size
            end = len(data) if i == n_splits - 1 else (i + 1) * chunk_size
            chunk = data[start:end]
            yield chunk

    def find_safe_splits(self, label, safe_mask, n_splits):
        """
        Finds approximately evenly spaced safe split points where pos is a local maximum
        and far enough from anomalies.
        """
        length = torch.tensor(label.shape[0], device=self.device)
        valid_splits_mask = ~safe_mask

        step = max(1, valid_splits_mask.sum().item() // (n_splits * 10))
        valid_splits = torch.nonzero(valid_splits_mask)[::step].squeeze(1)

        if valid_splits.numel() == 0 or n_splits < 1:
            return torch.tensor([length], device=self.device)

        ideal_splits = torch.linspace(0, length - 1, steps=n_splits + 1, device=self.device)[1:-1]        

        dists = torch.abs(valid_splits[:, None] - ideal_splits[None, :])
        min_indices = dists.argmin(dim=0)
        selected_splits = valid_splits[min_indices]
        selected_splits = torch.unique(selected_splits.sort().values, sorted=True)

        # Final safety check
        assert torch.all(label[selected_splits] != 1), "Some selected splits fall inside anomalies!"
        assert torch.all(safe_mask[selected_splits] == 0), "Some splits are too close to anomalies!"

        total_splits = torch.cat((selected_splits, length[None]), dim=0)
        return total_splits
    
    @time_it
    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    @time_it
    def get_unique_thresholds(self, score, compute_in_chunks=True, max_thresholds=200):
        if not compute_in_chunks:
            sorted_thresholds, _ = torch.sort(torch.unique(score), descending=True)
        else:
            chunks = self.process_in_chunks(score, divider=5)
            all_unique = torch.empty(0, device=score.device, dtype=score.dtype)
            for chunk in chunks:
                chunk_unique = torch.unique(chunk)
                all_unique = torch.cat((all_unique, chunk_unique))
            thresholds = torch.unique(all_unique)
            sorted_thresholds, _ = torch.sort(thresholds, descending=True)

            # Check that the two threshold arrays are equal
            # sorted_thresholds_whole, _ = torch.sort(torch.unique(score), descending=True)
            # if not torch.equal(sorted_thresholds, sorted_thresholds_whole):
            #     raise ValueError("Threshold arrays are not equal: sorted_thresholds and sorted_thresholds_whole differ.")

        if len(thresholds) > max_thresholds:
            warnings.warn(
                f"Number of unique thresholds is {len(thresholds)}, which may slow down computation. "
                "Consider rounding your scores to 2 decimals."
            )


        return sorted_thresholds
    
    @time_it
    def get_anomalies_coordinates(self, label: torch.Tensor):
        """
        Return the starting and ending points of all anomalies in label,
        both with and without edge inclusion.

        Args:
            label (torch.tensor): vector of 1s and 0s
        Returns:
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges))
        """
        device = label.device
        diff = torch.diff(label)
        start_no_edges = torch.where(diff == 1)[0] + 1
        end_no_edges = torch.where(diff == -1)[0]
        
        start_with_edges = torch.cat((torch.tensor([0], device=device), start_no_edges)) if label[0] else start_no_edges
        end_with_edges = torch.cat((end_no_edges, torch.tensor([len(label) - 1], device=device))) if label[-1] else end_no_edges

        if start_with_edges.shape != end_with_edges.shape:
            raise ValueError(f'The number of start and end points of anomalies does not match, {start_with_edges} != {end_with_edges}')

        return (start_no_edges, end_no_edges), (start_with_edges, end_with_edges)
    
    @time_it
    def create_safe_mask(self, label: torch.Tensor, start_points: torch.Tensor, end_points: torch.Tensor, extra_safe: bool = False) -> torch.Tensor:
        """
        A safe mask is a mask of the label that every anomaly is one point bigger (left and right)
        than the bigger slope. This allows us to mask the label safely, without changing anything in the implementation.
        """
        length = torch.tensor(label.shape[0], device=self.device)
        mask = torch.zeros(length, dtype=torch.int8, device=self.device)

        safe_extension = self.slope_size + 1 + int(extra_safe)
        
        start_safe_points = torch.clamp(start_points - safe_extension, min=0)
        end_safe_points = torch.clamp(end_points + safe_extension + 1, max=length - 1)
        
        add_start = torch.ones_like(start_safe_points, dtype=torch.int8, device=self.device)
        add_end = torch.ones_like(end_safe_points, dtype=torch.int8, device=self.device).mul(-1)

        mask.index_add_(0, start_safe_points, add_start)
        mask.index_add_(0, end_safe_points, add_end)

        mask = mask.cumsum(dim=0)
        mask = mask > 0
        mask[-1] = (end_safe_points[-1] == (length - 1)) if extra_safe else label[-1] # Why does it work?

        return mask
    
    @time_it
    def distance_from_anomaly(self, label: torch.Tensor, start_points: torch.Tensor, end_points: torch.Tensor, clip: bool = False) -> torch.Tensor:
        """
        Computes distance to closest anomaly boundary for each point. Uses PyTorch for GPU compatibility.
        """
        device = label.device
        length = label.size(0)

        pos = torch.full((length,), float('inf'), device=device)
        if start_points.numel() == 0 and end_points.numel() == 0:
            return pos

        anomaly_boundaries = torch.cat([start_points, end_points])
        indices = torch.arange(length, device=device)[:, None]

        pos_mem_size = len(anomaly_boundaries) * length * pos.element_size()
        if pos_mem_size > self.max_memory_tokens:
            n_chunks = max(1, pos_mem_size // self.max_memory_tokens)
            chunk_size = int(max(1, len(anomaly_boundaries) // n_chunks))

            for chunk in tqdm(anomaly_boundaries.split(chunk_size), desc='In distance from anomaly', disable=False if n_chunks > 10000 else True):
                curr_distances = torch.abs(indices - chunk[None, :])
                curr_min_distances = curr_distances.min(dim=1).values
                pos = torch.minimum(pos, curr_min_distances)
        else:
            distances = torch.abs(indices - anomaly_boundaries)
            pos = torch.min(distances, dim=1).values
        
        if clip:
            pos = pos.clone()
            pos[label.bool()] = 0
            pos = torch.clamp(pos, max=self.slope_size)

        return pos
    
    @time_it
    def add_slopes(self, label: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Computes slope transformations for multiple slope values.

        Args:
            label: Binary tensor indicating anomaly regions (shape: [T], dtype: float).
            pos  : Distance to the nearest anomaly boundary (shape: [T]).

        Returns:
            Tensor of shape (n_slopes, T) with transformed slope values.
        """
        device = label.device
        T = label.size(0)

        valid_mask = (self.slope_size - pos) >= 0
        slope_values = self.slope_values[1:, None]
        pos = pos[None, :]
        f_pos = torch.zeros((self.n_slopes, T), device=device)
        f_pos[0] = label

        numerator = (1 - self.zita) * pos
        f_pos[1:, valid_mask] = 1 - (numerator[:, valid_mask] / slope_values)
        f_pos[1:][(slope_values - pos) < 0] = 0
        f_pos[1:, label.bool()] = 1

        return f_pos

    @time_it
    def compute_existence(self, labels: torch.Tensor, score_mask: torch.Tensor, safe_mask: torch.Tensor = None, normalize: bool = True, allow_recursion: bool = True) -> torch.Tensor:
        """
        PyTorch version of the existence matrix computation. This function uses an approximation fallback mechanism
        in case the memory requirements exceed the limit set by 'max_memory_tokens'.

        Args:
            labels: Tensor of shape [s, T], binary anomaly label one for each slope.
            score_mask: Tensor of shape [t, T], score mask over every threshold.

        Returns:
            Tensor of shape [s, t] representing existence score (fraction of anomalies found).
        """
        if not self.existence:
            return 1 if normalize else (1, 1)
        device = labels.device

        # Compute mask for relevant points
        if safe_mask is not None and not self.global_mask:
            labels = labels[:, safe_mask]
            score_mask = score_mask[:, safe_mask]
        
        if labels.shape[1] < 1:
            return 0 if normalize else (0, 0)

        # Check size and fallback if too big
        s, T = labels.shape
        t = score_mask.shape[0]
        if allow_recursion and (s * t * T > self.max_memory_tokens):
            (n_anomalies_found_0, total_anomalies_0), _ = self.compute_existence(labels[0:1], score_mask, normalize=False, allow_recursion=False)
            (n_anomalies_found_s, total_anomalies_s), _ = self.compute_existence(labels[-1:], score_mask, normalize=False, allow_recursion=False)
            
            interp_weights = torch.linspace(0, 1, steps=s, device=device).view(1, -1)
            n_anomalies_found = n_anomalies_found_0 + (n_anomalies_found_s - n_anomalies_found_0) * interp_weights.T
            total_anomalies = total_anomalies_0 + (total_anomalies_s - total_anomalies_0) * interp_weights.T
        else:
            # Normalize labels to binary (0 or 1)
            norm_labels = (labels > 0).int()

            # Compute step function (stairs)
            diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((norm_labels.size(0), 1), device=device))
            diff = torch.clamp(diff, min=0, max=1)
            stairs = torch.cumsum(diff, dim=1)
            labels_stairs = norm_labels * stairs

            # Multiply each score with every labeled anomaly step
            score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]

            # Cumulative max along the time axis
            cm = torch.cummax(score_hat, dim=2).values  # shape: [B, S, T']

            # Compute differences along time and normalize
            cm_diff = torch.diff(cm, dim=2)
            cm_diff_norm = torch.clamp(cm_diff - 1, min=0)

            # Total anomalies and missed anomalies
            total_anomalies = stairs[:, -1][:, None]
            final_anomalies_missed = total_anomalies - cm[:, :, -1]
            n_anomalies_not_found = torch.sum(cm_diff_norm, dim=2) + final_anomalies_missed
            n_anomalies_found = total_anomalies - n_anomalies_not_found

        if normalize:
            return n_anomalies_found / total_anomalies
        else:
            return n_anomalies_found, total_anomalies

    @time_it
    def compute_confusion_matrix(self, labels: torch.Tensor, sm: torch.Tensor):
        if self.conf_matrix_mode == "dynamic":
            conf_matrix = self.conf_matrix_dyn(labels, sm)
        else:
            conf_matrix = self.conf_matrix_dyn_plus(labels, sm)
        fn, tp, positives = conf_matrix

        fp = sm.sum(axis=1) - tp
        negatives = labels[0].shape[0] - positives
        fpr = fp / negatives

        return fp, fn, tp, positives, negatives, fpr
    

    def conf_matrix_dyn(self, labels: torch.Tensor, sm: torch.Tensor):
        """
        Computes confusion matrix using only the dynamic (non-always-one) region of the labels.
        
        Args:
            labels: Tensor of shape [2, T], where labels[0] is binary and labels[1] encodes slope info.
            sm: Tensor of shape [S, T], where S is the number of slopes (score masks).
        
        Returns:
            false_negatives: Tensor [S]
            true_positives: Tensor [S]
            positives: Tensor [S]
        """
        if not self.global_mask:
            mask = torch.where(labels[-1] > 0)[0]
            labels = labels[:, mask]
            sm = sm[:, mask]
        
        true_positives = torch.matmul(labels, sm.float().T)
        false_negatives = torch.matmul(labels[0], (~sm).float().T)[None, :]

        positives = ((true_positives + false_negatives) + labels[0].sum()).div(2.0)

        return false_negatives, true_positives, positives

    def conf_matrix_dyn_plus(self, labels: torch.Tensor, sm: torch.Tensor):
        """
        Optimized confusion matrix computation that avoids redundant work in always-one regions.
        
        Args:
            labels: Tensor of shape [2, T]
            sm: Tensor of shape [S, T]
        
        Returns:
            false_negatives: Tensor [S]
            true_positives: Tensor [S]
            positives: Tensor [S]
        """
        if not self.global_mask:
            slope_mask = torch.where(torch.logical_and(labels[-1] > 0, labels[-1] < 1))[0]
        else:
            slope_mask = torch.where(labels[-1] < 1)[0]
        label_as_mask = torch.where(labels[0])[0]

        initial_tps = sm[:, label_as_mask].sum(dim=1)
        slope_tps = torch.matmul(labels[:, slope_mask], sm[:, slope_mask].float().T)
        true_positives = initial_tps + slope_tps
        false_negatives = (~sm)[:, label_as_mask].sum(dim=1)
        positives = ((true_positives + false_negatives) + label_as_mask.size(0)).div(2)

        return false_negatives, true_positives, positives
    
    @time_it
    def precision_recall_curve(self, tp: torch.Tensor, fp: torch.Tensor, positives: torch.Tensor, existence: torch.Tensor = None):
        """
        Computes precision-recall curve points given TP, FP, positives, and optional existence weighting.
        
        Args:
            tp: Tensor of shape [n_slopes]
            fp: Tensor of shape [n_slopes]
            positives: Tensor of shape [n_slopes]
            existence: Optional tensor of shape [n_slopes, N] or None

        Returns:
            precision: Tensor of shape [n_slopes, N+1]
            recall: Tensor of shape [n_slopes, N+1]
        """
        device = tp.device

        ones = torch.ones(self.n_slopes, 1, device=device)
        zeros = torch.zeros(self.n_slopes, 1, device=device)
        precision = torch.cat([ones, tp / (tp + fp)], dim=1)
        recall = torch.cat([zeros, tp / positives], dim=1)

        recall[recall > 1] = 1
        if existence is not None:
            recall[:, 1:] = recall[:, 1:] * existence
            
        return precision, recall

    @time_it
    def auc(self, x, y):
        width_pr = x[:, 1:] - x[:, :-1]
        height_pr = y[:, 1:]
        auc = torch.sum(torch.multiply(width_pr, height_pr), axis=1)
        return auc.mean()