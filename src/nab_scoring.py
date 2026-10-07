import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def sigmoid(x):
    """
    Standard sigmoid function.

    Parameters:
    - x: Float, input value.

    Returns:
    - Float, sigmoid output between 0 and 1.
    """
    return 1 / (1 + math.exp(-x))

def scaledSigmoid(relativePositionInWindow):
    """
    Scaled sigmoid function for NAB scoring.
    Maps relative positions within anomaly windows to scores.

    Parameters:
    - relativePositionInWindow: Float, position relative to the anomaly window.

    Returns:
    - Float, scaled sigmoid score.
    """
    if relativePositionInWindow > 3.0:
        return -1.0  # FP well beyond window, assign -1
    else:
        return 2 * sigmoid(-5 * relativePositionInWindow) - 1.0

def _label_windows(df, true_col='true_anomaly'):
    """
    Windows (start, end) of the runs of true_col == 1, as index labels: start is the first
    timestamp of a run, end the first timestamp after it (the last timestamp of df when the
    run reaches the end).
    """
    labels = df[true_col].to_numpy()
    previous = np.concatenate([[0], labels[:-1]])
    starts = np.flatnonzero((labels == 1) & (previous == 0))
    ends = np.flatnonzero((labels == 0) & (previous == 1))
    end_labels = list(df.index[ends])
    if len(starts) > len(ends):   # a run reaching the end of the series
        end_labels.append(df.index[-1])
    return list(zip(df.index[starts], end_labels))


# Score of a detection at the very start of its window: TP scores are divided by it so that the
# earliest detection earns exactly reward_tp, as in NAB (Sweeper.calcSweepScore)
MAX_TP = scaledSigmoid(-1.0)


def calculate_relative_position(df, true_col='true_anomaly', pred_col='predicted_anomaly'):
    """
    Calculate relative positions of detected anomalies within true anomaly windows.

    Parameters:
    - df: DataFrame, containing true and predicted anomaly labels.
    - true_col: String, column name for true anomaly labels.
    - pred_col: String, column name for predicted anomaly labels.

    Returns:
    - DataFrame with an additional column 'relative_position'.
    """
    df['relative_position'] = 0.0
    for start, end in _label_windows(df, true_col):
        window_length = (end - start).total_seconds()
        predictions = df.loc[start:end, pred_col]
        if predictions.any():
            first_detection = predictions.idxmax()
            df.loc[first_detection, 'relative_position'] = -(end - first_detection).total_seconds() / window_length
    return df

def calculate_baseline_score(df, true_col='true_anomaly', penalty_fn=2.0):
    """
    Calculate the baseline NAB score (assuming no anomalies are detected).

    Parameters:
    - df: DataFrame, containing true anomaly labels.
    - true_col: String, column name for true anomaly labels.
    - penalty_fn: Float, penalty for a missed anomaly.

    Returns:
    - Float, baseline NAB score.
    """
    fn_count = len(_label_windows(df, true_col))  # Count of all anomaly windows (false negatives if undetected)
    return -penalty_fn * fn_count

def calculate_perfect_score(df, true_col='true_anomaly', reward_tp=1.0):
    """
    Calculate the perfect NAB score (assuming all anomalies are correctly detected).

    Parameters:
    - df: DataFrame, containing true anomaly labels.
    - true_col: String, column name for true anomaly labels.
    - reward_tp: Float, reward for correctly detecting an anomaly.

    Returns:
    - Float, perfect NAB score.
    - Integer, count of anomaly windows.
    """
    tp_count = len(_label_windows(df, true_col))
    return reward_tp * tp_count, tp_count

def normalize_nab_score(score, baseline_score, perfect_score):
    """
    Normalize the NAB score between the baseline and perfect scores.

    Parameters:
    - score: Float, raw NAB score.
    - baseline_score: Float, baseline NAB score.
    - perfect_score: Float, perfect NAB score.

    Returns:
    - Float, normalized NAB score (0 to 100).
    """
    if perfect_score == baseline_score:
        return 0
    return 100 * (score - baseline_score) / (perfect_score - baseline_score)

def calculate_nab_score_with_window_based_tp_fn(df, anomaly_windows_test, nab_scoring_profile, true_col='true_anomaly', pred_col='predicted_anomaly',
                                                reward_tp=1.0, penalty_fp=0.11, penalty_fn=1.0):
    """
    Calculate NAB score with true positive and false negative windows.

    Parameters:
    - df: DataFrame, containing true and predicted anomaly labels.
    - anomaly_windows_test: DataFrame, containing ground truth anomaly windows.
    - nab_scoring_profile: String, scoring profile ("standard" or "reward_fn").
    - true_col: String, column name for true anomaly labels.
    - pred_col: String, column name for predicted anomaly labels.
    - reward_tp: Float, reward for correctly detecting an anomaly.
    - penalty_fp: Float, penalty for a false positive.
    - penalty_fn: Float, penalty for a missed anomaly.

    Returns:
    - Float, raw NAB score.
    - Float, normalized NAB score.
    - Integer, false positive count.
    - Integer, false negative count.
    - Dict, detection counters.
    """

    # Adjust penalties based on the scoring profile
    if nab_scoring_profile == "reward_fn":
        penalty_fn = 2.0  # Override penalty for false negatives
    elif nab_scoring_profile != "standard":
        raise ValueError(f"Unsupported NAB scoring profile: {nab_scoring_profile}")

    score = 0.0
    false_positive_count = 0
    false_negative_count = 0

    detection_counters = {
        'issue_detected': 0,
        'issue_detected_ids': [],
        'im_detected': 0,
        'im_detected_ids': [],
        'TestLog_detected': 0,
        'TestLog_detected_ids': [],
        'gt_issue_ids': [],
        'gt_im_ids': [],
        'gt_TestLog_ids': [],
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
    }
    source_counters = {1: ('issue_detected', 'issue_detected_ids', 'gt_issue_ids'),
                       2: ('im_detected', 'im_detected_ids', 'gt_im_ids'),
                       3: ('TestLog_detected', 'TestLog_detected_ids', 'gt_TestLog_ids')}

    # Step 1: Use the true anomaly windows
    windows = []  # This will store (start, end) from anomaly_windows_test
    anomaly_sources = {}  # Map windows to their sources
    for index, (start, end, source) in enumerate(zip(pd.to_datetime(anomaly_windows_test['anomaly_window_start']),
                                                     pd.to_datetime(anomaly_windows_test['anomaly_window_end']),
                                                     anomaly_windows_test['anomaly_source'])):
        if source in source_counters:
            detection_counters[source_counters[source][2]].append(index)
        windows.append((start, end))
        anomaly_sources[(start, end)] = source
    idx_anomaly_map = {key: i for i, key in enumerate(anomaly_sources)}
    log.debug(f'True anomaly windows: {windows}')

    # Step 2: Calculate relative positions for predictions
    df = calculate_relative_position(df, true_col=true_col, pred_col=pred_col)

    # === True Positive Scoring ===
    for start, end in windows:
        try:
            # Check for any predictions within the true anomaly window
            predictions = df.loc[start:end, pred_col]
            window_detected = predictions.any()
        except Exception as e:
            log.warning(f"Error accessing window {start} to {end}: {e}")
            continue

        if window_detected:
            try:
                # Get the first detection time and calculate the relative position
                first_detection = predictions.idxmax()
                relative_position = df.loc[first_detection, 'relative_position']
                tp_score = reward_tp * scaledSigmoid(relative_position) / MAX_TP
                score += tp_score
                detection_counters['tp'] += 1

                # Add source-specific counters
                anomaly_type = anomaly_sources[(start, end)]
                if anomaly_type in source_counters:
                    detection_counters[source_counters[anomaly_type][0]] += 1
                    detection_counters[source_counters[anomaly_type][1]].append(idx_anomaly_map[(start, end)])
                log.debug(f"TP detected at {first_detection}, relative position {relative_position}, score {tp_score}")
            except Exception as e:
                log.warning(f"Error in true positive calculation: {e}")
                continue
        else:
            # False Negative (FN) case
            score -= penalty_fn
            false_negative_count += 1
            detection_counters['fn'] += 1

    # === False Positive Scoring ===
    # Only the predicted timestamps can be false positives
    predicted_times = df.index[df[pred_col].to_numpy() == 1]
    for time in predicted_times:
        if any(start <= time <= end for start, end in windows):
            continue
        # FP outside any anomaly window
        try:
            # Find the last anomaly window before this FP
            last_anomaly_window = None
            for start, end in windows:
                if end < time:
                    last_anomaly_window = (start, end)
                else:
                    break

            if last_anomaly_window:
                last_end = last_anomaly_window[1]
                window_width = (last_end - last_anomaly_window[0]).total_seconds()

                # Calculate FP relative position from the right boundary of the last window
                fp_offset = (time - last_end).total_seconds()
                relative_position_fp = fp_offset / window_width

                if relative_position_fp > 3:
                    fp_score = -1.0 * penalty_fp  # FP far beyond window, score -1
                else:
                    fp_score = penalty_fp * scaledSigmoid(relative_position_fp)  # Scaled sigmoid score
            else:
                fp_score = -1.0 * penalty_fp  # FP before the first window: full penalty, as in NAB

            score += fp_score
            false_positive_count += 1
            detection_counters['fp'] += 1
        except Exception as e:
            log.warning(f"Error calculating FP score: {e}")
            continue

    # Calculate baseline and perfect scores
    baseline_score = calculate_baseline_score(df, true_col=true_col, penalty_fn=penalty_fn)
    perfect_score, _ = calculate_perfect_score(df, true_col=true_col, reward_tp=reward_tp)

    # Normalize the score
    normalized_score = normalize_nab_score(score, baseline_score, perfect_score)
    log.debug(f"Baseline score: {baseline_score}, perfect score: {perfect_score}, normalized NAB score: {normalized_score}")

    return score, normalized_score, false_positive_count, false_negative_count, detection_counters
