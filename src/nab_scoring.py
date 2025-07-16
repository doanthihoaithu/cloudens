import math
import pandas as pd
from scipy.stats import norm

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
    df['shift'] = df[true_col].shift(1, fill_value=0)
    df['start_window'] = (df[true_col] == 1) & (df['shift'] == 0)
    df['end_window'] = (df[true_col] == 0) & (df['shift'] == 1)

    windows = []
    start_time = None

    for time, row in df.iterrows():
        if row['start_window']:
            start_time = time
        if row['end_window'] and start_time is not None:
            windows.append((start_time, time))
            start_time = None

    if start_time is not None:
        windows.append((start_time, df.index[-1]))

    df.drop(['shift', 'start_window', 'end_window'], axis=1, inplace=True)

    df['relative_position'] = 0.0
    for start, end in windows:
        window_length = (end - start).total_seconds()
        window_detected = df.loc[start:end, pred_col].any()

        if window_detected:
            first_detection = df.loc[start:end, pred_col].idxmax()
            relative_position_tp = -(end - first_detection).total_seconds() / window_length
            df.loc[first_detection, 'relative_position'] = relative_position_tp

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
    df['shift'] = df[true_col].shift(1, fill_value=0)
    df['start_window'] = (df[true_col] == 1) & (df['shift'] == 0)
    df['end_window'] = (df[true_col] == 0) & (df['shift'] == 1)

    windows = []
    start_idx = None

    for i, row in df.iterrows():
        if row['start_window']:
            start_idx = i
        if row['end_window']:
            if start_idx is not None:
                windows.append((start_idx, i))
                start_idx = None

    if start_idx is not None:
        windows.append((start_idx, df.index[-1]))

    df.drop(['shift', 'start_window', 'end_window'], axis=1, inplace=True)

    fn_count = len(windows)  # Count of all anomaly windows (false negatives if undetected)
    baseline_score = -penalty_fn * fn_count
    return baseline_score

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
    df['shift'] = df[true_col].shift(1, fill_value=0)
    df['start_window'] = (df[true_col] == 1) & (df['shift'] == 0)
    df['end_window'] = (df[true_col] == 0) & (df['shift'] == 1)

    windows = []
    start_idx = None

    for i, row in df.iterrows():
        if row['start_window']:
            start_idx = i
        if row['end_window']:
            if start_idx is not None:
                windows.append((start_idx, i))
                start_idx = None

    if start_idx is not None:
        windows.append((start_idx, df.index[-1]))

    df.drop(['shift', 'start_window', 'end_window'], axis=1, inplace=True)

    tp_count = len(windows)
    perfect_score = reward_tp * tp_count
    return perfect_score, tp_count

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
    - Float, weighted NAB score.
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
    tp_count = 0
    tn_count = 0

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

    # Step 1: Use the true anomaly windows
    windows = []  # This will store (start, end) from anomaly_windows_test
    anomaly_sources = {}  # Map windows to their sources

    for index, (_,row) in enumerate(anomaly_windows_test.iterrows()):
        start = pd.to_datetime(row['anomaly_window_start'])
        end = pd.to_datetime(row['anomaly_window_end'])
        source = row['anomaly_source']
        if source == 1:
            detection_counters['gt_issue_ids'].append(index)
        elif source == 2:
            detection_counters['gt_im_ids'].append(index)
        elif source == 3:
            detection_counters['gt_TestLog_ids'].append(index)
        windows.append((start, end))
        anomaly_sources[(start, end)] = source

    # print(f'issue anomaly ids: {detection_counters["gt_issue_ids"]}')
    # print(f'im anomaly ids: {detection_counters["gt_im_ids"]}')
    # print(f'TestLog anomaly ids: {detection_counters["gt_TestLog_ids"]}')

    idx_anomaly_map = {key: i for i, key in enumerate(anomaly_sources)}

    # Print the identified true anomaly windows and sources
    print("True Anomaly Windows:", windows)
    print("Anomaly Sources:", anomaly_sources)

    # Step 2: Calculate relative positions for predictions
    df = calculate_relative_position(df, true_col=true_col, pred_col=pred_col)

    # === True Positive Scoring ===
    print("=== True Positive Scoring ===")
    for start, end in windows:
        # print(f"Analyzing anomaly window: {start} to {end}")

        try:
            # Check for any predictions within the true anomaly window
            window_detected = df.loc[start:end, pred_col].any()
        except Exception as e:
            print(f"Error accessing window {start} to {end}: {e}")
            continue

        if window_detected:
            try:
                # Get the first detection time and calculate the relative position
                first_detection = df.loc[start:end, pred_col].idxmax()
                relative_position = df.loc[first_detection, 'relative_position']
                tp_score = reward_tp * scaledSigmoid(relative_position)
                score += tp_score
                tp_count += 1
                detection_counters['tp'] += 1

                # Add source-specific counters
                anomaly_type = anomaly_sources[(start, end)]
                anomaly_index = idx_anomaly_map[(start, end)]
                if anomaly_type == 1:
                    detection_counters['issue_detected'] += 1
                    detection_counters['issue_detected_ids'].append(anomaly_index)
                    print("Issue Detected:", detection_counters['issue_detected'])
                elif anomaly_type == 2:
                    detection_counters['im_detected'] += 1
                    detection_counters['im_detected_ids'].append(anomaly_index)
                    print("Instant Messenger Detected:", detection_counters['im_detected'])
                elif anomaly_type == 3:
                    detection_counters['TestLog_detected'] += 1
                    detection_counters['TestLog_detected_ids'].append(anomaly_index)
                    print("TestLog Detected:", detection_counters['TestLog_detected'])

                # Print TP details
                print(f"TP Detected at: {first_detection}")
                print(f"Relative Position: {relative_position}")
                print(f"TP Score: {tp_score}\n")

            except Exception as e:
                print(f"Error in true positive calculation: {e}")
                continue
        else:
            # False Negative (FN) case
            score -= penalty_fn
            false_negative_count += 1
            detection_counters['fn'] += 1
            # print(f"No TP detected in window {start} to {end}, FN Penalty applied.\n")

    # === False Positive Scoring ===
    print("=== False Positive Scoring ===")
    for time, row in df.iterrows():
        if row[pred_col] == 1 and not any(start <= time <= end for start, end in windows):
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

                    score += fp_score
                    false_positive_count += 1
                    detection_counters['fp'] += 1

                    # Print FP
                    # print(f"FP Detected at: {time}")
                    # print(f"Relative Position: {relative_position_fp}")
                    # print(f"FP Score: {fp_score}\n")
            except Exception as e:
                print(f"Error calculating FP score: {e}")
                continue

        if row[true_col] == 0 and row[pred_col] == 0:
            tn_count += 1

    # Calculate baseline and perfect scores
    baseline_score = calculate_baseline_score(df, true_col=true_col, penalty_fn=penalty_fn)
    perfect_score, _ = calculate_perfect_score(df, true_col=true_col, reward_tp=reward_tp)

    # Normalize the score
    normalized_score = normalize_nab_score(score, baseline_score, perfect_score)

    # Display the results
    print(f"Baseline Score: {baseline_score}")
    print(f"Perfect Score: {perfect_score}")
    print(f"Normalized NAB Score: {normalized_score}")

    return score, normalized_score, false_positive_count, false_negative_count, detection_counters

