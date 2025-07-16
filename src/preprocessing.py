import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import re
import json

import os
import logging

from utils import build_adjacency_matrix_no_group

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def clean_training_data(df, anomaly_windows):
    """
    Removes anomalies from the training data based on ground truth anomaly windows.

    Parameters:
    - df: DataFrame, training data.
    - anomaly_windows: DataFrame, contains the start and end times of anomaly windows.

    Returns:
    - df_cleaned: DataFrame, cleaned training data without anomalies.
    """
    # Convert 'anomaly_window_start' and 'anomaly_window_end' to timezone-naive
    anomaly_windows['anomaly_window_start'] = anomaly_windows['anomaly_window_start'].dt.tz_localize(None)
    anomaly_windows['anomaly_window_end'] = anomaly_windows['anomaly_window_end'].dt.tz_localize(None)

    df_label = df.copy()
    # df_label['5XX_count'] = df_label.sum(axis=1)
    df_label['isAnomaly'] = 0

    # for index, row in df_label.iterrows():
    #     # Ensure that index is tz-naive before comparison
    #     is_anomaly = ((anomaly_windows['anomaly_window_start'] <= index) &
    #                   (index <= anomaly_windows['anomaly_window_end'])).any()
    #     df_label.at[index, 'isAnomaly'] = 1 if is_anomaly else 0

    for index, anomaly_window in anomaly_windows.iterrows():
        # Ensure that index is tz-naive before comparison
        anomaly_indices = df_label[
            (df_label.index >= anomaly_window['anomaly_window_start']) &
            (df_label.index <= anomaly_window['anomaly_window_end'])].index

        df_label.loc[anomaly_indices, 'isAnomaly'] = 1.0

    # df_cleaned = df_label[df_label['isAnomaly'] == 0].drop(columns=['isAnomaly', '5XX_count'])
    df_cleaned = df_label[df_label['isAnomaly'] == 0].drop(columns=['isAnomaly'])
    print(f'Cleaning training data..., Drop {df_label.shape[0] - df_cleaned.shape[0]} rows')
    return df_cleaned

def filter_anomaly_windows(gt_utc_df, start_date, end_date, test_start_date):
    """
    Filter and prepare anomaly windows for training and testing.

    Parameters:
    - gt_utc_df: DataFrame, original anomaly windows.
    - start_date: Timestamp, start date of the experiment.
    - end_date: Timestamp, end date of the experiment.
    - test_start_date: Timestamp, start date of the testing period.

    Returns:
    - anomaly_windows: DataFrame, training anomaly windows.
    - anomaly_windows_test: DataFrame, testing anomaly windows.
    """
    gt_utc_df['anomaly_window_start'] = gt_utc_df['anomaly_window_start'].dt.tz_localize(None)
    gt_utc_df['anomaly_window_end'] = gt_utc_df['anomaly_window_end'].dt.tz_localize(None)

    # Training anomaly windows
    anomaly_windows = gt_utc_df.loc[
        (gt_utc_df['anomaly_window_start'] >= start_date) &
        (gt_utc_df['anomaly_window_end'] <= end_date)
        ]

    # Testing anomaly windows
    anomaly_windows_test = anomaly_windows.loc[
        (anomaly_windows['anomaly_window_start'] >= test_start_date) &
        (anomaly_windows['anomaly_window_end'] <= end_date)
        ]

    return anomaly_windows, anomaly_windows_test

def load_and_prepare_data_according_to_config(cfg):
    """
    Load and prepare the 5XX features data and anomaly windows.

    Parameters:
    - cfg: DictConfig, configuration object.

    Returns:
    - df_prodlive: DataFrame, prepared 5XX features data.
    - gt_utc_df: DataFrame, anomaly windows data.
    """

    # grouping_data_dir = cfg.grouping_data_dir
    # filtered_raw_filename = cfg.filtered_raw_filename

    input_file = cfg['filtered_raw_file']
    meta_data_file = cfg['meta_data_file']
    adjacency_file = cfg['adjacency_file']

    gt_input_file = cfg['ground_truth_labels_file']
    minutes_before = cfg['minutes_before']

    # input_filename = cfg.train_test_config.input_files.observations_file_name
    # gt_input_filename = cfg.train_test_config.input_files.ground_truth_labels_file_name
    # minutes_before = cfg.train_test_config.anomaly_window.minutes_before

    # Load the 5XX and 4XX features only
    if os.path.exists(input_file) and os.path.exists(meta_data_file) and os.path.exists(adjacency_file):
        filtered_df = pd.read_parquet(input_file, engine='fastparquet', index='interval_start')
        log.info(f"File '{input_file}' loaded successfully.")

    else:
        log.info(f'Some of the following files are missing!')
        log.info('\n'.join([input_file, meta_data_file, adjacency_file]))
        log.info(f'Generating data according to filter {cfg["filter"]}')
        filtered_df = prepare_5xx_4xx_features(cfg)

    # filtered_df.fillna(0, inplace=True)
    filtered_df.index = pd.to_datetime(filtered_df.index)
    print('filtered_df.shape', filtered_df.shape)

    # Load anomaly windows
    gt_df = pd.read_csv(gt_input_file)
    gt_df['anomaly_start'] = pd.to_datetime(gt_df['anomaly_start'], utc=True)
    gt_df['anomaly_end'] = pd.to_datetime(gt_df['anomaly_end'], utc=True)

    gt_df['anomaly_window_start'] = gt_df['anomaly_start'] - pd.Timedelta(minutes=minutes_before)
    gt_df['anomaly_window_end'] = gt_df['anomaly_end']

    gt_utc_df = gt_df[['number', 'anomaly_window_start', 'anomaly_window_end', 'anomaly_source']].copy()

    return filtered_df, gt_utc_df

def prepare_5xx_4xx_features(cfg):
    """
    Prepare 4XX features from a parquet file. Filters columns, sums them up, and saves the result to a CSV file.

    :param cfg: Configuration object containing the input and output file paths and filename template.
    """
    # Construct input and output paths
    # input_path = os.path.join(cfg.data_preparation_pipeline.features_prep.input_file.path,
    #                           cfg.data_preparation_pipeline.features_prep.input_file.filename)
    # output_path = os.path.join(cfg.data_preparation_pipeline.features_prep.output_file.path,
    #                            cfg.data_preparation_pipeline.features_prep.output_file.filename_template)

    input_path = cfg['pivoted_raw_data_file']
    output_path = cfg['filtered_raw_file']

    # Inspect the schema of the Parquet file to identify columns
    parquet_file = pq.ParquetFile(input_path)
    all_columns = parquet_file.schema.names

    filter = cfg['filter']
    http_codes = filter['http_codes']
    aggregations = filter['aggregations']

    first_aggregation = aggregations[0]

    regex_mapping = dict({
        '1xx': f'_\\-1_.*{first_aggregation}',
        '2xx': f'_2\\d\\d_.*{first_aggregation}',
        '3xx': f'_3\\d\\d_.*{first_aggregation}',
        '4xx': f'_4\\d\\d_.*{first_aggregation}',
        '5xx': f'_5\\d\\d_.*{first_aggregation}'
    })

    # Filter columns using regex and include 'interval_start'
    # regex_patterns = cfg.data_preparation_pipeline.features_prep.regex_patterns
    regex_patterns = [regex_mapping[k] for k in http_codes]
    regex_patterns = [f'({p})' for p in regex_patterns]
    pattern_combined = '|'.join(regex_patterns)
    columns_to_keep = [col for col in all_columns if
                       re.search(pattern_combined, col)]

    columns_to_keep_dictionary = dict({
        f'{first_aggregation}': columns_to_keep.copy()
    })

    # print(f'Columns to keep: {len(columns_to_keep)}')
    node_ids = [c.replace(f'_{first_aggregation}', '') for c in columns_to_keep]
    node_features = list(aggregations)
    meta_data = dict({
        'node_feature_names': node_features,
        'flatten_method': 'node_ids*node_features',
        'num_nodes': len(node_ids),
        'node_ids': node_ids,
    })

    json.dump(meta_data, open(cfg['meta_data_file'], 'w'), indent=4)
    log.info('Save meta data to "%s"', cfg['meta_data_file'])

    adjacency_matrix = build_adjacency_matrix_no_group(meta_data['node_ids'])
    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    adjacency_matrix = np.array([node_indices, neighbor_indices])
    np.save(cfg['adjacency_file'], adjacency_matrix)
    log.info(f'Save adjacency matrix (shape={adjacency_matrix.shape}) to {cfg["adjacency_file"]}')

    if len(aggregations) > 1:
        for idx, aggr in enumerate(aggregations[1:]):
            columns_to_keep_for_other_code = [c.replace(f'_{first_aggregation}', f'_{aggr}') for c in columns_to_keep]
            columns_to_keep_dictionary[aggr] = columns_to_keep_for_other_code
            columns_to_keep.extend(columns_to_keep_for_other_code)

    if 'interval_start' in all_columns and 'interval_start' not in columns_to_keep:
        columns_to_keep.append('interval_start')

    # Load only the filtered columns
    filtered_df = pd.read_parquet(input_path, columns=columns_to_keep)
    is_nan = filtered_df.isna().values.any()
    print(f'%nan in the dataset:', filtered_df.iloc[:,1:].isnull().sum().sum()/(filtered_df.iloc[:,1:].shape[0]*filtered_df.iloc[:,1:].shape[1])*100)
    if is_nan:
        log.info(f'NaN values detected in "{input_path}", filling NaN values')
        for arr, list_c in columns_to_keep_dictionary.items():
            if arr == 'count':
                filtered_df[list_c] = filtered_df[list_c].fillna(0)
                print(f'Fillna with 0 for {arr} features')
            else:
                medians = filtered_df[list_c].median()
                filtered_df[list_c] = filtered_df[list_c].fillna(medians)
                print(f'Fillna with median for {arr} features')
    is_nan = filtered_df.isna().values.any()
    assert is_nan == False

    # Convert 'interval_start' to datetime and set it as index
    if 'interval_start' in filtered_df.columns:
        filtered_df['interval_start'] = pd.to_datetime(filtered_df['interval_start'], unit="s")
        filtered_df.set_index('interval_start', inplace=True)


    # Extract data center name from filename
    # match = re.search(r'pivoted_data_([^_]+)\.parquet', cfg.data_preparation_pipeline.features_prep.input_file.filename)
    # data_center_name = match.group(1) if match else 'unknown'

    # Construct full output file path
    # output_file_location = output_path.format(data_center_name=data_center_name)

    # Ensure the directory exists
    dir_to_save_to = os.path.dirname(output_path)
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    # Save to CSV
    filtered_df.to_parquet(output_path, index=True)
    log.info(f'Data saved as "{output_path}"')

    return filtered_df

