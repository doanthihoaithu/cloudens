import io
import json
import logging
import os
import ssl
import urllib.request
from datetime import timedelta

import torch
from torch.nn import functional as F

import hydra
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

import numpy as np
import pandas as pd
from torch_geometric_temporal import StaticGraphTemporalSignal, A3TGCN2, temporal_signal_split
from tqdm import tqdm

from preprocessing import clean_training_data, filter_anomaly_windows,\
    load_and_prepare_data_according_to_config
from utils import get_project_root, ProgressBar, NumpyEncoder

import wget

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class IBMDatasetLoader(object):

    def __init__(self, data_preparation_config):
        super(IBMDatasetLoader, self).__init__()

        log.info("Starting preparation for features")
        grouping_mode = data_preparation_config.grouping_mode
        self.grouping_mode = grouping_mode

        selected_http_codes = data_preparation_config.features_prep.filter.http_codes
        selected_aggregations = data_preparation_config.features_prep.filter.aggregations

        log.info('Grouping mode: {}'.format(grouping_mode))
        log.info('Selected http codes: {}'.format(selected_http_codes))
        log.info('Selected aggregations: {}'.format(selected_aggregations))

        root_dir = get_project_root()
        data_dir = os.path.join(root_dir, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        log.info('Detect project root directory: {}'.format(root_dir))
        log.info('Extracted data will be saved in {}'.format(data_dir))


        pivoted_raw_data_dir = os.path.join(root_dir, data_preparation_config.input_file.path)
        if not os.path.exists(pivoted_raw_data_dir):
            os.makedirs(pivoted_raw_data_dir)
            log.info('Created pivoted input directory: {}'.format(pivoted_raw_data_dir))
        else:
            log.info('Existed pivoted input directory: {}'.format(pivoted_raw_data_dir))

        anomaly_windows_relative_dir = data_preparation_config.ground_truth_labels.path

        anomaly_windows_dir = os.path.join(root_dir, anomaly_windows_relative_dir)
        if not os.path.exists(anomaly_windows_dir):
            os.makedirs(anomaly_windows_dir)
            log.info('Created anomaly windows directory: {}'.format(anomaly_windows_dir))
        else:
            log.info('Existed anomaly windows directory: {}'.format(anomaly_windows_dir))

        self.pivoted_raw_data_dir = pivoted_raw_data_dir
        self.anomaly_windows_dir = anomaly_windows_dir

        self.data_preparation_config = data_preparation_config
        print("data_preparation_config", self.data_preparation_config)

        # print("pivoted_raw_data_dir", self.pivoted_raw_data_dir)
        # print("label_data_dir", self.label_data_dir)
        # print("data_preparation_config", self.data_preparation_config)


        # self.global_cfg = global_cfg



        self._download_web_data()
        print("pivoted_raw_data_file", self.pivoted_raw_data_file)
        print("anomaly_windows_file", self.anomaly_windows_file)

        self.anomaly_windows, self.anomaly_windows_test = self._load_anomaly_windows()

        self._read_downloaded_data()

        self._split_data()
        # self.node_ids = []
        # self.node_features_names = []

    def _split_data(self):
        # start_date = self.data_preparation_config.train_test_config.experiment_parameters.start_date
        # train_end_date = self.data_preparation_config.train_test_config.experiment_parameters.train_end_date
        # test_start_date = self.data_preparation_config.train_test_config.experiment_parameters.test_start_date
        # end_date = self.data_preparation_config.train_test_config.experiment_parameters.end_date

        start_date = pd.Timestamp(self.data_preparation_config.train_test_config.experiment_parameters.start_date)
        train_end_date = pd.Timestamp(self.data_preparation_config.train_test_config.experiment_parameters.train_end_date)
        test_start_date = pd.Timestamp(self.data_preparation_config.train_test_config.experiment_parameters.test_start_date)
        end_date = pd.Timestamp(self.data_preparation_config.train_test_config.experiment_parameters.end_date) + timedelta(days=1)

        assert self.timestamp_and_label_df.shape[0] == self.X_raw.shape[0]
        print(self.X_raw.index[:5])
        print('Splitting data into train and test sets')

        # Extract experiment parameters
        # start_date = pd.Timestamp(start_date)
        # train_end_date = pd.Timestamp(train_end_date)
        # test_start_date = pd.Timestamp(test_start_date)
        # end_date = pd.Timestamp(end_date) + timedelta(days=1)

        # result_directory = cfg.evaluation.result_file.path
        # model_save_directory = cfg.evaluation.model_save_path
        # nab_scoring_profile = cfg.evaluation.nab_scoring_profile
        # use_model = cfg.evaluation.use_model

        # Filter data for training and testing
        working_data = self.X_raw.loc[start_date:end_date]
        train_data = working_data.loc[start_date:train_end_date]
        train_data = clean_training_data(train_data, self.anomaly_windows)

        test_data = working_data.loc[test_start_date:end_date]
        self.test_labels = self.timestamp_and_label_df.loc[test_start_date:end_date]['is_anomaly']
        self.test_index = test_data.index
        assert len(self.test_labels) == len(self.test_index)

        self.X_train_raw = train_data
        self.X_test_raw = test_data

        scaler = MinMaxScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train_raw)
        self.X_test_scaled = scaler.transform(self.X_test_raw)

        print(f'Train data raw shape {self.X_train_raw.shape}, Test data raw shape {self.X_test_raw.shape}')


    def _load_anomaly_windows(self):
        anomaly_windows_file = self.anomaly_windows_file
        data_config = self.data_preparation_config
        gt_df = pd.read_csv(anomaly_windows_file)
        gt_df['anomaly_start'] = pd.to_datetime(gt_df['anomaly_start'], utc=True)
        gt_df['anomaly_end'] = pd.to_datetime(gt_df['anomaly_end'], utc=True)

        minutes_before = data_config.train_test_config.anomaly_window.minutes_before
        gt_df['anomaly_window_start'] = gt_df['anomaly_start'] - pd.Timedelta(minutes=minutes_before)
        gt_df['anomaly_window_end'] = gt_df['anomaly_end']

        gt_utc_df = gt_df[['number', 'anomaly_window_start', 'anomaly_window_end', 'anomaly_source']].copy()

        start_date = pd.Timestamp(data_config.train_test_config.experiment_parameters.start_date)
        train_end_date = pd.Timestamp(data_config.train_test_config.experiment_parameters.train_end_date)
        test_start_date = pd.Timestamp(data_config.train_test_config.experiment_parameters.test_start_date)
        end_date = pd.Timestamp(data_config.train_test_config.experiment_parameters.end_date) + timedelta(days=1)

        anomaly_windows, anomaly_windows_test = filter_anomaly_windows(gt_utc_df, \
                                                                       start_date, \
                                                                       end_date, \
                                                                       test_start_date)
        return anomaly_windows, anomaly_windows_test

    def get_dataset(self, window_size: int = 6):
        self._get_edges_and_weights()
        self._generate_task(window_size)
        dataset_train = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features_train, self.targets_train
        )

        dataset_train, dataset_valid = temporal_signal_split(dataset_train, train_ratio=0.8)

        dataset_test = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features_test, self.targets_test
        )

        return dataset_train, dataset_valid, dataset_test

    def _generate_task(self, window_size: int, train_val_ratio: float=0.8):

        windowed_data = (torch.tensor(self.X_train_scaled).unfold(dimension=0, size=window_size + 1, step=1).permute(0, 2, 1)
                         .reshape(-1,window_size+1, self.num_nodes, self.num_node_features)).numpy()

        train_offset = int(train_val_ratio * windowed_data.shape[0])
        windowed_data_train = windowed_data[0:train_offset]
        windowed_data_valid = windowed_data[train_offset:]

        self.features_train = windowed_data_train[:, :window_size, :,:]
        self.targets_train = windowed_data_train[:, window_size, :,:]

        self.features_valid = windowed_data_valid[:, :window_size, :, :]
        self.targets_valid = windowed_data_valid[:, window_size, :, :]

        windowed_data = np.concatenate([self.X_test_scaled[0:1,:].repeat(window_size, 0), self.X_test_scaled], axis=0)
        windowed_data = torch.tensor(windowed_data).unfold(dimension=0,size=window_size+1,step=1).permute(0,2,1).reshape(-1,window_size+1, self.num_nodes, self.num_node_features).numpy()
        self.features_test = windowed_data[:, :window_size,:,:]
        self.targets_test = windowed_data[:, window_size,:,:]

    def _get_edges_and_weights(self):
        # self.edges = np.array(self.adjacency_matrix)
        # self.edge_weights = np.ones(self.edges.shape[1])
        self.edges = self.adjacency_matrix[:2,:]
        self.edge_weights = self.adjacency_matrix[2,:]

    def get_edges_as_tensor(self, device='cpu'):
        return torch.LongTensor(self.edges).to(device)

    def get_edge_weights_as_tensor(self, device='cpu'):
        return torch.FloatTensor(self.edge_weights).to(device)

    def _download_url(self, url, save_path):  # pragma: no cover
        context = ssl._create_unverified_context()

        wget.download(url, save_path, ProgressBar())
        # with urllib.request.urlopen(url, context=context) as dl_file:
        #     length = dl_file.getheader('content-length')
        #     if length:
        #         length = int(length)
        #         blocksize = max(4096, length // 10000)
        #     else:
        #         blocksize = 1000000
        #
        #     print(length, blocksize)
        #     with open(save_path, "wb") as out_file:
        #         size = 0
        #         while True:
        #             buf1 = dl_file.read(blocksize)
        #             if not buf1:
        #                 break
        #             out_file.write(buf1)
        #             size += len(buf1)
        #             if length:
        #                 print('{:.2f}\r done'.format(size / length), end='')
        #         # print()
        #         # out_file.write(dl_file.read())
    def _download_web_data(self):
        # curl - L - o
        # data / labels / anomaly_windows.csv
        # https: // zenodo.org / records / 14062900 / files / anomaly_windows.csv?download = 1
        # curl - L - o
        # data / massaged / pivoted_data_all.parquet
        # https: // zenodo.org / records / 14062900 / files / pivoted_data_all.parquet?download = 1
        data_url = "https://zenodo.org/records/14062900/files/pivoted_data_all.parquet?download=1"
        anomaly_url = "https://zenodo.org/records/14062900/files/anomaly_windows.csv?download=1"

        anomaly_windows_file = os.path.join(self.anomaly_windows_dir, 'anomaly_windows.csv')

        if not os.path.isfile(anomaly_windows_file):
            if not os.path.exists(self.anomaly_windows_dir):
                os.makedirs(self.anomaly_windows_dir)
            print("Downloading label data from Zenodo...")
            self._download_url(anomaly_url, anomaly_windows_file)
        else:
            print('Loading saved anomaly windows at:', anomaly_windows_file)

        self.anomaly_windows_file = anomaly_windows_file

        pivoted_raw_data_file = os.path.join(self.pivoted_raw_data_dir, 'pivoted_data_all.parquet')

        if not os.path.isfile(pivoted_raw_data_file):
            if not os.path.exists(self.pivoted_raw_data_dir):
                os.makedirs(self.pivoted_raw_data_dir)
            print("Downloading pivoted raw data from Zenodo...")
            self._download_url(data_url, pivoted_raw_data_file)
        else:
            print('Loading saved pivoted data at:', pivoted_raw_data_file)

        self.pivoted_raw_data_file = pivoted_raw_data_file


    def _read_downloaded_data(self):
        total_column_file = os.path.join(self.pivoted_raw_data_dir, 'columns.csv')
        if not os.path.isfile(total_column_file):
            if not os.path.exists(self.pivoted_raw_data_dir):
                os.makedirs(self.pivoted_raw_data_dir)
            print("Extract total columns from pivoted raw data at:", self.pivoted_raw_data_file)
            column_df = self._extract_total_columns()
        else:
            print('Loading total columns dataframe at', total_column_file)
            column_df = pd.read_csv(total_column_file)
        self._print_details_of_total_columns(column_df)

        timestamp_file = os.path.join(self.pivoted_raw_data_dir, 'timestamps_and_labels.csv')
        if not os.path.isfile(timestamp_file):
            if not os.path.exists(self.pivoted_raw_data_dir):
                os.makedirs(self.pivoted_raw_data_dir)
            print("Extract timestamps from pivoted raw data at:", self.pivoted_raw_data_file)
            self._extract_timestamps_and_labels()
            timestamp_df = pd.read_csv(timestamp_file, index_col='DateTime')
            timestamp_df.index = pd.to_datetime(timestamp_df.index)
            self.timestamp_file = timestamp_file
        else:
            print('Loading total columns dataframe at', timestamp_file)
            timestamp_df = pd.read_csv(timestamp_file, index_col='DateTime')
            timestamp_df.index = pd.to_datetime(timestamp_df.index)
            self.timestamp_file = timestamp_file
        self.timestamp_and_label_df = timestamp_df
        self.print_details_of_timestamps(timestamp_df)

        feature_df, meta_data, adjacency_matrix = self.construct_data_to_nodes_with_features(column_df)
        # feature_shaped = feature_df.values.reshape(-1, len(meta_data['node_ids']), len(meta_data['node_feature_names']))
        assert isinstance(feature_df.index, pd.DatetimeIndex)
        self.X_raw = feature_df
        self.X_raw.index = self.timestamp_and_label_df.index
        self.meta_data = meta_data
        self.adjacency_matrix = adjacency_matrix

        # print('Graph data', self.X_raw.shape)


    def construct_data_to_nodes_with_features(self, column_df):
        filter = self.data_preparation_config.features_prep.filter
        http_codes = filter.http_codes
        aggregations = filter.aggregations
        selected_group_mode = f'{self.grouping_mode}_{"_".join(http_codes)}_{"_".join(aggregations)}'
        self.selected_group_mode = selected_group_mode

        # GROUPING_APPROACH = {'center_component': 'center+component+type+calculation_type', \
        #                      'component_endpoint': 'component+endpoint+type+calculation_type', \
        #                      'center_endpoint': 'center+endpoint+type+calculation_type', \
        #                      'center_component_endpoint': 'center+component+endpoint+type+calculation_type', \
        #                      'center': 'center+type+calculation_type', \
        #                      'component': 'component+type+calculation_type', \
        #                      'endpoint': 'endpoint+type+calculation_type',\
        #                      'no_group_5xx_4xx': 'no_group+5xx+4xx',\
        #                      'no_group_5xx': 'no_group+5xx',
        #                      'no_group_4xx': 'no_group+4xx'}
        # group_method = GROUPING_APPROACH[selected_group_mode]

        grouping_data_dir = os.path.join(self.pivoted_raw_data_dir, 'extracted', selected_group_mode)
        if not os.path.exists(grouping_data_dir):
            os.makedirs(grouping_data_dir)

        feature_raw_file = os.path.join(grouping_data_dir, 'filtered_raw_data.parquet')
        meta_data_file = os.path.join(grouping_data_dir, 'meta_data.json')
        adjacency_file = os.path.join(grouping_data_dir, 'adjacency_matrix.npy')

        if os.path.exists(feature_raw_file) and os.path.exists(meta_data_file) and os.path.exists(adjacency_file):
            print(f'Found data files corresponding to grouping mode <{selected_group_mode}>. Loading existed files at: {grouping_data_dir}')
            feature_data_df, meta_data, adjacency_matrix = self._loading_features_according_to_grouping_mode(feature_raw_file, meta_data_file, adjacency_file)
            self.num_nodes = len(meta_data['node_ids'])
            self.num_node_features = len(meta_data['node_feature_names'])
            return feature_data_df, meta_data, adjacency_matrix
        else:
            print('Start reconstructing data according to grouping mode', selected_group_mode)
            if 'no_group' in selected_group_mode:
                loader_config = dict({
                    'pivoted_raw_data_file': self.pivoted_raw_data_file,
                    'ground_truth_labels_file': self.anomaly_windows_file,
                    # 'grouping_data_dir': grouping_data_dir,
                    'filtered_raw_file': feature_raw_file,
                    'feature_filename': 'features.parquet',
                    'meta_data_file': meta_data_file,
                    'adjacency_file': adjacency_file,
                    'filter': filter,
                    'minutes_before': self.data_preparation_config.train_test_config.anomaly_window.minutes_before,
                })
                df_prodlive, gt_utc_df = load_and_prepare_data_according_to_config(loader_config)

                # meta_data = dict({})
                # meta_data['node_ids'] = list(df_prodlive.columns)
                # meta_data['node_feature_names'] = http_codes
                # json.dump(meta_data, open(meta_data_file, 'w'), cls=NumpyEncoder)

                meta_data = json.load(open(meta_data_file, 'r'))
                log.info(f'Loaded meta data at  {meta_data_file}')

                print('Loading adjacency matrix at', adjacency_file)

                adjacency_matrix = np.load(adjacency_file)
                print(f'Number of edges: {adjacency_matrix.shape[1]}')

                self.num_nodes = len(meta_data['node_ids'])
                self.num_node_features = len(meta_data['node_feature_names'])



                reconstructed_df = df_prodlive
                print('Reconstruct_df.shape', reconstructed_df.shape)
                print(f'Saving feature data to {feature_raw_file}')
                reconstructed_df.to_parquet(feature_raw_file, index=None)
                print(f'Done saving feature data!')
                return reconstructed_df, meta_data, adjacency_matrix

            # else:
            #     selected_columns = group_method.split('+')
            #     grouped_df = column_df[selected_columns].value_counts().to_frame('count').reset_index()
            #     grouped_df = grouped_df[grouped_df['calculation_type'] == 'count']
            #
            #     # patterns = ['_5\d\d_endpoint']
            #     patterns = column_df['type'].unique()
            #     calculation_types = ['count']
            #     columns_dictionary = dict()
            #     center_component_pattern_length = grouped_df.shape[0]
            #
            #     for index, entry in tqdm(enumerate(
            #             grouped_df[selected_columns].values), total=center_component_pattern_length, \
            #             desc='Grouping columns for nodes...'):
            #         # print(f'{index}/{center_component_pattern_length}:', center, component, http_code_pattern)
            #         http_code_pattern = entry[-2]
            #         calculation_type = entry[-1]
            #         center_index = selected_columns.index('center') if 'center' in selected_columns else None
            #         if center_index is not None:
            #             center = entry[center_index]
            #         else:
            #             center = None
            #         component_index = selected_columns.index('component') if 'component' in selected_columns else None
            #         if component_index is not None:
            #             component = entry[component_index]
            #         else:
            #             component = None
            #         endpoint_index = selected_columns.index('endpoint') if 'endpoint' in selected_columns else None
            #         if endpoint_index is not None:
            #             endpoint = entry[endpoint_index]
            #         else:
            #             endpoint = None
            #
            #
            #
            #         # condition = [column_df[selected_columns[i]] == groups[i] for i, group_name in enumerate(groups)]
            #         if center_index is None:
            #             if endpoint is not None and component is not None:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['endpoint'] == endpoint) & \
            #                                                   (column_df['component'] == component) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #             elif endpoint is not None:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['endpoint'] == endpoint) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #             else:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['component'] == component) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #         elif component_index is None:
            #             if center is not None and endpoint is not None:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['center'] == center) & \
            #                                                   (column_df['endpoint'] == endpoint) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #             elif center is not None:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['center'] == center) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #             else:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['endpoint'] == endpoint) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #
            #         elif endpoint_index is None:
            #             if center is not None and component is not None:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['center'] == center) & \
            #                                                   (column_df['component'] == component) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #             elif center is not None:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['center'] == center) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #             else:
            #                 corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                                   (column_df['calculation_type'].isin(calculation_types)) & \
            #                                                   (column_df['component'] == component) & \
            #                                                   (column_df['type'] == http_code_pattern)
            #                                                   ]
            #         elif center_index and component_index and endpoint_index:
            #             corresponding_columns = column_df[(column_df['type'].isin(patterns)) & \
            #                                               (column_df['calculation_type'].isin(calculation_types)) & \
            #                                               (column_df['center'] == center) & \
            #                                               (column_df['component'] == component) & \
            #                                               (column_df['endpoint'] == endpoint) & \
            #                                               (column_df['type'] == http_code_pattern)
            #                                               ]
            #         else:
            #             corresponding_columns = column_df
            #         node_key = f'{"+".join(entry[:-2])}'
            #         if node_key not in columns_dictionary:
            #             columns_dictionary[node_key] = dict()
            #         columns_dictionary[node_key][f'{http_code_pattern}'] = \
            #             corresponding_columns['column_name'].values
            #
            #     print("Number of nodes:", len(columns_dictionary.keys()))
            #
            #     grouped_columns_file = os.path.join(grouping_data_dir, f'columns_group_by_{selected_group_mode}.json')
            #     json.dump(columns_dictionary, open(grouped_columns_file, 'w'), cls=NumpyEncoder)
            #     query_columns = self._combine_all_columns_from_dict(columns_dictionary)

            # parquet_file = pq.ParquetFile(self.pivoted_raw_data_file)
            # batch_dfs = []
            # for i, batch in enumerate(parquet_file.iter_batches(batch_size=10000, columns=query_columns)):
            #     # if i > 1 and (selected_group_mode == 'center_endpoint'):
            #     #     break
            #     print(f'Processing batch: {i}')
            #     batch_df = batch.to_pandas()
            #
            #     required_patterns = ['_-1_endpoint', '_2\\d\\d_endpoint', '_3\\d\\d_endpoint', \
            #                          '_4\\d\\d_endpoint', '_5\\d\\d_endpoint']
            #     node_dfs = []
            #     for node_id, column_group in columns_dictionary.items():
            #         data = dict()
            #         existing_http_code_patterns = list(column_group.keys())
            #
            #         for http_code_pattern in required_patterns:
            #             http_code = '5xx'
            #             if http_code_pattern == required_patterns[0]:
            #                 http_code = '1xx'
            #             elif http_code_pattern == required_patterns[1]:
            #                 http_code = '2xx'
            #             elif http_code_pattern == required_patterns[2]:
            #                 http_code = '3xx'
            #             elif http_code_pattern == required_patterns[3]:
            #                 http_code = '4xx'
            #             else:
            #                 http_code = '5xx'
            #
            #             if http_code_pattern in existing_http_code_patterns:
            #                 corresponding_columns = column_group[http_code_pattern]
            #                 data[f'{node_id}+{http_code}+count'] = batch_df[corresponding_columns].sum(axis=1)
            #             else:
            #                 data[f'{node_id}+{http_code}+count'] = pd.Series(np.zeros(len(batch)))
            #
            #         new_df = pd.DataFrame(data)
            #         node_dfs.append(new_df)
            #     batch_df = pd.concat(node_dfs, axis=1)
            #     batch_dfs.append(batch_df)
            #
            #
            # meta_data = dict({})
            # meta_data['node_ids'] = list(columns_dictionary.keys())
            # meta_data['node_feature_names'] = [f'{calculation_types[0]}_{code}' for code in ['1xx', '2xx', '3xx', '4xx', '5xx']]
            # json.dump(meta_data, open(meta_data_file, 'w'), cls=NumpyEncoder)
            # self.num_nodes = len(meta_data['node_ids'])
            # self.num_node_features = len(meta_data['node_feature_names'])
            #
            # adjacency_matrix = self._build_adjacency_matrix(meta_data['node_ids'])
            # node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
            # np.save(adjacency_file, np.array([node_indices, neighbor_indices]))
            #
            # reconstructed_df = pd.concat(batch_dfs, axis=0)
            # print('Reconstruct_df.shape', reconstructed_df.shape)
            # print(f'Saving feature data to {feature_raw_file}')
            # reconstructed_df.to_parquet(feature_raw_file, index=None)
            # print(f'Done saving feature data!')
            # return reconstructed_df, meta_data, adjacency_matrix

    def _build_adjacency_matrix(self, node_ids):
        matrix = []
        for index_i, i in enumerate(node_ids):
            i_distance = np.zeros(len(node_ids))
            i_tokens = np.array(i.split('+'))
            for index_j, j in enumerate(node_ids):
                j_tokens = np.array(j.split('+'))

                # correlated = np.any([i_tokens[token_index] == j_tokens[token_index] for token_index in range(len(i_tokens))])
                correlated = i_tokens[-1] == j_tokens[-1]
                if correlated:
                    i_distance[index_i] = 1

                if index_i == index_j:
                    i_distance[index_j] = 0
            matrix.append(i_distance)
        return np.array(matrix)

    def _loading_features_according_to_grouping_mode(self, feature_data_file, meta_data_file, adjacency_matrix_file):
        # feature_data_file = os.path.join(group_data_dir, 'filtered_raw_data.parquet')
        # meta_data_file = os.path.join(group_data_dir, 'meta_data.json')
        # adjacency_matrix_file = os.path.join(group_data_dir, 'adjacency_matrix.npy')

        print('Loading meta data at', meta_data_file)
        meta_data_json = json.load(open(meta_data_file, 'r'))
        node_ids = meta_data_json['node_ids']
        node_feature_names = meta_data_json['node_feature_names']
        print(f'Number of nodes:{len(node_ids)}, Node feature names:{node_feature_names}')

        print('Loading adjacency matrix at', adjacency_matrix_file)
        adjacency_matrix = np.load(adjacency_matrix_file)
        print(f'Number of edges: {adjacency_matrix.shape[1]}')

        print('Loading features at', feature_data_file)
        feature_df = pd.read_parquet(feature_data_file)
        feature_df.index = pd.to_datetime(feature_df.index)

        assert adjacency_matrix.shape[0] == 3
        assert feature_df.shape[1]/len(node_ids) == len(node_feature_names)

        print('Loading reconstructed time series at', feature_data_file)

        return feature_df, meta_data_json, adjacency_matrix


    def _combine_all_columns_from_dict(self, columns_dictionary):
        import itertools
        all_nodes_columns = [list(itertools.chain(*value.values())) for value in columns_dictionary.values()]
        # for index,(key, value) in enumerate(columns_dictionary.items()):
        #     node_columns = list(itertools.chain(*value.values()))
        #     print(index, key, len(node_columns))
        #     all_nodes_columns.append(node_columns)
        all_nodes_columns = list(itertools.chain(*all_nodes_columns))
        print("Number of query columns:", len(all_nodes_columns))
        return all_nodes_columns
    def _extract_timestamps_and_labels(self):
        print('Extracting timestamps from file', self.pivoted_raw_data_file)
        parquet_file = pq.ParquetFile(self.pivoted_raw_data_file)
        # for i in parquet_file.iter_batches(batch_size=1000):
        #     print("RecordBatch")
        #     print(i.to_pandas())
        sample_batch = next(parquet_file.iter_batches(columns=['interval_start']))
        timestamp_and_label_df = sample_batch.to_pandas()
        timestamp_and_label_df['DateTime'] = pd.to_datetime(timestamp_and_label_df['interval_start'], unit='s')
        timestamp_and_label_df.index = timestamp_and_label_df['DateTime']
        timestamp_and_label_df['is_anomaly'] = 0.0

        anomaly_windows = self.anomaly_windows
        for index, anomaly_window in anomaly_windows.iterrows():
            # Ensure that index is tz-naive before comparison
            anomaly_indices = timestamp_and_label_df[(timestamp_and_label_df['DateTime'] >= anomaly_window['anomaly_window_start']) &
                                                  (timestamp_and_label_df['DateTime'] <= anomaly_window['anomaly_window_end'])].index

            timestamp_and_label_df.loc[anomaly_indices, 'is_anomaly'] = 1.0
            # is_anomaly = ((anomaly_windows['anomaly_window_start'] <= index) &
            #               (index <= anomaly_windows['anomaly_window_end'])).any()
            # timestamp_and_label_df.at[index, 'is_anomaly'] = 1.0 if is_anomaly else 0.0

        timestamp_and_label_file = os.path.join(self.pivoted_raw_data_dir, 'timestamps_and_labels.csv')
        timestamp_and_label_df.to_csv(timestamp_and_label_file, index=False)
        print('Saved timestamps and labels at', timestamp_and_label_file)



    def _extract_total_columns(self):
        print('Extracting total columns from file', self.pivoted_raw_data_file)
        parquet_file = pq.ParquetFile(self.pivoted_raw_data_file)
        # for i in parquet_file.iter_batches(batch_size=1000):
        #     print("RecordBatch")
        #     print(i.to_pandas())

        raw_columns = parquet_file.schema.names
        # sample_batch = next(parquet_file.iter_batches(batch_size=5))
        # sample_df = sample_batch.to_pandas()
        #
        # raw_columns = sample_df.columns.tolist()
        column_list = self.extract_column_details(raw_columns)
        column_list.keys()

        dfs = []
        for key, value in column_list.items():
            print(key, len(value))
            data = pd.DataFrame({
                'column_name': value,
                'type': [key for x in range(len(value))],
                'calculation_type': [self.get_calculation_type(x) for x in value],
                'center': [x.split('_')[0] for x in value],
                'component': [x.split('_')[2] for x in value],
                'endpoint': [x.split('_')[5] for x in value],
                'http_code': [x.split('_')[4] for x in value],
                'host': [x.split('_')[1] for x in value],

            })
            dfs.append(data)
        column_df = pd.concat(dfs, axis=0, ignore_index=True)
        total_column_file_path = os.path.join(self.pivoted_raw_data_dir,'columns.csv')
        print('Saving total columns to file', total_column_file_path)
        column_df.to_csv(total_column_file_path, index=None)
        print('Done saving total columns at', total_column_file_path)
        return column_df


    def _print_details_of_total_columns(self, column_df):
        print('Counting datapoints by host:')
        print(column_df['host'].value_counts())
        print('Columns dataframe shape:', column_df.shape, " with names", column_df.columns.tolist())
        print(tabulate(column_df.head(), headers='keys', tablefmt='psql'))

    def print_details_of_timestamps(self, timestamp_df):
        print(f'Number of samples:{timestamp_df.shape[0]}')
        print(f'Min timestamp: {timestamp_df.index.min().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Max timestamp: {timestamp_df.index.max().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Number of anomaly rows: {timestamp_df["is_anomaly"].sum()}')
    def extract_column_details(self, columns):
        # Find out the number of columns (features) for various groups of status codes,
        # namely -1, 2XX, 3XX, 4XX, and 5XX
        import re
        # Define the regex patterns to search for
        patterns = ["_-1_endpoint", "_2\\d\\d_endpoint", "_3\\d\\d_endpoint", "_4\\d\\d_endpoint", "_5\\d\\d_endpoint"]
        # columns_grouped_by_code = dict()
        # Initialize a dictionary to store the counts for each pattern
        #     pattern_counts = {pattern: 0 for pattern in patterns}
        columns_grouped_by_code = {pattern: [] for pattern in patterns}

        # Count the number of column names that match each pattern individually
        for col in columns:
            for pattern in patterns:
                matched_columns = columns_grouped_by_code[pattern]
                if re.search(pattern, col):
                    #                 pattern_counts[pattern] += 1
                    matched_columns.append(col)

        return columns_grouped_by_code

    def get_calculation_type(self, columns_name: str):
        return columns_name[columns_name.rfind('_') + 1:]

    def get_num_node_features(self):
        return self.num_node_features


    def get_index_dataset(self, window_size=6, batch_size=8, shuffle=False, device='cpu'):
        self._get_edges_and_weights()
        self._generate_task(window_size)
        # dataset_train = StaticGraphTemporalSignal(
        #     self.edges, self.edge_weights, self.features_train, self.targets_train
        # )
        #
        # dataset_train, dataset_valid = temporal_signal_split(dataset_train, train_ratio=0.8)
        #
        # dataset_test = StaticGraphTemporalSignal(
        #     self.edges, self.edge_weights, self.features_test, self.targets_test
        # )

        train_input = self.features_train  # (27399, 207, 2, 12)
        train_target = self.targets_train  # (27399, 207, 12)
        train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
        train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)  # (B, N, T)
        train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                                   drop_last=False)

        valid_input = self.features_valid  # (27399, 207, 2, 12)
        valid_target = self.targets_valid  # (27399, 207, 12)
        valid_x_tensor = torch.from_numpy(valid_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
        valid_target_tensor = torch.from_numpy(valid_target).type(torch.FloatTensor).to(device)  # (B, N, T)
        valid_dataset_new = torch.utils.data.TensorDataset(valid_x_tensor, valid_target_tensor)
        valid_loader = torch.utils.data.DataLoader(valid_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                                   drop_last=False)

        test_input = self.features_test  # (, 207, 2, 12)
        test_target = self.targets_test  # (, 207, 12)
        self.count_5xx = test_target[:,:,0].sum(axis=-1)
        test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
        test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)  # (B, N, T)
        test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                                  drop_last=False)

        return train_loader, valid_loader, test_loader, torch.LongTensor(self.edges)

# @hydra.main(config_path="../conf", config_name="config.yaml")
# def main(cfg: DictConfig):
#     grouping_mode = cfg.data_preparation_pipeline.graph_representation.grouping_mode
#
#     root_dir = get_project_root()
#     data_dir = os.path.join(root_dir, 'data')
#     # label_data_dir = os.path.join(root_dir, 'data/labels')
#     print(data_dir)
#     # print(label_data_dir)
#     start_date = cfg.train_test_config.experiment_parameters.start_date
#     train_end_date = cfg.train_test_config.experiment_parameters.train_end_date
#     test_start_date = cfg.train_test_config.experiment_parameters.test_start_date
#     end_date = cfg.train_test_config.experiment_parameters.end_date
#
#
#     data_config = dict({
#         "pivoted_raw_data_dir": os.path.join(data_dir,'massaged'),
#         'anomaly_windows_dir': os.path.join(data_dir,'labels'),
#         'start_date': start_date, # Actual Satrt 26 Jan 2024
#         'train_end_date': train_end_date,
#         'test_start_date': test_start_date,
#         'minutes_before': cfg.train_test_config.anomaly_window.minutes_before,
#     #    end_date: '2024-03-02'
#         'end_date': end_date,
#         'features_prep': cfg.data_preparation_pipeline.features_prep,
#     })
#
#     ibm_dataset_loader = IBMDatasetLoader(data_dir, data_config, grouping_mode, cfg)
#
#     # Creating Dataloaders
#
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     batch_size = 32
#     dataset_train, dataset_valid, dataset_test, edge_index = ibm_dataset_loader.get_index_dataset(window_size=6, batch_size=batch_size, device=DEVICE)
#     print("Training dataset infor", len(dataset_train))
#     print("Validation dataset infor", len(dataset_valid))
#     print("Testing dataset infor", len(dataset_test))
#     # print("Training dataset infor",len(dataset_train.features), len(dataset_train.targets))
#     # # print("Validation dataset infor", len(dataset_valid.features), len(dataset_valid.targets))
#     # print("Testing dataset infor", len(dataset_test.features), len(dataset_test.targets))
#
#
#
#     # Making the model
#     class TemporalGNN(torch.nn.Module):
#         def __init__(self, node_features, periods, batch_size):
#             super(TemporalGNN, self).__init__()
#             # Attention Temporal Graph Convolutional Cell
#             self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods,
#                                 batch_size=batch_size)  # node_features=2, periods=12
#             # Equals single-shot prediction
#             self.linear = torch.nn.Linear(32, node_features)
#
#         def forward(self, x, edge_index):
#             """
#             x = Node features for T time steps
#             edge_index = Graph edge indices
#             """
#             x = x.permute(0,2,3,1)
#             h = self.tgnn(x, edge_index)  # x [b, 207, 2, 12]  returns h [b, 207, 12]
#             h = F.relu(h)
#             h = self.linear(h)
#             return h
#
#     print(f'Device: {DEVICE}')
#     model = TemporalGNN(node_features=5, periods=6, batch_size=batch_size).to(DEVICE)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = torch.nn.MSELoss()
#
#     print('Net\'s state_dict:')
#     total_param = 0
#     for param_tensor in model.state_dict():
#         print(param_tensor, '\t', model.state_dict()[param_tensor].size())
#         total_param += np.prod(model.state_dict()[param_tensor].size())
#     print('Net\'s total params:', total_param)
#     # --------------------------------------------------
#     print('Optimizer\'s state_dict:')  # If you notice here the Attention is a trainable parameter
#     for var_name in optimizer.state_dict():
#         print(var_name, '\t', optimizer.state_dict()[var_name])
#
#     # Loading the graph once because it's a static graph
#
#     # for snapshot in dataset_train:
#     #     static_edge_index = snapshot.edge_index.to(DEVICE)
#     #     break
#
#     static_edge_index = edge_index.to(DEVICE)
#
#     # num_of_batch = int(len(dataset_train.features)/8)
#     # dataset_train = DataLoader(dataset_train, batch_size=8)
#
#     # Training the model
#     model.train()
#
#     for epoch in range(50):
#         step = 0
#         loss_list = []
#         for index, (encoder_inputs, labels) in tqdm(enumerate(dataset_train), total=len(dataset_train), desc=f'Training...'):
#             # encoder_inputs = sample.x.to(DEVICE)
#             # labels = sample.y.to(DEVICE)
#             y_hat = model(encoder_inputs, static_edge_index)  # Get model predictions
#             loss = loss_fn(y_hat,
#                            labels)  # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             step = step + 1
#             loss_list.append(loss.item())
#             # if step % 100 == 0:
#             #     print(sum(loss_list) / len(loss_list))
#         print("Epoch {} train RMSE: {:.7f}".format(epoch, sum(loss_list) / len(loss_list)))
#
#     model._save_to_state_dict('')
#
#     model.eval()
#     step = 0
#     # Store for analysis
#     total_loss = []
#     for encoder_inputs, labels in dataset_test:
#         # Get model predictions
#         y_hat = model(encoder_inputs, static_edge_index)
#         # Mean squared error
#         loss = loss_fn(y_hat, labels)
#         total_loss.append(loss.item())
#         # Store for analysis below
#         # test_labels.append(labels)
#         # predictions.append(y_hat)
#
# if __name__ == "__main__":
#     main()