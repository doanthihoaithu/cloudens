import ast
import itertools
import json

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from utils import get_project_root


def plot_training_history(model_name, training_history, model_save_dir):
    epochs = training_history['epochs']
    train_losses = training_history['train_losses']
    valid_losses = training_history['valid_losses']

    plt.plot(range(epochs), train_losses, color='blue', label='train loss')
    plt.plot(range(epochs), valid_losses, color='red', label='val loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} training history')

    os.makedirs(model_save_dir, exist_ok=True)
    png_filepath = os.path.join(model_save_dir, f'{model_name}_train_val_losses.png')
    if os.path.exists(png_filepath):
        os.remove(png_filepath)  # Remove the file if it already exists

    plt.savefig(png_filepath, dpi=300)

    training_time_file_path = os.path.join(model_save_dir, f'{model_name}_training_time.csv')
    pd.DataFrame(data=dict({'training_time': [training_history['training_time']], 'epochs': [epochs]})).to_csv(training_time_file_path)
    print('Saved training history to', png_filepath)
    print('Saved training time to ', training_time_file_path)
def plot_results(results_df, is_anomalies, anomaly_windows, result_directory, file_name_prefix='', raw_nab_score=None,
                 normalized_nab_score=None, model='ANN'):
    """
    Plots the anomaly detection results, highlighting ground truth anomalies and predicted anomalies.

    Parameters:
    - results_df (DataFrame): DataFrame containing anomaly results and 5XX count data.
    - is_anomalies (Series): Binary series indicating predicted anomalies.
    - anomaly_windows (DataFrame): DataFrame containing ground truth anomaly windows.
    - result_directory (str): Directory to save the plot.
    - file_name_prefix (str): Prefix for the saved file name.
    - raw_nab_score (float): Raw NAB score for the results.
    - normalized_nab_score (float): Normalized NAB score for the results.
    - model (str): Model name used for anomaly detection.
    """
    # Create a figure for the plot
    plt.figure(figsize=(20, 8))

    # Normalize the 5XX count column to a range between 0 and 1 for better visualization
    results_df['5XX_count_normalized'] = (
            (results_df['5XX_count'] - results_df['5XX_count'].min()) /
            (results_df['5XX_count'].max() - results_df['5XX_count'].min())
    )

    # Plot normalized 5XX counts
    plt.plot(results_df.index, results_df['5XX_count_normalized'], label='5XX Count (Normalized)', color='blue',
             alpha=0.6)

    # Set axis labels with appropriate font sizes
    plt.ylabel('5XX Count (Normalized)', fontsize=14)
    plt.xlabel('Time', fontsize=14)

    # # Convert anomaly window columns to datetime format
    # Create a copy to avoid SettingWithCopyWarning
    anomaly_windows = anomaly_windows.copy()

    # Convert anomaly window columns to datetime format
    anomaly_windows.loc[:, 'anomaly_window_start'] = pd.to_datetime(anomaly_windows['anomaly_window_start'])
    anomaly_windows.loc[:, 'anomaly_window_end'] = pd.to_datetime(anomaly_windows['anomaly_window_end'])

    # Map source labels to anomaly labels
    # Map numeric anomaly sources to human-readable labels
    source_map = {1: 'IssueTracker', 2: 'InstantMessenger', 3: 'TestLog'}
    anomaly_windows.loc[:, 'anomaly_label'] = anomaly_windows['anomaly_source'].map(source_map)

    # Plot ground truth anomalies using shaded areas with different colors for each source
    anomaly_sources = ['IssueTracker', 'InstantMessenger', 'TestLog']
    colors = ['green', 'orange', 'purple']
    legend_labels_used = set()

    for source, color in zip(anomaly_sources, colors):
        source_anomalies = anomaly_windows[anomaly_windows['anomaly_label'] == source]
        for _, row in source_anomalies.iterrows():
            label = f'{source} Anomaly' if source not in legend_labels_used else ''
            plt.axvspan(row['anomaly_window_start'], row['anomaly_window_end'], color=color, alpha=0.1, label=label)
            legend_labels_used.add(source)

    # Mark predicted anomalies with red 'X' markers
    plt.scatter(
        results_df.index[is_anomalies == 1],
        results_df['5XX_count_normalized'][is_anomalies == 1],
        color='red', marker='x', label='Predicted Anomalies'
    )

    # Set the plot title and include NAB scores if available
    title = f'Anomaly Detection Results - {model}'
    if raw_nab_score is not None and normalized_nab_score is not None:
        title += f'\nRaw NAB Score: {raw_nab_score:.2f}, Normalized NAB Score: {normalized_nab_score:.2f}'
    plt.title(title, fontsize=16)

    # Add a legend with a customized font size
    plt.legend(loc='upper right', fontsize=12)

    # Customize tick font sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add gridlines to the plot
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file in the specified directory
    png_filepath = os.path.join(result_directory, f'{file_name_prefix}_{model}_anomaly_detection_results.png')
    if os.path.exists(png_filepath):
        os.remove(png_filepath)  # Remove the file if it already exists
    plt.savefig(png_filepath, dpi=300)  # Save with high DPI for better image quality

    # Show the plot
    plt.show()
def generate_figure_for_fpr(plotting_config):
    models = plotting_config.false_positive_rate.models
    model_name_1 = 'A3TGCN' if 'A3TGCN' in models else 'GRU'
    model_name_2 = 'GRU' if 'GRU' in models else 'GRU'
    http_codes = plotting_config.false_positive_rate.subsets.http_codes
    aggregations = plotting_config.false_positive_rate.subsets.aggregations
    null_padding_features = plotting_config.null_padding_features
    null_padding_targets = [False]
    fill_nan_values= plotting_config.fill_nan_values
    train_models_dir = plotting_config.model_save_path
    window_size = plotting_config.slide_win

    nab_profiles = plotting_config.nab_profiles

    total_graph_dfs = []
    total_gru_dfs = []

    for code, aggregation, fill_nan_value, null_padding_feature, null_padding_target, nab_profile in itertools.product(http_codes, aggregations,
                                                                                                           fill_nan_values,
                                                                                                           null_padding_features,
                                                                                                           null_padding_targets, nab_profiles):
        y_right_label = 'Normalized NAB Reward Low FN Score' if nab_profile == 'low_fn' else 'Normalized NAB Standard Score'

        feature_subset = f'no_group_{code}_{aggregation}'
        # model_name_1 = 'A3TGCN'
        # model_name_2 = 'GRU'
        model_config_1 = f'{model_name_1}'
        if null_padding_feature == True and null_padding_target == True:
            model_config_1 = f'{model_name_1}_null_padding_both'
        elif null_padding_feature == True and null_padding_target == False:
            model_config_1 = f'{model_name_1}_null_padding_feature'
        elif null_padding_feature == False and null_padding_target == True:
            model_config_1 = f'{model_name_1}_null_padding_target'
        else:
            model_config_1 = f'{model_name_1}'
        # if not os.path.exists(os.path.join(train_models_dir, f'window_{window_size}', feature_subset, model_config_1)):
        #     model_config_1 = f'{model_name_1}'

        gridsearch_file_name_1 = f'{model_name_1}_grid_search.csv'
        gridsearch_file_name_2 = f'{model_name_2}_grid_search.csv'


        gridsearch_file_1 = os.path.join(get_project_root(), train_models_dir,
                                         f'window_{window_size}',
                                         feature_subset,
                                         f'fill_nan_with_{fill_nan_value}',
                                         model_config_1,
                                         gridsearch_file_name_1)
        gridsearch_file_2 = os.path.join(get_project_root(), train_models_dir,
                                         f'window_{window_size}',
                                         feature_subset,
                                         f'fill_nan_with_{fill_nan_value}',
                                         model_name_2,
                                         gridsearch_file_name_2)

        if (not os.path.exists(gridsearch_file_1)) or (not os.path.exists(gridsearch_file_2)):
            continue

        print(gridsearch_file_1)
        print(gridsearch_file_2)
        cloudens_grid_search_df_1 = pd.read_csv(gridsearch_file_1)
        training_time_df = pd.read_csv(os.path.join(os.path.dirname(gridsearch_file_1), f'{model_name_1}_training_time.csv'))
        inference_time_df = pd.read_csv(os.path.join(os.path.dirname(gridsearch_file_1), f'inference_time.csv'))
        cloudens_grid_search_df_1['model'] = model_name_1
        cloudens_grid_search_df_1['fill_nan_value'] = fill_nan_value
        cloudens_grid_search_df_1['null_padding_feature'] = null_padding_feature
        cloudens_grid_search_df_1['null_padding_target'] = null_padding_target
        cloudens_grid_search_df_1['http_code'] = code
        cloudens_grid_search_df_1['aggregation'] = aggregation
        cloudens_grid_search_df_1['training_time'] = training_time_df['training_time'].values[0]
        cloudens_grid_search_df_1['inference_time'] = inference_time_df['inference_time'].values[0]

        cloudens_grid_search_df_2 = pd.read_csv(gridsearch_file_2)
        cloudens_grid_search_df_2['model'] = model_name_2
        cloudens_grid_search_df_2['fill_nan_value'] = fill_nan_value
        cloudens_grid_search_df_2['http_code'] = code
        cloudens_grid_search_df_2['aggregation'] = aggregation
        training_time_df = pd.read_csv(
            os.path.join(os.path.dirname(gridsearch_file_2), f'{model_name_2}_training_time.csv'))
        inference_time_df = pd.read_csv(os.path.join(os.path.dirname(gridsearch_file_2), f'inference_time.csv'))
        cloudens_grid_search_df_2['training_time'] = training_time_df['training_time'].values[0]
        cloudens_grid_search_df_2['inference_time'] = inference_time_df['inference_time'].values[0]

        cloudens_grid_search_df = pd.concat([cloudens_grid_search_df_1, cloudens_grid_search_df_2], axis=0,
                                            ignore_index=True)

        total_graph_dfs.append(cloudens_grid_search_df_1)
        total_gru_dfs.append(cloudens_grid_search_df_2)

        confusion_matrix_list = cloudens_grid_search_df['confusion_matrix'].str[1:-1].tolist()
        tn_list, fp_list, fn_list, tp_list = [], [], [], []
        for f in confusion_matrix_list:
            tn, fp, fn, tp = np.fromstring(f, dtype=int, sep=', ')
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            tp_list.append(tp)
        tn_list = np.array(tn_list)
        fp_list = np.array(fp_list)
        fn_list = np.array(fn_list)
        tp_list = np.array(tp_list)
        cloudens_grid_search_df[['tn', 'fp', 'fn', 'tp']] = np.stack([tn_list, fp_list, fn_list, tp_list], axis=1)
        cloudens_grid_search_df['false_positive_rate'] = (1 - cloudens_grid_search_df['tp'] / (
                    cloudens_grid_search_df['fp'] + cloudens_grid_search_df['tp'])) * 100
        cloudens_mahalanobis_df = cloudens_grid_search_df[
            cloudens_grid_search_df['post_processing_strategy'] == 'mahalanobis']
        cloudens_likelihood_df = cloudens_grid_search_df[
            cloudens_grid_search_df['post_processing_strategy'] == 'likelihood']
        sorted_mahalanobis_df = cloudens_mahalanobis_df.sort_values(by='anomaly_threshold', ascending=True, inplace=False)
        sorted_likelihood_df = cloudens_likelihood_df.sort_values(by='anomaly_threshold', ascending=True, inplace=False)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 5),
                                 # gridspec_kw={'width_ratios': [5, 5]}
                                 )
        fig.suptitle(f'FPR vs NAB Score on the log subset "{code} {aggregation}"', fontsize=12, y=0.0)

        plt.subplots_adjust(wspace=0.25)

        ax1 = axes[0]
        ax3 = axes[1]

        x1 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'A3TGCN']['anomaly_threshold'].values
        y1 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'A3TGCN']['false_positive_rate'].values
        y2 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'A3TGCN']['reward_fn_normalized'].values
        y3 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'A3TGCN']['standard_normalized'].values

        y12 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'GRU']['false_positive_rate'].values
        y22 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'GRU']['reward_fn_normalized'].values
        y32 = sorted_mahalanobis_df[sorted_mahalanobis_df['model'] == 'GRU']['standard_normalized'].values

        ax1.plot(x1, y1, 'o-', label='A3TGCN FPR', color='green')
        ax1.plot(x1, y12, 'o--', label='GRU FPR', color='green')
        ax1.legend(loc='upper left')
        ax1.set_xticks(x1)
        ax1.set_xlabel('Mahalanobis Distance Threshold')
        ax1.set_ylabel('False Positive Rate', color='g')
        ax1.tick_params(axis='y', colors="green")
        # ax1.set_ylim([50, 100])
        # ax1.legend(loc='upper left')
        ax1.legend(loc=(0.0, 1.04))

        ax2 = ax1.twinx()
        ax2.plot(x1, y2 if nab_profile == 'low_fn' else y3, 'o-', label='A3TGCN NAB', color='blue')
        ax2.plot(x1, y22 if nab_profile == 'low_fn' else y32, 'o--', label='GRU NAB', color='blue')
        # ax2.plot(x1, y3, '*--', label='A3TGCN Standard', color='blue')
        # ax2.plot(x1, y32, '*--', label='GRU Standard', color='red')
        ax2.set_ylabel(y_right_label, color='b')
        ax2.tick_params(axis='y', colors="blue")
        # ax2.set_ylim([5, 30])
        # ax2.legend(loc='best')
        ax2.legend(loc=(0.8, 1.04))
        x_threshold = x1[y2.argmax()] if nab_profile == 'low_fn' else x1[y3.argmax()]
        ax2.axvline(x=x_threshold, color='b', linestyle='--')
        ax2.annotate(' Selected threshold', xy=(x_threshold, ax2.get_ylim()[-1] * 0.95), color='b')

        x2 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'A3TGCN']['anomaly_threshold'].iloc[1:].values
        y4 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'A3TGCN']['false_positive_rate'].iloc[1:].values
        y5 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'A3TGCN']['reward_fn_normalized'].iloc[1:].values
        y6 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'A3TGCN']['standard_normalized'].iloc[1:].values

        y42 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'GRU']['false_positive_rate'].iloc[1:].values
        y52 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'GRU']['reward_fn_normalized'].iloc[1:].values
        y62 = sorted_likelihood_df[sorted_likelihood_df['model'] == 'GRU']['standard_normalized'].iloc[1:].values

        ax3.plot(x2, y4, 'o-', label='A3TGCN FPR', color='green')
        ax3.plot(x2, y42, 'o--', label='GRU FPR', color='green')
        ax3.legend(loc='upper left')
        ax3.set_xticks(x2)
        ax3.set_xlabel('Likelihood Function Threshold')
        ax3.set_ylabel('False Positive Rate', color='g')
        # ax3.legend(loc='upper left')
        ax3.legend(loc=(0.0, 1.04))
        ax3.tick_params(axis='y', colors="green")
        ax3.tick_params(axis='x', labelsize=7)
        # ax3.set_ylim([87, 94])

        ax4 = ax3.twinx()
        ax4.plot(x2, y5 if nab_profile == 'low_fn' else y6, 'o-', label='A3TGCN NAB', color='blue')
        ax4.plot(x2, y52 if nab_profile == 'low_fn' else y62, 'o--', label='GRU NAB', color='blue')
        # ax4.plot(x2, y6, '*-', label='A3TGCN Standard', color='blue')
        # ax4.plot(x2, y62, '*--', label='GRU Standard', color='blue')
        ax4.set_ylabel(y_right_label, color='b')
        # ax4.legend(loc='upper right')
        ax4.legend(loc=(0.8, 1.04))
        ax4.tick_params(axis='y', colors="blue")
        # ax4.set_xlim([0.9984, 0.99985])
        x_likelihood_threshold = x2[y5.argmax()] if nab_profile == 'low_fn' else x2[y6.argmax()]
        ax4.axvline(x=x_likelihood_threshold, color='b', linestyle='--')
        ax4.annotate(' Selected threshold', xy=(x_likelihood_threshold, ax4.get_ylim()[-1] * 0.95), color='b')
        # ax3.set_xscale('symlog')
        # ax4.sharey(ax2)
        # plt.xticks(x1)
        # fig.tight_layout()
        fig.autofmt_xdate()

        combined_figure_save_dir = os.path.join(get_project_root(),plotting_config.output_dir, f'window_{window_size}')
        os.makedirs(combined_figure_save_dir, exist_ok=True)
        combined_file_name = f'{feature_subset}_fill_nan_with_{fill_nan_value}_{model_config_1}_{model_name_2}_{nab_profile}_combined.png'

        fig.savefig(os.path.join(combined_figure_save_dir, combined_file_name), bbox_inches='tight')
        print(f'Figure saved to {combined_figure_save_dir}')

        path1 = os.path.join(combined_figure_save_dir, f'{feature_subset}_fill_nan_with_{fill_nan_value}_{model_config_1}_{model_name_2}_{nab_profile}_mahalanobis.png' )
        extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(path1, bbox_inches=extent1.expanded(1.3, 1.5))
        print(f'Figure saved to {path1}')

        path2 = os.path.join(combined_figure_save_dir, f'{feature_subset}_fill_nan_with_{fill_nan_value}_{model_config_1}_{model_name_2}_{nab_profile}_likelihood.png' )
        extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(path2, bbox_inches=extent3.expanded(1.3, 1.5))
        print(f'Figure saved to {path2}')

    total_graph_results_df = pd.concat(total_graph_dfs, ignore_index=True)
    total_gru_results_df = pd.concat(total_gru_dfs, ignore_index=True)
    total_gru_results_df.drop_duplicates(inplace=True)
    combined_df = pd.concat([total_graph_results_df, total_gru_results_df], axis=0, ignore_index=True)
    print(f'combined_df shape: {combined_df.shape}')
    anomaly_marks, detected_anomaly_ids = process_detection_counters(
        combined_df['detection_counters'].tolist())

    print(anomaly_marks.shape, detected_anomaly_ids.shape, combined_df.shape)
    combined_df[anomaly_marks] = detected_anomaly_ids
    combined_df.to_csv(os.path.join(combined_figure_save_dir,'combined_grid_search_results.csv'), index=False)
    print(f'Combined grid search results saved to {combined_figure_save_dir}')

def process_detection_counters(detection_counters_list):
    # Example
    # {'issue_detected': 0,
    # 'issue_detected_ids': [],
    # 'im_detected': 3,
    # 'im_detected_ids': [5, 7, 9],
    # 'TestLog_detected': 0,
    # 'TestLog_detected_ids': [],
    # 'gt_issue_ids': [3, 12, 13],
    # 'gt_im_ids': [0, 5, 6, 7, 8, 9, 14, 16, 17],
    # 'gt_TestLog_ids': [1, 2, 4, 10, 11, 15, 18],
    # 'tp': 3,
    # 'tn': 0,
    # 'fp': 123,
    # 'fn': 16}

    print(f'Length of detection counters: {len(detection_counters_list)}')
    dfs = []
    columns = None
    for detection_counters in detection_counters_list:
        detection_counters = ast.literal_eval(detection_counters)
        # print(detection_counters)
        # print(type(detection_counters))
        issue_detected = detection_counters['issue_detected']
        issue_detected_ids = detection_counters['issue_detected_ids']
        im_detected = detection_counters['im_detected']
        im_detected_ids = detection_counters['im_detected_ids']
        TestLog_detected = detection_counters['TestLog_detected']
        TestLog_detected_ids = detection_counters['TestLog_detected_ids']
        gt_issue_ids = detection_counters['gt_issue_ids']
        gt_im_ids = detection_counters['gt_im_ids']
        gt_TestLog_ids = detection_counters['gt_TestLog_ids']
        number_of_anomalies = len(gt_issue_ids) + len(gt_im_ids) + len(gt_TestLog_ids)
        column_names = [f'anomaly_{i}' for i in range(number_of_anomalies)]
        columns = column_names
        df = pd.DataFrame(columns=column_names)
        df[column_names] = np.asarray([False] * number_of_anomalies).reshape(1,-1)
        for anomaly_id in issue_detected_ids:
            df[f'anomaly_{anomaly_id}'] = [True]

        for anomaly_id in im_detected_ids:
            df[f'anomaly_{anomaly_id}'] = [True]

        for anomaly_id in TestLog_detected_ids:
            df[f'anomaly_{anomaly_id}'] = [True]

        dfs.append(df.values.reshape(-1))
    print(f'Length of detection counters: {len(dfs)}')
    dfs = np.array(dfs, dtype=bool)
    print(f'dfs shape after np.concate', dfs.shape)
    final_df = pd.DataFrame(columns=columns)
    final_df[columns] = dfs
    print(f'Length of detection counters after concat: {final_df.shape}')
    return final_df.columns, final_df.values


