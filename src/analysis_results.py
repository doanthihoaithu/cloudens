import ast
import colorsys
import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
from omegaconf import DictConfig

from utils import get_project_root

ONE_COLUMN_FIGURE_WIDTH = 5
TWO_COLUMN_FIGURE_WIDTH = 12

MODEL_DISPLAY_NAME_MAP = {
    'A3TGCN_null_padding_feature': 'ClouDens',
}

SCORING_STRATEGY_DISPLAY_NAME_MAP = {
    'likelihood': 'LF',
    'mahalanobis': 'MD',
}

NAB_PROFILE_DISPLAY_NAME_MAP = {
    'standard': 'Standard',
    'reward_fn': 'Low FN',
}

# Hue (in colorsys HLS space, [0, 1]) assigned to each model family's color
MODEL_FAMILY_HUE_MAP = {
    'A3TGCN': 2 / 3,  # blue
    'GRU': 1 / 3,     # green
}

TITLE_FONT_SIZE = 10
LEGEND_FONT_SIZE = TITLE_FONT_SIZE - 2
TICK_FONT_SIZE = TITLE_FONT_SIZE - 2
AXIS_LABEL_FONT_SIZE = TITLE_FONT_SIZE - 2

NUM_TESTING_ANOMALIES = 19

ANOMALY_GROUP_ID_KEYS = ['issue_detected_ids', 'im_detected_ids', 'TestLog_detected_ids']

# Ground-truth id list (fixed per anomaly index, independent of subset/strategy)
# corresponding to each detected-id key
GROUND_TRUTH_ID_KEY_MAP = {
    'issue_detected_ids': 'gt_issue_ids',
    'im_detected_ids': 'gt_im_ids',
    'TestLog_detected_ids': 'gt_TestLog_ids',
}

ANOMALY_GROUP_DISPLAY_NAME_MAP = {
    'issue_detected_ids': 'Issue Tracker: 3',
    'im_detected_ids': 'Instant Messenger: 9',
    'TestLog_detected_ids': 'Test Log: 7',
}

ANOMALY_GROUP_COLOR_MAP = {
    'issue_detected_ids': 'green',
    'im_detected_ids': 'orange',
    'TestLog_detected_ids': 'violet',
}

UNDETECTED_CELL_COLOR = 'white'

ANOMALY_GROUP_OPACITY = 0.4

DETECTED_ANOMALIES_TABLE_ROW_HEIGHT = 0.3



def text_subset_wrapper(http_code, agg):
    return "\\mathtt{" + f'{http_code}\ {agg}' + "}"

def text_subset_wrapper_latex(http_code, agg):
    return "\\texttt{" + f'{http_code} {agg}' + "}"


def _merge_computation_time_data(results_dir, supported_models, supported_sliding_windows,
                                 imputation_strategies, http_codes, aggregations, graph_models,
                                 null_padding_features,
                                 null_padding_targets,
                                 ):
    index_columns = ['model_name', 'is_graph_model', 'sliding_window', 'http_code', 'aggregation', 'imputation_strategy',
                     'null_padding_feature',
                     'null_padding_target']
    merged_df = pd.DataFrame()
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)

    # Collect times keyed by (http_code, agg) subset → model → window
    subsets = [(hc, agg) for hc in http_codes for agg in aggregations]

    extended_supported_models = []
    for m in supported_models:
        if m not in graph_models:
            extended_supported_models.append(m)
        else:
            for (null_padding_feature, null_padding_target) in itertools.product(null_padding_features,null_padding_targets):
                if null_padding_feature == False and null_padding_target == False:
                    new_name = m
                elif null_padding_feature == True and null_padding_target == False:
                    new_name = f'{m}_null_padding_feature'
                elif null_padding_feature == False and null_padding_target == True:
                    new_name = f'{m}_null_padding_target'
                else:
                    new_name = f'{m}_null_padding_both'
                extended_supported_models.append(new_name)

    train_data = {s: {m: {w: [] for w in supported_sliding_windows} for m in extended_supported_models} for s in subsets}
    infer_data = {s: {m: {w: [] for w in supported_sliding_windows} for m in extended_supported_models} for s in subsets}

    for window in supported_sliding_windows:
        for http_code in http_codes:
            for agg in aggregations:
                subset = (http_code, agg)
                for imputation in imputation_strategies:
                    for model in supported_models:
                        if model not in graph_models:
                            null_padding_combinations = itertools.product([False],[False])
                        else:
                            null_padding_combinations = itertools.product(null_padding_features,null_padding_targets)

                        for (null_padding_feature, null_padding_target) in null_padding_combinations:

                            if model not in graph_models:
                                model_folder = model
                            else:
                                if null_padding_feature == False and null_padding_target == False:
                                    model_folder = model
                                elif null_padding_feature == True and null_padding_target == False:
                                    model_folder = f'{model}_null_padding_feature'
                                elif null_padding_feature == False and null_padding_target == True:
                                    model_folder = f'{model}_null_padding_target'
                                else:
                                    model_folder = f'{model}_null_padding_both'
                            base = dict(
                                model_name=model,
                                is_graph_model=model in graph_models,
                                sliding_window=window,
                                http_code=http_code,
                                aggregation=agg,
                                imputation_strategy=imputation,
                                null_padding_feature=null_padding_feature,
                                null_padding_target=null_padding_target,
                            )
                            training_time_csv_path = os.path.join(
                                get_project_root(),
                                results_dir,
                                f'window_{window}',
                                f'no_group_{http_code}_{agg}',
                                f'fill_nan_with_{imputation}',
                                model_folder,
                                f'{model}_training_time.csv'
                            )
                            inference_time_csv_path = os.path.join(
                                get_project_root(),
                                results_dir,
                                f'window_{window}',
                                f'no_group_{http_code}_{agg}',
                                f'fill_nan_with_{imputation}',
                                model_folder,
                                f'inference_time.csv'
                            )

                            if os.path.exists(training_time_csv_path):
                                df = pd.read_csv(training_time_csv_path)
                                t = df['training_time'].values[0]
                                train_data[subset][model_folder][window].append(t)
                                row_df = pd.DataFrame([{**base, 'training_time': t, 'epochs': df['epochs'].values[0]}])
                                row_df.set_index(index_columns, inplace=True)
                                merged_df = row_df.combine_first(merged_df)

                            if os.path.exists(inference_time_csv_path):
                                inference_df = pd.read_csv(inference_time_csv_path, index_col=0)
                                t = inference_df['inference_time'].values[0]
                                infer_data[subset][model_folder][window].append(t)
                                row_df = pd.DataFrame([{**base, 'inference_time': t}])
                                row_df.set_index(index_columns, inplace=True)
                                merged_df = row_df.combine_first(merged_df)

    csv_saved_path = os.path.join(merged_results_dir, 'computation_time_comparision.csv')
    merged_df.to_csv(csv_saved_path, index=True)
    print(f'Merged CSV saved to {csv_saved_path}')

    return csv_saved_path


def merge_computation_time(results_dir, supported_models, supported_sliding_windows,
                           imputation_strategies, http_codes, aggregations, graph_models,
                           null_padding_features,
                           null_padding_targets,
                           use_existing_file=False,
                           ):
    csv_saved_path = os.path.join(results_dir, 'merged_results', 'computation_time_comparision.csv')
    if use_existing_file and os.path.exists(csv_saved_path):
        print(f'Reusing existing computation time CSV at {csv_saved_path}')
        return csv_saved_path

    return _merge_computation_time_data(results_dir, supported_models, supported_sliding_windows,
                                        imputation_strategies, http_codes, aggregations, graph_models,
                                        null_padding_features, null_padding_targets)

def plot_computation_time(results_dir, supported_models, supported_sliding_windows,
                          http_codes, aggregations, graph_models,
                          null_padding_features, null_padding_targets,
                          is_one_column_figure=False):
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    csv_path = os.path.join(merged_results_dir, 'computation_time_comparision.csv')
    if not os.path.exists(csv_path):
        print(f'CSV not found at {csv_path}; run merge_computation_time first')
        return None

    df = pd.read_csv(csv_path)

    # Average training/inference time over missing_imputation_strategies (column
    # 'imputation_strategy') for each (model, null padding, subset, window) combination
    group_cols = ['model_name', 'null_padding_feature', 'null_padding_target',
                  'http_code', 'aggregation', 'sliding_window']
    df = df.groupby(group_cols, as_index=False)[['training_time', 'inference_time']].mean()

    # Reconstruct extended model list (same ordering as merge_computation_time)
    extended_supported_models = []
    for m in supported_models:
        if m not in graph_models:
            extended_supported_models.append(m)
        else:
            for (npf, npt) in itertools.product(null_padding_features, null_padding_targets):
                if not npf and not npt:
                    new_name = m
                elif npf and not npt:
                    new_name = f'{m}_null_padding_feature'
                elif not npf and npt:
                    new_name = f'{m}_null_padding_target'
                else:
                    new_name = f'{m}_null_padding_both'
                extended_supported_models.append(new_name)

    subsets = [(hc, agg) for hc in http_codes for agg in aggregations]
    train_data = {s: {m: {w: [] for w in supported_sliding_windows} for m in extended_supported_models} for s in subsets}
    infer_data = {s: {m: {w: [] for w in supported_sliding_windows} for m in extended_supported_models} for s in subsets}

    for _, row in df.iterrows():
        model_name = row['model_name']
        npf = str(row['null_padding_feature']).strip().lower() == 'true'
        npt = str(row['null_padding_target']).strip().lower() == 'true'
        http_code = row['http_code']
        agg = row['aggregation']
        window = int(row['sliding_window'])
        subset = (http_code, agg)

        if subset not in train_data:
            continue

        if model_name not in graph_models:
            model_folder = model_name
        else:
            if not npf and not npt:
                model_folder = model_name
            elif npf and not npt:
                model_folder = f'{model_name}_null_padding_feature'
            elif not npf and npt:
                model_folder = f'{model_name}_null_padding_target'
            else:
                model_folder = f'{model_name}_null_padding_both'

        if model_folder not in train_data[subset]:
            continue
        if window not in train_data[subset][model_folder]:
            continue

        if 'training_time' in row and pd.notna(row['training_time']):
            train_data[subset][model_folder][window].append(float(row['training_time']))
        if 'inference_time' in row and pd.notna(row['inference_time']):
            infer_data[subset][model_folder][window].append(float(row['inference_time']))

    n_rows = len(subsets)
    n_windows = len(supported_sliding_windows)
    bar_width = 0.7 / n_windows
    colors = plt.cm.tab10(np.linspace(0, 0.45, n_windows))

    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, axes = plt.subplots(n_rows, 2, figsize=(figure_width, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (http_code, agg) in enumerate(subsets):
        subset = (http_code, agg)
        ax_train = axes[row_idx, 0]
        ax_infer = axes[row_idx, 1]

        train_means = {
            m: {w: np.mean(vs) if vs else np.nan for w, vs in train_data[subset][m].items()}
            for m in extended_supported_models
        }
        infer_means = {
            m: {w: np.mean(vs) if vs else np.nan for w, vs in infer_data[subset][m].items()}
            for m in extended_supported_models
        }

        valid_models = [m for m in extended_supported_models
                        if not all(np.isnan(v) for v in train_means[m].values())]
        x = np.arange(len(valid_models))

        for i, window in enumerate(supported_sliding_windows):
            offset = (i - n_windows / 2 + 0.5) * bar_width
            ax_train.bar(x + offset, [train_means[m][window] for m in valid_models],
                         bar_width, label=f'Win {window}', color=colors[i])
            ax_infer.bar(x + offset, [infer_means[m][window] for m in valid_models],
                         bar_width, label=f'Win {window}', color=colors[i])

        row_label = f'{http_code} / {agg}'
        for ax, ylabel in [(ax_train, 'Training Time (s)\n[log scale]'),
                           (ax_infer, 'Inference Time (s)\n[log scale]')]:
            ax.set_xticks(x)
            display_labels = [MODEL_DISPLAY_NAME_MAP.get(m, m) for m in valid_models]
            ax.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=TICK_FONT_SIZE)
            ax.set_yscale('log')
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
            ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
            ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

        ax_train.set_title(f'Training Time — {row_label}', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax_infer.set_title(f'Inference Time — {row_label}', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f'Win {w}')
        for i, w in enumerate(supported_sliding_windows)
    ]
    fig.legend(handles=legend_handles, title='Window', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
               loc='center left', bbox_to_anchor=(1.0, 0.5),
               ncol=1, frameon=True)

    fig.tight_layout(rect=(0, 0, 0.9, 1))
    out_path = os.path.join(merged_results_dir, 'computation_time_comparision.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Computation time plot saved to {out_path}')
    return out_path


def plot_computation_time_combined(results_dir, supported_models, supported_sliding_windows,
                                   http_codes, aggregations, graph_models,
                                   null_padding_features, null_padding_targets,
                                   is_one_column_figure=False):
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    csv_path = os.path.join(merged_results_dir, 'computation_time_comparision.csv')
    if not os.path.exists(csv_path):
        print(f'CSV not found at {csv_path}; run merge_computation_time_combined first')
        return None

    df = pd.read_csv(csv_path)

    # Average training/inference time over missing_imputation_strategies (column
    # 'imputation_strategy') for each (model, null padding, subset, window) combination
    group_cols = ['model_name', 'null_padding_feature', 'null_padding_target',
                  'http_code', 'aggregation', 'sliding_window']
    df = df.groupby(group_cols, as_index=False)[['training_time', 'inference_time']].mean()

    # Reconstruct extended model list (same ordering as merge_computation_time)
    extended_supported_models = []
    for m in supported_models:
        if m not in graph_models:
            extended_supported_models.append(m)
        else:
            for (npf, npt) in itertools.product(null_padding_features, null_padding_targets):
                if not npf and not npt:
                    new_name = m
                elif npf and not npt:
                    new_name = f'{m}_null_padding_feature'
                elif not npf and npt:
                    new_name = f'{m}_null_padding_target'
                else:
                    new_name = f'{m}_null_padding_both'
                extended_supported_models.append(new_name)

    # Unlike plot_computation_time, every (model, subset) pair gets its own
    # x-axis position within the same pair of subplots
    subsets = [(hc, agg) for hc in http_codes for agg in aggregations]
    pairs = [(subset, m) for subset in subsets for m in extended_supported_models]
    train_data = {p: {w: [] for w in supported_sliding_windows} for p in pairs}
    infer_data = {p: {w: [] for w in supported_sliding_windows} for p in pairs}

    for _, row in df.iterrows():
        model_name = row['model_name']
        npf = str(row['null_padding_feature']).strip().lower() == 'true'
        npt = str(row['null_padding_target']).strip().lower() == 'true'
        http_code = row['http_code']
        agg = row['aggregation']
        window = int(row['sliding_window'])
        subset = (http_code, agg)

        if model_name not in graph_models:
            model_folder = model_name
        else:
            if not npf and not npt:
                model_folder = model_name
            elif npf and not npt:
                model_folder = f'{model_name}_null_padding_feature'
            elif not npf and npt:
                model_folder = f'{model_name}_null_padding_target'
            else:
                model_folder = f'{model_name}_null_padding_both'

        pair = (subset, model_folder)
        if pair not in train_data:
            continue
        if window not in train_data[pair]:
            continue

        if 'training_time' in row and pd.notna(row['training_time']):
            train_data[pair][window].append(float(row['training_time']))
        if 'inference_time' in row and pd.notna(row['inference_time']):
            infer_data[pair][window].append(float(row['inference_time']))

    n_windows = len(supported_sliding_windows)
    bar_width = 0.7 / n_windows
    colors = plt.cm.tab10(np.linspace(0, 0.45, n_windows))

    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, (ax_train, ax_infer) = plt.subplots(1, 2, figsize=(figure_width, 3))

    train_means = {
        p: {w: np.mean(vs) if vs else np.nan for w, vs in train_data[p].items()}
        for p in pairs
    }
    infer_means = {
        p: {w: np.mean(vs) if vs else np.nan for w, vs in infer_data[p].items()}
        for p in pairs
    }

    valid_pairs = [p for p in pairs if not all(np.isnan(v) for v in train_means[p].values())]
    x = np.arange(len(valid_pairs))

    def format_bar_value(v):
        return '' if np.isnan(v) else f'{v:.1f}'

    for i, window in enumerate(supported_sliding_windows):
        offset = (i - n_windows / 2 + 0.5) * bar_width
        train_values = [train_means[p][window] for p in valid_pairs]
        infer_values = [infer_means[p][window] for p in valid_pairs]
        train_bars = ax_train.bar(x + offset, train_values,
                                  bar_width, label=f'Win {window}', color=colors[i], alpha=0.8,
                                  # edgecolor=colors[i]
                                  )
        infer_bars = ax_infer.bar(x + offset, infer_values,
                                  bar_width, label=f'Win {window}', color=colors[i], alpha=0.8,
                                  # edgecolor=colors[i]
                                  )

        # Label bars with their actual (pre-log-scaling) value, outside the bar
        ax_train.bar_label(train_bars, labels=[format_bar_value(v) for v in train_values],
                           fontsize=TICK_FONT_SIZE - 3, rotation=90, padding=2,
                           label_type='edge')
        ax_infer.bar_label(infer_bars, labels=[format_bar_value(v) for v in infer_values],
                           fontsize=TICK_FONT_SIZE - 3, rotation=90, padding=2,
                           label_type='edge')

    display_labels = [
        f'{MODEL_DISPLAY_NAME_MAP.get(m, m)}\n${text_subset_wrapper(http_code, agg)}$'
        for ((http_code, agg), m) in valid_pairs
    ]
    for ax, ylabel in [(ax_train, 'Training Time (s) [log scale]'),
                       (ax_infer, 'Inference Time (s) [log scale]')]:
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=TICK_FONT_SIZE)
        ax.set_yscale('log')
        # ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 10 * 0.3)

    ax_train.set_title('Training Time (s) [log scale]', fontsize=TITLE_FONT_SIZE,
                       # fontweight='bold'
                       )
    ax_infer.set_title('Inference Time (s) [log scale]', fontsize=TITLE_FONT_SIZE,
                       # fontweight='bold'
                       )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f'{w}', alpha=0.8)
        for i, w in enumerate(supported_sliding_windows)
    ]
    fig.legend(handles=legend_handles,
               title='Window',
               fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=n_windows, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.subplots_adjust(wspace=0.15)
    out_path = os.path.join(merged_results_dir, 'computation_time_comparision_combined.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Combined computation time plot saved to {out_path}')
    return out_path


def load_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        proposed_model_detail, results_dir, supported_sliding_windows,
        http_codes, aggregations, missing_imputation_strategies,
        use_existing_file=False,
):
    model_name = proposed_model_detail['proposed_model_name']
    null_padding_feature = proposed_model_detail['null_padding_feature']
    null_padding_target = proposed_model_detail['null_padding_target']
    is_graph = proposed_model_detail['is_graph']

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    csv_saved_path = os.path.join(merged_results_dir, f'{model_name}_optimal_scoring_hyperparameters.csv')
    if use_existing_file and os.path.exists(csv_saved_path):
        print(f'Reusing existing optimal scoring hyperparameters CSV at {csv_saved_path}')
        return pd.read_csv(csv_saved_path)

    if not is_graph:
        model_folder = model_name
    elif null_padding_feature and not null_padding_target:
        model_folder = f'{model_name}_null_padding_feature'
    elif not null_padding_feature and null_padding_target:
        model_folder = f'{model_name}_null_padding_target'
    elif null_padding_feature and null_padding_target:
        model_folder = f'{model_name}_null_padding_both'
    else:
        model_folder = model_name

    nab_profiles = ['standard', 'reward_fn']
    scoring_strategies = ['likelihood', 'mahalanobis']

    records = []
    for window in supported_sliding_windows:
        for http_code in http_codes:
            for agg in aggregations:
                for imputation in missing_imputation_strategies:
                    grid_search_csv_path = os.path.join(
                        results_dir,
                        f'window_{window}',
                        f'no_group_{http_code}_{agg}',
                        f'fill_nan_with_{imputation}',
                        model_folder,
                        f'{model_name}_grid_search.csv'
                    )
                    if not os.path.exists(grid_search_csv_path):
                        continue

                    df = pd.read_csv(grid_search_csv_path)

                    for strategy in scoring_strategies:
                        strategy_df = df[df['post_processing_strategy'] == strategy]
                        if strategy_df.empty:
                            continue

                        for profile in nab_profiles:
                            normalized_column = f'{profile}_normalized'
                            # NAB_{profile}_rank in the grid search csv ranks rows across all
                            # scoring strategies combined, so it is recomputed here scoped to
                            # this strategy's rows; the top row is thus guaranteed rank == 1.
                            strategy_rank = strategy_df[normalized_column].rank(ascending=False)
                            best_idx = strategy_rank.idxmin()
                            best_row = strategy_df.loc[best_idx]
                            records.append(dict(
                                sliding_window=window,
                                http_code=http_code,
                                aggregation=agg,
                                imputation_strategy=imputation,
                                nab_profile=profile,
                                post_processing_strategy=strategy,
                                long_window=best_row['long_window'],
                                short_window=best_row['short_window'],
                                anomaly_threshold=best_row['anomaly_threshold'],
                                topk=best_row['topk'],
                                overal_rank=best_row[f'NAB_{profile}_rank'],
                                standard_normalized=best_row['standard_normalized'],
                                reward_fn_normalized=best_row['reward_fn_normalized'],
                                confusion_matrix=best_row['confusion_matrix'],
                                detection_counters=best_row['detection_counters'],
                                precision=best_row['precision'],
                                recall=best_row['recall'],
                                f1=best_row['f1'],
                                accuracy=best_row['accuracy'],
                                standard_raw=best_row['standard_raw'],
                                reward_fn_raw=best_row['reward_fn_raw'],
                            ))

    optimal_hyperparameters_df = pd.DataFrame(records)

    os.makedirs(merged_results_dir, exist_ok=True)
    optimal_hyperparameters_df.to_csv(csv_saved_path, index=False)
    print(f'Optimal scoring hyperparameters CSV saved to {csv_saved_path}')

    return optimal_hyperparameters_df


def load_shorten_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        proposed_model_detail, results_dir, supported_sliding_windows,
        http_codes, aggregations, missing_imputation_strategies,
        use_existing_file=False,
):
    model_name = proposed_model_detail['proposed_model_name']
    null_padding_feature = proposed_model_detail['null_padding_feature']
    null_padding_target = proposed_model_detail['null_padding_target']
    is_graph = proposed_model_detail['is_graph']

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    csv_saved_path = os.path.join(merged_results_dir, f'{model_name}_shorten_optimal_scoring_hyperparameters.csv')
    if use_existing_file and os.path.exists(csv_saved_path):
        print(f'Reusing existing shorten optimal scoring hyperparameters CSV at {csv_saved_path}')
        return pd.read_csv(csv_saved_path)

    if not is_graph:
        model_folder = model_name
    elif null_padding_feature and not null_padding_target:
        model_folder = f'{model_name}_null_padding_feature'
    elif not null_padding_feature and null_padding_target:
        model_folder = f'{model_name}_null_padding_target'
    elif null_padding_feature and null_padding_target:
        model_folder = f'{model_name}_null_padding_both'
    else:
        model_folder = model_name

    nab_profiles = ['reward_fn']
    scoring_strategies = ['likelihood', 'mahalanobis']

    records = []
    for window in supported_sliding_windows:
        for http_code in http_codes:
            for agg in aggregations:
                # best (normalized_value, record) seen so far per (strategy, profile),
                # compared across imputation strategies so only the winning one is kept
                best_candidates = {}

                for imputation in missing_imputation_strategies:
                    grid_search_csv_path = os.path.join(
                        results_dir,
                        f'window_{window}',
                        f'no_group_{http_code}_{agg}',
                        f'fill_nan_with_{imputation}',
                        model_folder,
                        f'{model_name}_grid_search.csv'
                    )
                    if not os.path.exists(grid_search_csv_path):
                        continue

                    df = pd.read_csv(grid_search_csv_path)

                    for strategy in scoring_strategies:
                        strategy_df = df[df['post_processing_strategy'] == strategy]
                        if strategy_df.empty:
                            continue

                        for profile in nab_profiles:
                            normalized_column = f'{profile}_normalized'
                            strategy_rank = strategy_df[normalized_column].rank(ascending=False)
                            best_idx = strategy_rank.idxmin()
                            best_row = strategy_df.loc[best_idx]
                            normalized_value = best_row[normalized_column]

                            key = (strategy, profile)
                            if key in best_candidates and best_candidates[key][0] >= normalized_value:
                                continue

                            best_candidates[key] = (normalized_value, dict(
                                sliding_window=window,
                                http_code=http_code,
                                aggregation=agg,
                                imputation_strategy=imputation,
                                nab_profile=profile,
                                post_processing_strategy=strategy,
                                long_window=best_row['long_window'],
                                short_window=best_row['short_window'],
                                anomaly_threshold=best_row['anomaly_threshold'],
                                topk=best_row['topk'],
                                overal_rank=best_row[f'NAB_{profile}_rank'],
                                standard_normalized=best_row['standard_normalized'],
                                reward_fn_normalized=best_row['reward_fn_normalized'],
                                confusion_matrix=best_row['confusion_matrix'],
                                detection_counters=best_row['detection_counters'],
                                precision=best_row['precision'],
                                recall=best_row['recall'],
                                f1=best_row['f1'],
                                accuracy=best_row['accuracy'],
                                standard_raw=best_row['standard_raw'],
                                reward_fn_raw=best_row['reward_fn_raw'],
                            ))

                records.extend(record for _, record in best_candidates.values())

    shorten_optimal_hyperparameters_df = pd.DataFrame(records)

    os.makedirs(merged_results_dir, exist_ok=True)
    shorten_optimal_hyperparameters_df.to_csv(csv_saved_path, index=False)
    print(f'Shorten optimal scoring hyperparameters CSV saved to {csv_saved_path}')

    return shorten_optimal_hyperparameters_df


def extract_most_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        optimal_hyperparameters_df, results_dir, proposed_model_detail,
):
    model_name = proposed_model_detail['proposed_model_name']
    scoring_strategies = ['likelihood', 'mahalanobis']

    reward_fn_df = optimal_hyperparameters_df[optimal_hyperparameters_df['nab_profile'] == 'reward_fn']

    records = []
    group_cols = ['http_code', 'aggregation', 'sliding_window']
    for _, group_df in reward_fn_df.groupby(group_cols):
        if group_df.empty:
            continue

        # Optimal imputation for this <subset, sliding_window> pair is whichever
        # row (across every imputation_strategy and scoring strategy) scores
        # highest on reward_fn_normalized
        best_row = group_df.loc[group_df['reward_fn_normalized'].idxmax()]
        optimal_imputation = best_row['imputation_strategy']

        for strategy in scoring_strategies:
            strategy_rows = group_df[
                (group_df['imputation_strategy'] == optimal_imputation) &
                (group_df['post_processing_strategy'] == strategy)
            ]
            if strategy_rows.empty:
                continue
            records.append(strategy_rows.iloc[0].to_dict())

    most_optimal_hyperparameters_df = pd.DataFrame(records)

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    csv_saved_path = os.path.join(merged_results_dir, f'{model_name}_most_optimal_scoring_hyperparameters.csv')
    most_optimal_hyperparameters_df.to_csv(csv_saved_path, index=False)
    print(f'Most optimal scoring hyperparameters CSV saved to {csv_saved_path}')

    return most_optimal_hyperparameters_df


def _format_trimmed(value, decimals):
    formatted = f'{value:.{decimals}f}'
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


# anomaly_threshold lives on different scales per scoring strategy (likelihood
# is a probability close to 1, mahalanobis is a percentile up to 100), so each
# strategy gets its own display format
SCORING_HYPERPARAMETER_DISPLAY_MAP = {
    'likelihood': {
        'long_window': ('W', lambda v: str(int(v))),
        'short_window': ('W\'', lambda v: str(int(v))),
        'anomaly_threshold': ('L_t', lambda v: _format_trimmed(v, 5)),
        'topk': ('K', lambda v: str(int(v))),
    },
    'mahalanobis': {
        'long_window': ('W', lambda v: str(int(v))),
        'short_window': ('W\'', lambda v: str(int(v))),
        'anomaly_threshold': ('\\epsilon', lambda v: _format_trimmed(v, 1)),
        'topk': ('K', lambda v: str(int(v))),
    },
}


def save_most_optimal_hyperparameters_to_latex(
        most_optimal_hyperparameters_df, results_dir, selected_sliding_window,
        shown_parameter_columns=None,
):
    scoring_strategies = ['likelihood', 'mahalanobis']
    all_parameter_columns = list(SCORING_HYPERPARAMETER_DISPLAY_MAP[scoring_strategies[0]].keys())
    if shown_parameter_columns is None:
        shown_parameter_columns = all_parameter_columns
    parameter_columns = [col for col in all_parameter_columns if col in shown_parameter_columns]

    window_df = most_optimal_hyperparameters_df[
        most_optimal_hyperparameters_df['sliding_window'] == selected_sliding_window
    ]

    subsets = sorted(set(zip(window_df['http_code'], window_df['aggregation'])))

    rows = []
    for http_code, agg in subsets:
        subset_rows = window_df[
            (window_df['http_code'] == http_code) & (window_df['aggregation'] == agg)
        ]
        if subset_rows.empty:
            continue

        # Optimal imputation is shared across scoring strategies for a given
        # subset/sliding_window, see extract_most_optimal_scoring_hyperparameter_...
        optimal_imputation = subset_rows.iloc[0]['imputation_strategy']

        hyperparameter_strings = {}
        for strategy in scoring_strategies:
            strategy_rows = subset_rows[subset_rows['post_processing_strategy'] == strategy]
            if strategy_rows.empty:
                hyperparameter_strings[strategy] = '--'
                continue

            row = strategy_rows.iloc[0]
            strategy_display_map = SCORING_HYPERPARAMETER_DISPLAY_MAP[strategy]
            hyperparameter_strings[strategy] = ', '.join(
                f'{abbrev}={fmt(row[col])}'
                for col, (abbrev, fmt) in strategy_display_map.items()
                if col in parameter_columns
            )
            hyperparameter_strings[strategy] = f'${hyperparameter_strings[strategy]}$'

        rows.append((
            f'${text_subset_wrapper_latex(http_code, agg)}$',
            optimal_imputation,
            hyperparameter_strings['likelihood'],
            hyperparameter_strings['mahalanobis'],
        ))

    header = ['Subset', 'Imputation', 'Likelihood', 'Mahalanobis']

    lines = [
        '\\begin{tabular}{llll}',
        '\\toprule',
        ' & '.join(header) + ' \\\\',
        '\\midrule',
    ]
    lines.extend(
        f'{subset_name} & {imputation} & {likelihood_str} & {mahalanobis_str} \\\\'
        for subset_name, imputation, likelihood_str, mahalanobis_str in rows
    )
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    latex_table = '\n'.join(lines)

    merged_results_dir = os.path.join(results_dir, 'merged_results','latex')
    os.makedirs(merged_results_dir, exist_ok=True)
    out_path = os.path.join(
        merged_results_dir, f'most_optimal_hyperparameters_window_{selected_sliding_window}.tex'
    )
    with open(out_path, 'w') as f:
        f.write(latex_table)
    print(f'Most optimal hyperparameters LaTeX table saved to {out_path}')

    return out_path


def compare_model_performance_across_sliding_windows(
        optimal_hyperparameters_df, results_dir, supported_models, graph_models,
        supported_sliding_windows, http_codes, aggregations,
        missing_imputation_strategies, null_padding_features, null_padding_targets,
        nab_profiles,
        is_one_column_figure=False,
):
    scoring_strategies = ['likelihood', 'mahalanobis']
    subplot_columns = list(itertools.product(scoring_strategies, nab_profiles))

    subsets = sorted(set(
        (hc, agg) for hc in http_codes for agg in aggregations
        if not optimal_hyperparameters_df[
            (optimal_hyperparameters_df['http_code'] == hc) &
            (optimal_hyperparameters_df['aggregation'] == agg)
        ].empty
    ))

    # (model_name, model_folder) pairs — graph models get one entry per
    # null-padding combination, following the naming used in plot_computation_time
    model_variants = []
    for m in supported_models:
        if m not in graph_models:
            model_variants.append((m, m))
        else:
            for (npf, npt) in itertools.product(null_padding_features, null_padding_targets):
                if not npf and not npt:
                    model_folder = m
                elif npf and not npt:
                    model_folder = f'{m}_null_padding_feature'
                elif not npf and npt:
                    model_folder = f'{m}_null_padding_target'
                else:
                    model_folder = f'{m}_null_padding_both'
                model_variants.append((m, model_folder))

    n_rows = len(subsets)
    n_cols = len(subplot_columns)

    # Each model family (base model name) gets a fixed hue from MODEL_FAMILY_HUE_MAP
    # (e.g. blue for A3TGCN, green for GRU), falling back to an evenly spaced hue for
    # any unmapped model; variants within a family (e.g. the graph model's null-padding
    # variants) share that hue but differ in lightness so they read as related while
    # still being distinguishable.
    fallback_hues = iter(np.linspace(0, 1, len(supported_models), endpoint=False))
    model_colors = {}
    for model_name in supported_models:
        family_hue = MODEL_FAMILY_HUE_MAP.get(model_name, next(fallback_hues))
        family_variant_folders = [
            model_folder for m, model_folder in model_variants if m == model_name
        ]
        lightness_values = np.linspace(0.35, 0.65, len(family_variant_folders))
        for model_folder, lightness in zip(family_variant_folders, lightness_values):
            model_colors[model_folder] = colorsys.hls_to_rgb(family_hue, lightness, 0.85)

    linestyles = {'zero': 'solid', 'mean': 'dashed', 'median': 'dotted'}

    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figure_width, 2.5 * n_rows), squeeze=False)

    for row_idx, (http_code, agg) in enumerate(subsets):
        for col_idx, (strategy, profile) in enumerate(subplot_columns):
            ax = axes[row_idx][col_idx]
            normalized_column = f'{profile}_normalized'

            for model_name, model_folder in model_variants:
                for imputation in missing_imputation_strategies:
                    x_values = []
                    y_values = []

                    for window in supported_sliding_windows:
                        opt_rows = optimal_hyperparameters_df[
                            (optimal_hyperparameters_df['sliding_window'] == window) &
                            (optimal_hyperparameters_df['http_code'] == http_code) &
                            (optimal_hyperparameters_df['aggregation'] == agg) &
                            (optimal_hyperparameters_df['imputation_strategy'] == imputation) &
                            (optimal_hyperparameters_df['nab_profile'] == profile) &
                            (optimal_hyperparameters_df['post_processing_strategy'] == strategy)
                        ]
                        if opt_rows.empty:
                            continue
                        opt_row = opt_rows.iloc[0]

                        grid_search_csv_path = os.path.join(
                            results_dir,
                            f'window_{window}',
                            f'no_group_{http_code}_{agg}',
                            f'fill_nan_with_{imputation}',
                            model_folder,
                            f'{model_name}_grid_search.csv'
                        )
                        if not os.path.exists(grid_search_csv_path):
                            continue

                        model_df = pd.read_csv(grid_search_csv_path)
                        matched_rows = model_df[
                            (model_df['post_processing_strategy'] == strategy) &
                            np.isclose(model_df['long_window'], opt_row['long_window']) &
                            np.isclose(model_df['short_window'], opt_row['short_window']) &
                            np.isclose(model_df['anomaly_threshold'], opt_row['anomaly_threshold']) &
                            np.isclose(model_df['topk'], opt_row['topk'])
                        ]
                        if matched_rows.empty:
                            continue

                        x_values.append(window)
                        y_values.append(matched_rows.iloc[0][normalized_column])

                    if x_values:
                        display_name = MODEL_DISPLAY_NAME_MAP.get(model_folder, model_folder)
                        ax.plot(
                            x_values, y_values, marker='o', markersize=4,
                            color=model_colors[model_folder], linestyle=linestyles[imputation],
                            label=f'{display_name} ({imputation})',
                        )

            strategy_display_name = SCORING_STRATEGY_DISPLAY_NAME_MAP.get(strategy, strategy)
            profile_display_name = NAB_PROFILE_DISPLAY_NAME_MAP.get(profile, profile)
            ax.set_title(f'NAB Score\n${text_subset_wrapper(http_code,agg)}$ — {strategy_display_name} — {profile_display_name}',
                        fontsize=TITLE_FONT_SIZE,
                         # fontweight='bold'
                         )
            ax.set_xlabel('Sliding window', fontsize=AXIS_LABEL_FONT_SIZE)
            # ax.set_ylabel('NAB score', fontsize=AXIS_LABEL_FONT_SIZE)
            ax.set_xticks(supported_sliding_windows)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            ax.tick_params(labelsize=TICK_FONT_SIZE)

    model_legend_handles = [
        plt.Line2D([0], [0], color=model_colors[model_folder],
                   label=MODEL_DISPLAY_NAME_MAP.get(model_folder, model_folder))
        for _, model_folder in model_variants
    ]
    imputation_legend_handles = [
        plt.Line2D([0], [0], color='black', linestyle=ls, label=imp)
        for imp, ls in linestyles.items()
    ]
    fig.legend(
        handles=model_legend_handles + imputation_legend_handles,
        loc='upper center', bbox_to_anchor=(0.5, 1.04),
        ncol=len(model_variants) + len(linestyles), fontsize=LEGEND_FONT_SIZE, frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    out_path = os.path.join(merged_results_dir, 'model_performance_across_sliding_windows.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Model performance comparison plot saved to {out_path}')

    return out_path


# Which triangle half of a cell (split along the top-left → bottom-right diagonal)
# each scoring strategy is drawn in
SCORING_STRATEGY_TRIANGLE_MAP = {
    'likelihood': 'upper',
    'mahalanobis': 'lower',
}


def _detected_group_color(detection_counters, anomaly_id, undetected_color):
    for group_key in ANOMALY_GROUP_ID_KEYS:
        if anomaly_id in detection_counters.get(group_key, []):
            group_color = to_rgb(ANOMALY_GROUP_COLOR_MAP[group_key])
            return np.array(group_color) * ANOMALY_GROUP_OPACITY + \
                   np.array(undetected_color) * (1 - ANOMALY_GROUP_OPACITY)
    return np.array(undetected_color)


def _is_anomaly_detected(detection_counters, anomaly_id):
    return any(anomaly_id in detection_counters.get(group_key, []) for group_key in ANOMALY_GROUP_ID_KEYS)


def _blended_group_background(group_key, undetected_color):
    group_color = to_rgb(ANOMALY_GROUP_COLOR_MAP[group_key])
    return np.array(group_color) * ANOMALY_GROUP_OPACITY + \
           np.array(undetected_color) * (1 - ANOMALY_GROUP_OPACITY)


def plot_table_of_detected_anomalies(
        optimal_hyperparameters_df, results_dir, sliding_window, nab_profile,
        is_one_column_figure=False,
):
    scoring_strategies = ['likelihood', 'mahalanobis']
    normalized_column = f'{nab_profile}_normalized'

    profile_df = optimal_hyperparameters_df[
        (optimal_hyperparameters_df['sliding_window'] == sliding_window) &
        (optimal_hyperparameters_df['nab_profile'] == nab_profile)
    ]

    subsets = sorted(set(zip(profile_df['http_code'], profile_df['aggregation'])))
    undetected_color = to_rgb(UNDETECTED_CELL_COLOR)

    row_labels = [f'${text_subset_wrapper(hc, agg)}$' for hc, agg in subsets]

    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, ax = plt.subplots(figsize=(figure_width, DETECTED_ANOMALIES_TABLE_ROW_HEIGHT * len(subsets) + 1))

    # Ground truth is fixed across subsets/strategies, so the anomaly index -> group
    # mapping used for the column label backgrounds only needs to be captured once
    ground_truth_group_by_anomaly_id = {}

    for row_idx, (http_code, agg) in enumerate(subsets):
        subset_rows = profile_df[
            (profile_df['http_code'] == http_code) & (profile_df['aggregation'] == agg)
        ]

        imputation_by_strategy = {}
        for strategy in scoring_strategies:
            strategy_rows = subset_rows[subset_rows['post_processing_strategy'] == strategy]
            if strategy_rows.empty:
                continue

            # Optimal imputation for this subset/strategy is the one whose selected
            # hyperparameters score highest on the profile's normalized metric,
            # considering every imputation_strategy present in strategy_rows
            optimal_row = strategy_rows.loc[strategy_rows[normalized_column].idxmax()]
            strategy_display_name = SCORING_STRATEGY_DISPLAY_NAME_MAP.get(strategy, strategy)
            imputation_by_strategy[strategy_display_name] = optimal_row['imputation_strategy']
            detection_counters = ast.literal_eval(optimal_row['detection_counters'])

            if not ground_truth_group_by_anomaly_id:
                for group_key, gt_key in GROUND_TRUTH_ID_KEY_MAP.items():
                    for anomaly_id in detection_counters.get(gt_key, []):
                        ground_truth_group_by_anomaly_id[anomaly_id] = group_key

            triangle = SCORING_STRATEGY_TRIANGLE_MAP[strategy]
            for anomaly_id in range(NUM_TESTING_ANOMALIES):
                cell_color = _detected_group_color(detection_counters, anomaly_id, undetected_color)
                x, y = anomaly_id, row_idx
                if triangle == 'upper':
                    vertices = [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5)]
                else:
                    vertices = [(x - 0.5, y - 0.5), (x - 0.5, y + 0.5), (x + 0.5, y + 0.5)]
                ax.add_patch(plt.Polygon(vertices, closed=True, facecolor=cell_color,
                                         edgecolor='none'))

        if imputation_by_strategy:
            assert len(set(imputation_by_strategy.values())) == 1, (
                f'{http_code} {agg}: optimal imputation differs across scoring strategies: '
                f'{imputation_by_strategy}'
            )

    for anomaly_id in range(NUM_TESTING_ANOMALIES):
        for row_idx in range(len(subsets)):
            ax.plot([anomaly_id - 0.5, anomaly_id + 0.5], [row_idx - 0.5, row_idx + 0.5],
                   color='lightgray', linewidth=0.5, zorder=2)

    ax.set_xlim(-0.5, NUM_TESTING_ANOMALIES - 0.5)
    ax.set_ylim(len(subsets) - 0.5, -0.5)
    # One data unit is one cell along both axes, so equal aspect keeps every cell square
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(range(NUM_TESTING_ANOMALIES))
    ax.set_xticklabels(range(NUM_TESTING_ANOMALIES), fontsize=TICK_FONT_SIZE)
    for anomaly_id, tick_label in enumerate(ax.get_xticklabels()):
        group_key = ground_truth_group_by_anomaly_id.get(anomaly_id)
        if group_key is None:
            continue
        group_color = to_rgb(ANOMALY_GROUP_COLOR_MAP[group_key])
        blended_color = np.array(group_color) * ANOMALY_GROUP_OPACITY + \
                        np.array(undetected_color) * (1 - ANOMALY_GROUP_OPACITY)
        tick_label.set_bbox(dict(facecolor=blended_color, edgecolor='none', pad=1.5))
    ax.set_yticks(range(len(subsets)))
    ax.set_yticklabels(row_labels, fontsize=TICK_FONT_SIZE - 2)
    ax.set_xlabel('Ground-truth anomaly index', fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_xticks(np.arange(-0.5, NUM_TESTING_ANOMALIES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subsets), 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    strategy_triangle_note = ' / '.join(
        f'{SCORING_STRATEGY_DISPLAY_NAME_MAP.get(s, s)}: {t} triangle'
        for s, t in SCORING_STRATEGY_TRIANGLE_MAP.items()
    )
    ax.set_title(strategy_triangle_note, fontsize=TICK_FONT_SIZE)

    profile_display_name = NAB_PROFILE_DISPLAY_NAME_MAP.get(nab_profile, nab_profile)
    # fig.suptitle(f'Detected Anomalies — Window {sliding_window} — {profile_display_name}',
    #             fontsize=TITLE_FONT_SIZE, fontweight='bold', y=1.28)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=ANOMALY_GROUP_OPACITY,
                      label=ANOMALY_GROUP_DISPLAY_NAME_MAP[key])
        for key, color in ANOMALY_GROUP_COLOR_MAP.items()
    ]
    fig.legend(handles=legend_handles, title='Anomaly Source',
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
              loc='upper center', bbox_to_anchor=(0.5, 0.95),
              ncol=len(legend_handles), frameon=True)

    fig.tight_layout()
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    out_path = os.path.join(
        merged_results_dir, f'table_detected_anomalies_window_{sliding_window}_{nab_profile}.png'
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Detected anomalies table saved to {out_path}')

    return out_path


# Color assigned to the winning scoring strategy's bar
SCORING_STRATEGY_COLOR_MAP = {
    'likelihood': 'steelblue',
    'mahalanobis': 'indianred',
}

# Color assigned to the winning scoring strategy's bar
SCORING_STRATEGY_COLOR_MAP_BOLD = {
    'likelihood': 'blue',
    'mahalanobis': 'red',
}

# Corner radius for rounded bars, in points (a physical unit independent of DPI
# and of the data scale on either axis)
BAR_CORNER_RADIUS_POINTS = 4


def _round_bar_corners(bars, radius_points=BAR_CORNER_RADIUS_POINTS):
    ax = bars.patches[0].axes

    # rounding_size/mutation_aspect are both defined in data units, so a fixed
    # physical radius (points, converted to pixels via the figure's DPI) is
    # mapped separately per axis using the current data-to-display scale (x
    # and y axes generally have different data ranges mapped over similar
    # pixel extents, so a naive shared radius would look stretched)
    radius_px = radius_points * ax.figure.dpi / 72.0
    inv = ax.transData.inverted()
    origin_disp = ax.transData.transform((0, 0))
    origin_data = inv.transform(origin_disp)
    radius_x_data = abs(inv.transform(origin_disp + [radius_px, 0])[0] - origin_data[0])
    radius_y_data = abs(inv.transform(origin_disp + [0, radius_px])[1] - origin_data[1])
    mutation_aspect = radius_y_data / radius_x_data if radius_x_data else 1

    for rect in bars.patches:
        # Take the bounding box's bottom-left corner and positive extents so a
        # negative-going bar (either a vertical bar's height or a horizontal
        # bar's width) still rounds correctly instead of the box flipping
        # inside-out around a fixed corner
        x = min(rect.get_x(), rect.get_x() + rect.get_width())
        y = min(rect.get_y(), rect.get_y() + rect.get_height())
        width, height = abs(rect.get_width()), abs(rect.get_height())
        rounded = FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f'round,pad=0,rounding_size={radius_x_data}',
            mutation_aspect=mutation_aspect,
            facecolor=rect.get_facecolor(), edgecolor=rect.get_edgecolor(),
            linewidth=rect.get_linewidth(), zorder=rect.get_zorder(),
        )
        rect.remove()
        ax.add_patch(rounded)


def plot_nab_score_of_optimal_configuration(
        optimal_hyperparameters_df, results_dir, sliding_window, nab_profile,
        is_one_column_figure=False,
):
    scoring_strategies = ['likelihood', 'mahalanobis']
    normalized_column = f'{nab_profile}_normalized'

    profile_df = optimal_hyperparameters_df[
        (optimal_hyperparameters_df['sliding_window'] == sliding_window) &
        (optimal_hyperparameters_df['nab_profile'] == nab_profile)
    ]

    subsets = sorted(set(zip(profile_df['http_code'], profile_df['aggregation'])))
    x_labels = [f'${text_subset_wrapper(hc, agg)}$' for hc, agg in subsets]

    # scores[strategy][subset_idx]; NaN where a strategy has no candidate rows
    # for that subset
    scores = {strategy: np.full(len(subsets), np.nan) for strategy in scoring_strategies}

    for subset_idx, (http_code, agg) in enumerate(subsets):
        subset_rows = profile_df[
            (profile_df['http_code'] == http_code) & (profile_df['aggregation'] == agg)
        ]

        imputation_by_strategy = {}
        for strategy in scoring_strategies:
            strategy_rows = subset_rows[subset_rows['post_processing_strategy'] == strategy]
            if strategy_rows.empty:
                continue

            # Optimal configuration for this subset/strategy is the imputation_strategy
            # whose selected hyperparameters score highest on the profile's normalized
            # metric, considering every imputation_strategy present in strategy_rows
            optimal_row = strategy_rows.loc[strategy_rows[normalized_column].idxmax()]
            scores[strategy][subset_idx] = optimal_row[normalized_column]
            imputation_by_strategy[strategy] = optimal_row['imputation_strategy']

        if imputation_by_strategy:
            assert len(set(imputation_by_strategy.values())) == 1, (
                f'{http_code} {agg}: optimal imputation differs across scoring strategies: '
                f'{imputation_by_strategy}'
            )

    n_strategies = len(scoring_strategies)
    bar_width = 0.7 / n_strategies
    x = np.arange(len(x_labels))

    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, ax = plt.subplots(figsize=(figure_width, 2.5))

    all_scores = np.concatenate(list(scores.values())) if x_labels else np.array([0])
    valid_scores = all_scores[~np.isnan(all_scores)]
    min_score = min(valid_scores.min(), 0) if len(valid_scores) else 0
    max_score = valid_scores.max() if len(valid_scores) else 1
    score_range = max_score - min_score
    # Extend the lower limit further when there are negative bars so their
    # value/config labels (drawn below the bar tip) have room and don't overlap
    # the axis edge or x-tick labels
    ylim_min = (min_score - 0.25 * score_range) if min_score < 0 else min_score
    ax.set_ylim(ylim_min, max_score + 0.3 * score_range)

    for i, strategy in enumerate(scoring_strategies):
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        strategy_display_name = SCORING_STRATEGY_DISPLAY_NAME_MAP.get(strategy, strategy)
        bars = ax.bar(x + offset, scores[strategy], bar_width,
                      color=SCORING_STRATEGY_COLOR_MAP.get(strategy, 'gray'),
                      label=strategy_display_name)
        bar_value_labels = [
            '' if np.isnan(score) else f'{score:.2f}'
            for score in scores[strategy]
        ]
        ax.bar_label(bars, labels=bar_value_labels, fontsize=TICK_FONT_SIZE-2, padding=2)
        _round_bar_corners(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=TICK_FONT_SIZE, rotation=30)
    ax.set_ylabel('NAB score', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=-5)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    # matplotlib stacks a legend's title above its entries by default; using an
    # invisible proxy handle as the title instead keeps everything on one row
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    title_handle = plt.Line2D([], [], color='none')
    fig.legend([title_handle] + legend_handles, ['Scoring strategy:'] + legend_labels,
              fontsize=LEGEND_FONT_SIZE,
              loc='upper center', bbox_to_anchor=(0.5, 1.08),
              ncol=len(legend_labels) + 1, frameon=True, handletextpad=0.5)

    fig.tight_layout()
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    out_path = os.path.join(
        merged_results_dir, f'nab_score_optimal_configuration_window_{sliding_window}_{nab_profile}.png'
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'NAB score of optimal configuration plot saved to {out_path}')

    return out_path


def plot_table_and_nab_score_together(
        optimal_hyperparameters_df, results_dir, sliding_window, nab_profiles,
        is_one_column_figure=False,
):
    scoring_strategies = ['likelihood', 'mahalanobis']

    # Detected cells/hyperparameters are selected using the first profile; every
    # profile in nab_profiles then gets its own bar-chart panel, each reporting
    # that profile's own score for this same selected configuration
    selection_profile = nab_profiles[0]
    selection_normalized_column = f'{selection_profile}_normalized'

    profile_df = optimal_hyperparameters_df[
        (optimal_hyperparameters_df['sliding_window'] == sliding_window) &
        (optimal_hyperparameters_df['nab_profile'] == selection_profile)
    ]

    subsets = sorted(set(zip(profile_df['http_code'], profile_df['aggregation'])))
    undetected_color = to_rgb(UNDETECTED_CELL_COLOR)
    row_labels = [f'${text_subset_wrapper(hc, agg)}$' for hc, agg in subsets]

    # scores[profile][strategy][row_idx], indexed the same way as the table's
    # rows so every panel is matched by subset
    scores = {
        profile: {strategy: np.full(len(subsets), np.nan) for strategy in scoring_strategies}
        for profile in nab_profiles
    }

    n_bar_panels = len(nab_profiles)
    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, axes = plt.subplots(
        1, 1 + n_bar_panels, sharey=True,
        figsize=(figure_width, DETECTED_ANOMALIES_TABLE_ROW_HEIGHT * len(subsets) + 1),
        gridspec_kw={'width_ratios': [NUM_TESTING_ANOMALIES] + [6] * n_bar_panels, 'wspace': 0.05},
    )
    ax_table, *ax_bars_list = axes

    # Ground truth is fixed across subsets/strategies, so the anomaly index -> group
    # mapping can be captured once from any row up front, before the cells are drawn
    ground_truth_group_by_anomaly_id = {}
    first_detection_counters = ast.literal_eval(profile_df.iloc[0]['detection_counters'])
    for group_key, gt_key in GROUND_TRUTH_ID_KEY_MAP.items():
        for anomaly_id in first_detection_counters.get(gt_key, []):
            ground_truth_group_by_anomaly_id[anomaly_id] = group_key

    for row_idx, (http_code, agg) in enumerate(subsets):
        subset_rows = profile_df[
            (profile_df['http_code'] == http_code) & (profile_df['aggregation'] == agg)
        ]

        imputation_by_strategy = {}
        for strategy in scoring_strategies:
            strategy_rows = subset_rows[subset_rows['post_processing_strategy'] == strategy]
            if strategy_rows.empty:
                continue

            # Optimal imputation/hyperparameters for this subset/strategy are the
            # ones that score highest on the selection profile's normalized
            # metric, considering every imputation_strategy present in strategy_rows
            optimal_row = strategy_rows.loc[strategy_rows[selection_normalized_column].idxmax()]
            strategy_display_name = SCORING_STRATEGY_DISPLAY_NAME_MAP.get(strategy, strategy)
            imputation_by_strategy[strategy_display_name] = optimal_row['imputation_strategy']
            for profile in nab_profiles:
                scores[profile][strategy][row_idx] = optimal_row[f'{profile}_normalized']
            detection_counters = ast.literal_eval(optimal_row['detection_counters'])

            # Triangle fill marks whether this strategy (under its own optimal
            # hyperparameters) detected the anomaly: raw scoring-strategy color
            # (matching the bar chart) when detected, raw anomaly-source color at
            # low opacity when not
            triangle = SCORING_STRATEGY_TRIANGLE_MAP[strategy]
            strategy_color = SCORING_STRATEGY_COLOR_MAP.get(strategy, 'gray')
            for anomaly_id in range(NUM_TESTING_ANOMALIES):
                if _is_anomaly_detected(detection_counters, anomaly_id):
                    facecolor, alpha = strategy_color, 1
                else:
                    group_key = ground_truth_group_by_anomaly_id.get(anomaly_id)
                    facecolor = ANOMALY_GROUP_COLOR_MAP.get(group_key, UNDETECTED_CELL_COLOR)
                    alpha = ANOMALY_GROUP_OPACITY*0.25

                x, y = anomaly_id, row_idx
                if triangle == 'upper':
                    vertices = [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5)]
                else:
                    vertices = [(x - 0.5, y - 0.5), (x - 0.5, y + 0.5), (x + 0.5, y + 0.5)]
                ax_table.add_patch(plt.Polygon(vertices, closed=True, facecolor=facecolor,
                                               alpha=alpha, edgecolor='none', zorder=1))

        if imputation_by_strategy:
            assert len(set(imputation_by_strategy.values())) == 1, (
                f'{http_code} {agg}: optimal imputation differs across scoring strategies: '
                f'{imputation_by_strategy}'
            )

    for anomaly_id in range(NUM_TESTING_ANOMALIES):
        for row_idx in range(len(subsets)):
            ax_table.plot([anomaly_id - 0.5, anomaly_id + 0.5], [row_idx - 0.5, row_idx + 0.5],
                         color='lightgray', linewidth=0.5, zorder=2)

    ax_table.set_xlim(-0.5, NUM_TESTING_ANOMALIES - 0.5)
    ax_table.set_ylim(len(subsets) - 0.5, -0.5)
    # One data unit is one cell along both axes, so equal aspect keeps every cell square
    ax_table.set_aspect('equal', adjustable='box')
    ax_table.set_xticks(range(NUM_TESTING_ANOMALIES))
    ax_table.set_xticklabels(range(NUM_TESTING_ANOMALIES), fontsize=TICK_FONT_SIZE)
    for anomaly_id, tick_label in enumerate(ax_table.get_xticklabels()):
        group_key = ground_truth_group_by_anomaly_id.get(anomaly_id)
        if group_key is None:
            continue
        blended_color = _blended_group_background(group_key, undetected_color)
        tick_label.set_bbox(dict(facecolor=blended_color, edgecolor='none', pad=1.5))
    ax_table.set_yticks(range(len(subsets)))
    ax_table.set_yticklabels(row_labels, fontsize=TICK_FONT_SIZE)
    ax_table.set_xlabel('Ground-truth anomaly index', fontsize=TITLE_FONT_SIZE)

    ax_table.set_xticks(np.arange(-0.5, NUM_TESTING_ANOMALIES, 1), minor=True)
    ax_table.set_yticks(np.arange(-0.5, len(subsets), 1), minor=True)
    ax_table.grid(which='minor', color='lightgray', linewidth=0.5)
    ax_table.tick_params(which='minor', length=0)

    strategy_triangle_note = ' / '.join(
        f'{SCORING_STRATEGY_DISPLAY_NAME_MAP.get(s, s)}: {t} triangle'
        for s, t in SCORING_STRATEGY_TRIANGLE_MAP.items()
    )
    ax_table.set_title(strategy_triangle_note, fontsize=TICK_FONT_SIZE)

    # Each bar chart panel is rotated to horizontal bars so a subset's score
    # lines up with that subset's row in the table; one panel per nab_profile
    n_strategies = len(scoring_strategies)
    bar_height = 0.7 / n_strategies
    y = np.arange(len(subsets))

    for ax_bars, profile in zip(ax_bars_list, nab_profiles):
        profile_scores = scores[profile]

        all_scores = np.concatenate(list(profile_scores.values())) if subsets else np.array([0])
        valid_scores = all_scores[~np.isnan(all_scores)]
        min_score = min(valid_scores.min(), 0) if len(valid_scores) else 0
        max_score = valid_scores.max() if len(valid_scores) else 1
        score_range = max_score - min_score
        xlim_min = (min_score - 0.25 * score_range) if min_score < 0 else min_score
        ax_bars.set_xlim(xlim_min, max_score + 0.3 * score_range)

        for i, strategy in enumerate(scoring_strategies):
            offset = (i - n_strategies / 2 + 0.5) * bar_height
            strategy_display_name = SCORING_STRATEGY_DISPLAY_NAME_MAP.get(strategy, strategy)
            bars = ax_bars.barh(y + offset, profile_scores[strategy], bar_height,
                                color=SCORING_STRATEGY_COLOR_MAP.get(strategy, 'gray'),
                                label=strategy_display_name)
            bar_value_labels = [
                '' if np.isnan(score) else f'{score:.2f}'
                for score in profile_scores[strategy]
            ]
            ax_bars.bar_label(bars, labels=bar_value_labels, fontsize=TICK_FONT_SIZE, padding=2)
            _round_bar_corners(bars)

        ax_bars.set_xlabel(
            f'NAB score\nunder {NAB_PROFILE_DISPLAY_NAME_MAP.get(profile, profile)} profile',
            fontsize=TITLE_FONT_SIZE
        )
        ax_bars.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
        ax_bars.tick_params(axis='y', labelleft=False, length=0)
        ax_bars.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.6)

    # Each panel gets its own legend, anchored above that panel's own title
    # rather than sharing one figure-wide legend; the scoring-strategy legend
    # only needs to appear once, above the first bar panel
    anomaly_source_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=ANOMALY_GROUP_OPACITY,
                      label=ANOMALY_GROUP_DISPLAY_NAME_MAP[key])
        for key, color in ANOMALY_GROUP_COLOR_MAP.items()
    ]
    ax_table.legend(
        handles=anomaly_source_handles, title='Anomaly Source',
        fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
        loc='lower center', bbox_to_anchor=(0.5, 1.1),
        ncol=len(anomaly_source_handles), frameon=True,
    )
    ax_bars_list[0].legend(
        title='Scoring strategy',
        fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
        loc='lower center', bbox_to_anchor=(0.5, 1.1),
        ncol=n_strategies, frameon=True,
    )

    fig.tight_layout()

    # set_aspect('equal') on ax_table shrinks its box to keep cells square, so
    # after layout it is usually narrower than its allocated slot, leaving a gap
    # before the bar panels; shift every bar panel left by that same gap (so
    # their relative widths/spacing are preserved) and match ax_table's height
    # so all panels line up row-for-row with minimal gap between them
    fig.canvas.draw()
    table_bbox = ax_table.get_position()
    panel_gap = 0.02
    first_bar_bbox = ax_bars_list[0].get_position()
    shift = first_bar_bbox.x0 - (table_bbox.x1 + panel_gap)
    for ax_bars in ax_bars_list:
        bbox = ax_bars.get_position()
        ax_bars.set_position([bbox.x0 - shift, table_bbox.y0, bbox.width, table_bbox.height])

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    out_path = os.path.join(
        merged_results_dir, f'table_and_nab_score_window_{sliding_window}_{"_".join(nab_profiles)}.png'
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Combined detected anomalies table and NAB score plot saved to {out_path}')

    return out_path


def _parse_ensemble_combo_subsets(combo_id):
    subsets = []
    for token in combo_id.split('+'):
        http_code, agg = token.replace('no_group_', '', 1).split('_', 1)
        subsets.append((http_code, agg))
    return subsets


# Fixed slot order for the 8-character combo code returned by
# _format_ensemble_combo_code: one slot per (http_code, aggregation), '4xx'
# slots first then '5xx', each group ordered avg/count/max/min
COMBO_CODE_SLOTS = [
    (http_code, agg)
    for http_code in ('4xx', '5xx')
    for agg in ('avg', 'count', 'max', 'min')
]


def _format_ensemble_combo_code(combo_id):
    used_subsets = set(_parse_ensemble_combo_subsets(combo_id))
    codes = ['*' if slot in used_subsets else '0' for slot in COMBO_CODE_SLOTS]
    # First 4 slots are '4xx', last 4 are '5xx' (see COMBO_CODE_SLOTS), so
    # splitting down the middle puts each http_code's flags on its own line
    midpoint = len(codes) // 2
    return ''.join(codes[:midpoint]) + '\n' + ''.join(codes[midpoint:])


def _count_detected_anomalies(detection_counters):
    return sum(
        detection_counters.get(key, 0)
        for key in ('issue_detected', 'im_detected', 'TestLog_detected')
    )


def _format_detected_anomalies_cell(detection_counters, group_key):
    ids_key = f'{group_key}_ids'
    gt_key = GROUND_TRUTH_ID_KEY_MAP[ids_key]
    detected_ids = detection_counters.get(ids_key, [])
    total_count = len(detection_counters.get(gt_key, []))
    return f'{detection_counters.get(group_key, 0)}/{total_count}\n{detected_ids}'


# Columns of the confusion-matrix table drawn after the bar charts in
# plot_table_and_nab_score_of_ensembles, keyed the same way as the
# confusion_matrix tuple (tn, fp, fn, tp) unpacked from each row
CONFUSION_MATRIX_COLUMNS = ['tp', 'tn', 'fp', 'fn']
CONFUSION_MATRIX_COLUMN_LABELS = ['TP', 'TN', 'FP', 'FN']
# The confusion-matrix table's leading column holds the encoded combo-code row
# label (not split diagonally, unlike the TP/TN/FP/FN columns after it)
CONFUSION_MATRIX_LABEL_COLUMN_HEADER = 'Ensemble'


def plot_table_and_nab_score_of_ensembles(
        results_dir, sliding_window, nab_profiles, top_k,
        is_one_column_figure=False,
):
    scoring_strategies = ['likelihood', 'mahalanobis']
    ensemble_dir = os.path.join(results_dir, f'window_{sliding_window}', 'ensemble')
    file_prefix = 'ensemble_final_result_optimal_configuration_'

    # combo_id -> {strategy: first row (the ensemble-level aggregate) of that
    # ensemble's result CSV}
    combo_rows = {}
    for filename in sorted(os.listdir(ensemble_dir)):
        if not (filename.startswith(file_prefix) and filename.endswith('.csv')):
            continue
        remainder = filename[len(file_prefix):-len('.csv')]
        strategy = next((s for s in scoring_strategies if remainder.startswith(f'{s}_')), None)
        if strategy is None:
            continue
        combo_id = remainder[len(strategy) + 1:]

        result_df = pd.read_csv(os.path.join(ensemble_dir, filename))
        if result_df.empty:
            continue
        combo_rows.setdefault(combo_id, {})[strategy] = result_df.iloc[0]

    assert combo_rows, f'No ensemble result CSVs starting with {file_prefix!r} found in {ensemble_dir}'

    # For each subset count (1-subset combos, 2-subset combos, ...), keep
    # ensembles selected by three criteria: the single best mahalanobis
    # ensemble, the single best likelihood ensemble (both ranked by
    # reward_fn_normalized), and, separately for each scoring strategy, the
    # top_k ensembles by that strategy's own number of detected ground-truth
    # anomalies (summed across the three anomaly-source groups); a combo
    # winning more than one of these criteria is still only shown once
    reward_fn_column = 'reward_fn_normalized'
    combo_ids_by_subset_count = {}
    for combo_id in combo_rows:
        subset_count = len(_parse_ensemble_combo_subsets(combo_id))
        combo_ids_by_subset_count.setdefault(subset_count, []).append(combo_id)

    selected_combo_ids = set()
    for group_combo_ids in combo_ids_by_subset_count.values():
        for strategy in scoring_strategies:
            strategy_scores = {
                combo_id: combo_rows[combo_id][strategy][reward_fn_column]
                for combo_id in group_combo_ids
                if strategy in combo_rows[combo_id]
            }
            if strategy_scores:
                selected_combo_ids.add(max(strategy_scores, key=strategy_scores.get))

        for strategy in scoring_strategies:
            detected_counts = {
                combo_id: _count_detected_anomalies(
                    ast.literal_eval(combo_rows[combo_id][strategy]['detection_counters'])
                )
                for combo_id in group_combo_ids
                if strategy in combo_rows[combo_id]
            }
            top_detected_combo_ids = sorted(detected_counts, key=detected_counts.get, reverse=True)[:top_k]
            selected_combo_ids.update(top_detected_combo_ids)

    # Keep every ensemble found on disk (all_combo_rows) separate from the
    # top_k-selected ones actually drawn in the table/bar chart below, so the
    # exported summary can still be used to look up any specific ensemble
    # (e.g. by export_selected_ensembles_to_latex_for_each_scoring_strategy)
    # even if it wasn't one of the automatically selected top_k combos
    all_combo_rows = combo_rows
    combo_rows = {combo_id: all_combo_rows[combo_id] for combo_id in selected_combo_ids}
    # Rows are ordered by number of combined subsets first (ascending), then
    # alphabetically within a subset count for a stable, deterministic order
    combo_ids = sorted(
        combo_rows.keys(),
        key=lambda combo_id: (len(_parse_ensemble_combo_subsets(combo_id)), combo_id)
    )

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)

    # One row per (ensemble, scoring strategy), since the confusion matrix and
    # detected ground-truth anomalies are strategy-specific
    ensemble_summary_records = []
    all_combo_ids = sorted(
        all_combo_rows.keys(),
        key=lambda combo_id: (len(_parse_ensemble_combo_subsets(combo_id)), combo_id)
    )
    for combo_id in all_combo_ids:
        for strategy in scoring_strategies:
            row = all_combo_rows[combo_id].get(strategy)
            if row is None:
                continue
            tn, fp, fn, tp = ast.literal_eval(row['confusion_matrix'])
            detection_counters = ast.literal_eval(row['detection_counters'])
            ensemble_summary_records.append({
                'ensemble_id': combo_id,
                'scoring_strategy': strategy,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'standard_nab_score': row['standard_normalized'],
                'reward_fn_nab_score': row['reward_fn_normalized'],
                'issue_detected': _format_detected_anomalies_cell(detection_counters, 'issue_detected'),
                'im_detected': _format_detected_anomalies_cell(detection_counters, 'im_detected'),
                'TestLog_detected': _format_detected_anomalies_cell(detection_counters, 'TestLog_detected'),
            })
    ensemble_summary_df = pd.DataFrame.from_records(ensemble_summary_records)
    ensemble_summary_csv_path = os.path.join(
        merged_results_dir,
        f'ensembles_summary_window_{sliding_window}_{"_".join(nab_profiles)}.csv'
    )
    ensemble_summary_df.to_csv(ensemble_summary_csv_path, index=False)
    print(f'Ensembles summary saved to {ensemble_summary_csv_path}')

    undetected_color = to_rgb(UNDETECTED_CELL_COLOR)
    # The encoded combo codes are shown inside the confusion-matrix table's
    # leading column, so the axes' own y-tick labels just number the rows in
    # display order
    combo_codes = [_format_ensemble_combo_code(combo_id) for combo_id in combo_ids]
    row_labels = [str(i) for i in range(1, len(combo_ids) + 1)]

    # scores[profile][strategy][row_idx], indexed the same way as the table's
    # rows so every panel is matched by ensemble combo
    scores = {
        profile: {strategy: np.full(len(combo_ids), np.nan) for strategy in scoring_strategies}
        for profile in nab_profiles
    }

    n_bar_panels = len(nab_profiles)
    # +1 for the leading encoded-combo-code column
    n_confusion_columns = len(CONFUSION_MATRIX_COLUMNS) + 1
    figure_width = ONE_COLUMN_FIGURE_WIDTH if is_one_column_figure else TWO_COLUMN_FIGURE_WIDTH
    fig, axes = plt.subplots(
        1, 1 + n_bar_panels + 1, sharey=True,
        figsize=(figure_width, DETECTED_ANOMALIES_TABLE_ROW_HEIGHT * len(combo_ids) + 1),
        gridspec_kw={
            'width_ratios': [n_confusion_columns, NUM_TESTING_ANOMALIES] + [6] * n_bar_panels,
            'wspace': 0.05,
        },
    )
    ax_confusion, ax_table, *ax_bars_list = axes

    # Ground truth is fixed across combos/strategies, so the anomaly index -> group
    # mapping can be captured once from any available row up front
    first_row = next(row for strategy_rows in combo_rows.values() for row in strategy_rows.values())
    first_detection_counters = ast.literal_eval(first_row['detection_counters'])
    ground_truth_group_by_anomaly_id = {}
    for group_key, gt_key in GROUND_TRUTH_ID_KEY_MAP.items():
        for anomaly_id in first_detection_counters.get(gt_key, []):
            ground_truth_group_by_anomaly_id[anomaly_id] = group_key

    # confusion_by_combo[combo_id][strategy] = {'tp': ..., 'tn': ..., 'fp': ..., 'fn': ...}
    confusion_by_combo = {combo_id: {} for combo_id in combo_ids}

    for row_idx, combo_id in enumerate(combo_ids):
        strategy_rows = combo_rows[combo_id]

        for strategy in scoring_strategies:
            row = strategy_rows.get(strategy)
            if row is None:
                continue

            for profile in nab_profiles:
                scores[profile][strategy][row_idx] = row[f'{profile}_normalized']
            detection_counters = ast.literal_eval(row['detection_counters'])
            tn, fp, fn, tp = ast.literal_eval(row['confusion_matrix'])
            confusion_by_combo[combo_id][strategy] = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

            # Triangle fill marks whether this strategy's ensemble (OR-combined
            # across the combo's subsets) detected the anomaly: raw
            # scoring-strategy color (matching the bar chart) when detected, raw
            # anomaly-source color at low opacity when not
            triangle = SCORING_STRATEGY_TRIANGLE_MAP[strategy]
            strategy_color = SCORING_STRATEGY_COLOR_MAP.get(strategy, 'gray')
            for anomaly_id in range(NUM_TESTING_ANOMALIES):
                if _is_anomaly_detected(detection_counters, anomaly_id):
                    facecolor, alpha = strategy_color, 1
                else:
                    group_key = ground_truth_group_by_anomaly_id.get(anomaly_id)
                    facecolor = ANOMALY_GROUP_COLOR_MAP.get(group_key, UNDETECTED_CELL_COLOR)
                    alpha = ANOMALY_GROUP_OPACITY * 0.25

                x, y = anomaly_id, row_idx
                if triangle == 'upper':
                    vertices = [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5)]
                else:
                    vertices = [(x - 0.5, y - 0.5), (x - 0.5, y + 0.5), (x + 0.5, y + 0.5)]
                ax_table.add_patch(plt.Polygon(vertices, closed=True, facecolor=facecolor,
                                               alpha=alpha, edgecolor='none', zorder=1))

    for anomaly_id in range(NUM_TESTING_ANOMALIES):
        for row_idx in range(len(combo_ids)):
            ax_table.plot([anomaly_id - 0.5, anomaly_id + 0.5], [row_idx - 0.5, row_idx + 0.5],
                         color='lightgray', linewidth=0.5, zorder=2)

    ax_table.set_xlim(-0.5, NUM_TESTING_ANOMALIES - 0.5)
    ax_table.set_ylim(len(combo_ids) - 0.5, -0.5)
    # One data unit is one cell along both axes, so equal aspect keeps every cell square
    ax_table.set_aspect('equal', adjustable='box')
    ax_table.set_xticks(range(NUM_TESTING_ANOMALIES))
    ax_table.set_xticklabels(range(NUM_TESTING_ANOMALIES), fontsize=TICK_FONT_SIZE)
    for anomaly_id, tick_label in enumerate(ax_table.get_xticklabels()):
        group_key = ground_truth_group_by_anomaly_id.get(anomaly_id)
        if group_key is None:
            continue
        blended_color = _blended_group_background(group_key, undetected_color)
        tick_label.set_bbox(dict(facecolor=blended_color, edgecolor='none', pad=1.5))
    # The encoded combo-code row labels are shown on ax_confusion instead (it's
    # the leftmost panel), so ax_table's own y-axis stays unlabeled
    ax_table.tick_params(axis='y', labelleft=False, length=0)
    ax_table.set_xlabel('Ground-truth anomaly index', fontsize=TITLE_FONT_SIZE)

    ax_table.set_xticks(np.arange(-0.5, NUM_TESTING_ANOMALIES, 1), minor=True)
    ax_table.set_yticks(np.arange(-0.5, len(combo_ids), 1), minor=True)
    ax_table.grid(which='minor', color='lightgray', linewidth=0.5)
    ax_table.tick_params(which='minor', length=0)

    strategy_triangle_note = ' / '.join(
        f'{SCORING_STRATEGY_DISPLAY_NAME_MAP.get(s, s)}: {t} triangle'
        for s, t in SCORING_STRATEGY_TRIANGLE_MAP.items()
    )
    ax_table.set_title(strategy_triangle_note, fontsize=TICK_FONT_SIZE)

    # Each bar chart panel is rotated to horizontal bars so a combo's score
    # lines up with that combo's row in the table; one panel per nab_profile
    n_strategies = len(scoring_strategies)
    bar_height = 0.7 / n_strategies
    y = np.arange(len(combo_ids))

    for ax_bars, profile in zip(ax_bars_list, nab_profiles):
        profile_scores = scores[profile]

        all_scores = np.concatenate(list(profile_scores.values())) if combo_ids else np.array([0])
        valid_scores = all_scores[~np.isnan(all_scores)]
        min_score = min(valid_scores.min(), 0) if len(valid_scores) else 0
        max_score = valid_scores.max() if len(valid_scores) else 1
        score_range = max_score - min_score
        xlim_min = (min_score - 0.25 * score_range) if min_score < 0 else min_score
        ax_bars.set_xlim(xlim_min, max_score + 0.3 * score_range)

        for i, strategy in enumerate(scoring_strategies):
            offset = (i - n_strategies / 2 + 0.5) * bar_height
            strategy_display_name = SCORING_STRATEGY_DISPLAY_NAME_MAP.get(strategy, strategy)
            bars = ax_bars.barh(y + offset, profile_scores[strategy], bar_height,
                                color=SCORING_STRATEGY_COLOR_MAP.get(strategy, 'gray'),
                                label=strategy_display_name)
            bar_value_labels = [
                '' if np.isnan(score) else f'{score:.2f}'
                for score in profile_scores[strategy]
            ]
            ax_bars.bar_label(bars, labels=bar_value_labels, fontsize=TICK_FONT_SIZE, padding=2)
            _round_bar_corners(bars)

        ax_bars.set_xlabel(
            f'NAB score\nunder {NAB_PROFILE_DISPLAY_NAME_MAP.get(profile, profile)} profile',
            fontsize=TITLE_FONT_SIZE
        )
        ax_bars.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
        ax_bars.tick_params(axis='y', labelleft=False, length=0)
        ax_bars.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.6)

    # Confusion-matrix table, styled like ax_table: same per-combo rows and
    # the same upper/lower triangle split per scoring strategy, but with a
    # leading encoded-combo-code column followed by TP/TN/FP/FN columns
    # instead of ground-truth anomaly indices. Each triangle is tinted with
    # its strategy's raw color at low opacity and labeled with that
    # strategy's count for the cell's metric
    label_col_idx = 0
    # Same blending as the ground-truth column labels' anomaly-source
    # background, but with a fixed blue tint rather than one that varies per
    # anomaly group
    label_cell_color = np.array(to_rgb('blue')) * ANOMALY_GROUP_OPACITY + \
                        np.array(undetected_color) * (1 - ANOMALY_GROUP_OPACITY)
    for row_idx, combo_id in enumerate(combo_ids):
        # The encoded combo code doesn't vary by scoring strategy, so its cell
        # is a single flat rectangle rather than a strategy-split triangle pair
        vertices = [
            (label_col_idx - 0.5, row_idx - 0.5), (label_col_idx + 0.5, row_idx - 0.5),
            (label_col_idx + 0.5, row_idx + 0.5), (label_col_idx - 0.5, row_idx + 0.5),
        ]
        ax_confusion.add_patch(plt.Polygon(vertices, closed=True, facecolor=label_cell_color,
                                           edgecolor='none', zorder=1))
        ax_confusion.text(label_col_idx, row_idx, combo_codes[row_idx], ha='center', va='center',
                          fontsize=TICK_FONT_SIZE - 4, color='black', zorder=3)

        for strategy in scoring_strategies:
            confusion = confusion_by_combo[combo_id].get(strategy)
            if confusion is None:
                continue

            triangle = SCORING_STRATEGY_TRIANGLE_MAP[strategy]
            strategy_color = SCORING_STRATEGY_COLOR_MAP_BOLD.get(strategy, 'gray')
            for metric_idx, column in enumerate(CONFUSION_MATRIX_COLUMNS):
                x, y = metric_idx + 1, row_idx
                if triangle == 'upper':
                    text_x, text_y = x + 1 / 6, y - 1 / 6
                else:
                    text_x, text_y = x - 1 / 6, y + 1 / 6
                # No cell fill: the strategy is instead identified by the
                # text color, matching that strategy's bar/legend color
                # -45 degrees aligns the label with the cell's top-left-to-bottom-right
                # diagonal (the shared edge between the two triangles)
                ax_confusion.text(text_x, text_y, str(confusion[column]), ha='center', va='center',
                                 rotation=-45, fontsize=TICK_FONT_SIZE - 4, color=strategy_color, zorder=3)

    # The diagonal split only applies to the strategy-specific TP/TN/FP/FN
    # columns, not the leading encoded-combo-code column (range starts at 1)
    for col_idx in range(1, n_confusion_columns):
        for row_idx in range(len(combo_ids)):
            ax_confusion.plot([col_idx - 0.5, col_idx + 0.5], [row_idx - 0.5, row_idx + 0.5],
                              color='lightgray', linewidth=0.5, zorder=2)

    # One data unit is one cell along both axes, so equal aspect keeps every cell square
    ax_confusion.set_xlim(-0.5, n_confusion_columns - 0.5)
    ax_confusion.set_aspect('equal', adjustable='box')
    ax_confusion.set_xticks(range(n_confusion_columns))
    ax_confusion.set_xticklabels(
        [CONFUSION_MATRIX_LABEL_COLUMN_HEADER] + CONFUSION_MATRIX_COLUMN_LABELS, fontsize=TICK_FONT_SIZE
    )
    ax_confusion.set_xticks(np.arange(-0.5, n_confusion_columns, 1), minor=True)
    ax_confusion.set_yticks(np.arange(-0.5, len(combo_ids), 1), minor=True)
    ax_confusion.grid(which='minor', color='lightgray', linewidth=0.5)
    ax_confusion.tick_params(which='minor', length=0)
    # Rows are numbered in display order (matching ax_table's rows) now that
    # the encoded combo code is shown inside the table's leading column
    ax_confusion.set_yticks(range(len(combo_ids)))
    ax_confusion.set_yticklabels(row_labels, fontsize=TICK_FONT_SIZE)
    ax_confusion.set_title(strategy_triangle_note, fontsize=TICK_FONT_SIZE)

    # Each panel gets its own legend, anchored above that panel's own title
    # rather than sharing one figure-wide legend; the scoring-strategy legend
    # only needs to appear once, above the first bar panel
    anomaly_source_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=ANOMALY_GROUP_OPACITY,
                      label=ANOMALY_GROUP_DISPLAY_NAME_MAP[key])
        for key, color in ANOMALY_GROUP_COLOR_MAP.items()
    ]
    ax_table.legend(
        handles=anomaly_source_handles, title='Anomaly Source',
        fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
        loc='lower center', bbox_to_anchor=(0.5, 1.1),
        ncol=len(anomaly_source_handles), frameon=True,
    )
    ax_bars_list[0].legend(
        title='Scoring strategy',
        fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
        loc='lower center', bbox_to_anchor=(0.5, 1.1),
        ncol=n_strategies, frameon=True,
    )

    fig.tight_layout()

    # set_aspect('equal') on ax_table shrinks its box to keep cells square, so
    # after layout it is usually narrower than its allocated slot, leaving a gap
    # before the bar panels; shift every bar panel left by that same gap (so
    # their relative widths/spacing are preserved) and match ax_table's height
    # so all panels line up row-for-row with minimal gap between them
    fig.canvas.draw()
    table_bbox = ax_table.get_position()
    panel_gap = 0.02
    first_bar_bbox = ax_bars_list[0].get_position()
    shift = first_bar_bbox.x0 - (table_bbox.x1 + panel_gap)
    for ax_bars in ax_bars_list:
        bbox = ax_bars.get_position()
        ax_bars.set_position([bbox.x0 - shift, table_bbox.y0, bbox.width, table_bbox.height])

    # ax_confusion also uses equal aspect (square cells), so rather than reuse
    # its post-draw box directly (which may not have shrunk to the same cell
    # size as ax_table), its width is rederived from ax_table's cell size —
    # same physical row height, scaled to the confusion table's column count —
    # then it's placed flush against ax_table's left edge (no gap), so the two
    # tables read as one continuous table
    fig_width_inches, fig_height_inches = fig.get_size_inches()
    confusion_width = (
        table_bbox.height * fig_height_inches * n_confusion_columns
        / (len(combo_ids) * fig_width_inches)
    )
    ax_confusion.set_position([
        table_bbox.x0 - confusion_width, table_bbox.y0, confusion_width, table_bbox.height
    ])

    out_path = os.path.join(
        merged_results_dir,
        f'table_and_nab_score_of_ensembles_window_{sliding_window}_{"_".join(nab_profiles)}.png'
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Combined ensembles detected anomalies table and NAB score plot saved to {out_path}')

    return out_path, ensemble_summary_df


def _normalize_ensemble_subsets(subset_strings):
    subsets = []
    for subset_string in subset_strings:
        http_code, agg = subset_string.replace('_', ' ').split()
        subsets.append((http_code, agg))
    return frozenset(subsets)


def export_selected_ensembles_to_latex_for_each_scoring_strategy(
        ensemble_summary_df, selected_ensembles, results_dir,
):
    """
    selected_ensembles maps a group key to either a curated list of ensembles
    (each given as e.g. ['5xx_count', '4xx avg']) or None:
      - int key N: filter to ensembles combining exactly N subsets; None means
        keep every N-subset ensemble, a list means keep only those given
      - 'All': same as an int key, but N is the largest subset count present
        in ensemble_summary_df (i.e. the ensemble combining every subset)
    """
    latex_dir = os.path.join(results_dir, 'merged_results', 'latex')
    os.makedirs(latex_dir, exist_ok=True)

    detected_columns = ['issue_detected', 'im_detected', 'TestLog_detected']
    header = [
        'Ens.', 'TP', 'TN', 'FP', 'FN',
        'Standard NAB', 'Reward FN NAB',
        'Issue Tracker', 'Instant Messenger', 'Test Log',
        'Avg Alarms/Day',
    ]
    # Test period spans 92 days; every predicted-positive (TP or FP) counts as
    # an alarm, so the average daily alarm volume is (TP + FP) / 92
    EVALUATION_PERIOD_DAYS = 92

    def ensemble_row_to_latex(row):
        # ensemble_id's subset order just follows however the ensemble was
        # originally combined, not any canonical order, so sort for a
        # consistent, predictable display order
        ensemble_subsets = sorted(_parse_ensemble_combo_subsets(row['ensemble_id']))
        ensemble_label = ','.join(
            text_subset_wrapper_latex(hc, agg) for hc, agg in ensemble_subsets
        )
        # LaTeX tabular cells don't render embedded newlines without extra
        # packages, so the "count/total\n[ids]" text is flattened to one line
        detected_cells = [row[col].replace('\n', ' ') for col in detected_columns]
        avg_alarms_per_day = (row['tp'] + row['fp']) / EVALUATION_PERIOD_DAYS
        return (
            f"{ensemble_label} & {row['tp']} & {row['tn']} & {row['fp']} & {row['fn']} & "
            f"{row['standard_nab_score']:.2f} & {row['reward_fn_nab_score']:.2f} & "
            f"{detected_cells[0]} & {detected_cells[1]} & {detected_cells[2]} & "
            f"{avg_alarms_per_day:.2f} \\\\"
        )

    # Each scoring strategy gets its own file; within a strategy, every
    # ensemble selected by any group in selected_ensembles is combined into a
    # single flat table (one row per ensemble, no per-group subsections)
    out_paths = {}
    for strategy, strategy_df in ensemble_summary_df.groupby('scoring_strategy'):
        subset_counts = strategy_df['ensemble_id'].apply(
            lambda cid: len(_parse_ensemble_combo_subsets(cid))
        )
        max_subset_count = subset_counts.max() if not strategy_df.empty else 0

        selected_ensemble_ids = set()
        for group_key, wanted_ensembles in selected_ensembles.items():
            # 'All' means the ensemble combining every subset, i.e. the group
            # whose subset count equals the largest one present in the data
            subset_count = max_subset_count if group_key == 'All' else group_key
            group_df = strategy_df[subset_counts == subset_count]
            if wanted_ensembles is not None:
                wanted_subset_sets = [_normalize_ensemble_subsets(e) for e in wanted_ensembles]
                group_df = group_df[
                    group_df['ensemble_id'].apply(
                        lambda cid: frozenset(_parse_ensemble_combo_subsets(cid)) in wanted_subset_sets
                    )
                ]
            selected_ensemble_ids.update(group_df['ensemble_id'])

        if not selected_ensemble_ids:
            continue

        combined_df = strategy_df[strategy_df['ensemble_id'].isin(selected_ensemble_ids)]
        # Order rows by subset count first (ascending), then alphabetically,
        # matching plot_table_and_nab_score_of_ensembles' row order
        combined_df = combined_df.assign(
            _subset_count=combined_df['ensemble_id'].apply(
                lambda cid: len(_parse_ensemble_combo_subsets(cid))
            )
        ).sort_values(['_subset_count', 'ensemble_id'])

        lines = [
            '\\begin{tabular}{l' + 'r' * (len(header) - 1) + '}',
            '\\toprule',
            ' & '.join(header) + ' \\\\',
            '\\midrule',
        ]
        # An empty row needs the same number of & separators as a real row
        # (len(header) - 1) so the column count stays consistent
        empty_row = ' & '.join([''] * len(header)) + ' \\\\'
        for _, row in combined_df.iterrows():
            lines.append(ensemble_row_to_latex(row))
            lines.append(empty_row)
        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')

        out_path = os.path.join(latex_dir, f'selected_ensembles_{strategy}.tex')
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'Selected {strategy} ensembles LaTeX table saved to {out_path}')
        out_paths[strategy] = out_path

    return out_paths


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    supported_models  = cfg.supported_models
    supported_sliding_windows = cfg.supported_sliding_windows

    # graph_models = ['T-GCN','ST-GCN','A3TGCN','GDN','MTAD-GAT','STformer']
    graph_models = ['A3TGCN']
    missing_imputation_stategies = ['zero','mean','median']
    http_codes = ['5xx','4xx']
    aggregations = ['count','avg','min','max']
    null_padding_features = [True]
    null_padding_targets = [False]
    results_dir = cfg.evaluation.model_save_path
    results_dir = os.path.join(get_project_root(), results_dir)

    use_existing_file = cfg.plotting.use_existing_file
    merge_computation_time(results_dir, supported_models, supported_sliding_windows,
                           missing_imputation_stategies,
                           http_codes, aggregations, graph_models,
                           null_padding_features,
                           null_padding_targets,
                           use_existing_file=use_existing_file
                           )
    plot_computation_time(results_dir, supported_models, supported_sliding_windows,
                          http_codes, aggregations, graph_models,
                          null_padding_features, null_padding_targets, is_one_column_figure=True)

    aggregations=['count']

    plot_computation_time_combined(results_dir, supported_models, supported_sliding_windows,
                                   http_codes, aggregations, graph_models,
                                   null_padding_features, null_padding_targets, is_one_column_figure=True)

    # plot_computation_time(results_dir, supported_models, supported_sliding_windows,
    #                       http_codes, aggregations, graph_models,
    #                       null_padding_features,
    #                       null_padding_targets)

    proposed_model_name = 'A3TGCN'
    proposed_model_detail = dict(
        proposed_model_name = proposed_model_name,
        null_padding_feature = True,
        null_padding_target = False,
        is_graph = proposed_model_name in graph_models,
    )

    aggregations = ['count', 'avg', 'min', 'max']
    optimal_hyperparameters_df = load_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        proposed_model_detail,
        results_dir,
        supported_sliding_windows,
        http_codes,
        aggregations,
        missing_imputation_stategies,
        use_existing_file=use_existing_file
    )

    most_optimal_hyperparameters_df = extract_most_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(optimal_hyperparameters_df,
                                                                                                                                             results_dir,
                                                                                                                                             proposed_model_detail,
                                                                                                                                  )
    save_most_optimal_hyperparameters_to_latex(most_optimal_hyperparameters_df,
                                               results_dir,
                                               selected_sliding_window=6,
                                               shown_parameter_columns = ['anomaly_threshold']
                                               )

    # most_optimal_hyperparameters_df = load_shorten_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
    #     proposed_model_detail,
    #     results_dir,
    #     supported_sliding_windows,
    #     http_codes,
    #     aggregations,
    #     missing_imputation_stategies,
    #     use_existing_file=False
    # )

    supported_models = ['GRU','A3TGCN']
    http_codes = ['5xx','4xx']
    aggregations = ['count']
    supported_sliding_windows = [6,12,18,24,30]
    null_padding_features= [True]
    null_padding_targets = [False]
    nab_profiles = ['reward_fn']
    compare_model_performance_across_sliding_windows(
        optimal_hyperparameters_df,
        results_dir,
        supported_models,
        graph_models,
        supported_sliding_windows,
        http_codes,
        aggregations,
        missing_imputation_stategies,
        null_padding_features,
        null_padding_targets,
        nab_profiles,
        is_one_column_figure=True
    )

    supported_models = ['GRU', 'A3TGCN']
    http_codes = ['5xx', '4xx']
    aggregations = ['count','avg','min','max']
    supported_sliding_windows = [6]
    null_padding_features = [True]
    null_padding_targets = [False]
    # nab_profiles = ['reward_fn','standard']
    nab_profile = 'reward_fn'
    sliding_window = 6
    plot_table_of_detected_anomalies(
        most_optimal_hyperparameters_df,
        results_dir,
        sliding_window,
        nab_profile,
        is_one_column_figure=True
    )
    plot_nab_score_of_optimal_configuration(
        most_optimal_hyperparameters_df,
        results_dir,
        sliding_window,
        nab_profile,
        is_one_column_figure=True
    )

    nab_profiles=['reward_fn']
    plot_table_and_nab_score_together(
        most_optimal_hyperparameters_df,
        results_dir,
        sliding_window,
        nab_profiles,
        is_one_column_figure=False
    )

    _, ensemble_df = plot_table_and_nab_score_of_ensembles(
        results_dir,
        sliding_window,
        nab_profiles,
        top_k=2,
        is_one_column_figure=False
    )

    selected_ensembles = {
        2: [['5xx_count', '4xx count'], ['5xx_count', '4xx avg']],
        3: [['5xx avg', '5xx max', '4xx avg']],
        4: [['5xx count', '5xx avg', '5xx min', '4xx avg'],['5xx count', '5xx avg', '5xx min', '4xx max']],
        'All': None
    }
    export_selected_ensembles_to_latex_for_each_scoring_strategy(ensemble_df, selected_ensembles, results_dir)





if __name__ == '__main__':
    main()