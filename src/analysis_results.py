import colorsys
import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    'reward_fn': 'Reward FN',
}

TITLE_FONT_SIZE = 9
LEGEND_FONT_SIZE = TITLE_FONT_SIZE - 2
TICK_FONT_SIZE = TITLE_FONT_SIZE - 2
AXIS_LABEL_FONT_SIZE = TITLE_FONT_SIZE - 2


def text_subset_wrapper(http_code, agg):
    return "\\mathtt{" + f'{http_code}\ {agg}' + "}"


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
                           ):
    _merge_computation_time_data(results_dir, supported_models, supported_sliding_windows,
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
                           fontsize=TICK_FONT_SIZE - 2, rotation=90, padding=2,
                           label_type='edge')
        ax_infer.bar_label(infer_bars, labels=[format_bar_value(v) for v in infer_values],
                           fontsize=TICK_FONT_SIZE - 2, rotation=90, padding=2,
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
):
    model_name = proposed_model_detail['proposed_model_name']
    null_padding_feature = proposed_model_detail['null_padding_feature']
    null_padding_target = proposed_model_detail['null_padding_target']
    is_graph = proposed_model_detail['is_graph']

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

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    csv_saved_path = os.path.join(merged_results_dir, f'{model_name}_optimal_scoring_hyperparameters.csv')
    optimal_hyperparameters_df.to_csv(csv_saved_path, index=False)
    print(f'Optimal scoring hyperparameters CSV saved to {csv_saved_path}')

    return optimal_hyperparameters_df


def load_shorten_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        proposed_model_detail, results_dir, supported_sliding_windows,
        http_codes, aggregations, missing_imputation_strategies,
):
    model_name = proposed_model_detail['proposed_model_name']
    null_padding_feature = proposed_model_detail['null_padding_feature']
    null_padding_target = proposed_model_detail['null_padding_target']
    is_graph = proposed_model_detail['is_graph']

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

    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)
    csv_saved_path = os.path.join(merged_results_dir, f'{model_name}_shorten_optimal_scoring_hyperparameters.csv')
    shorten_optimal_hyperparameters_df.to_csv(csv_saved_path, index=False)
    print(f'Shorten optimal scoring hyperparameters CSV saved to {csv_saved_path}')

    return shorten_optimal_hyperparameters_df


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

    # Each model family (base model name) gets a maximally distinct hue, evenly
    # spaced around the color wheel; variants within a family (e.g. the graph
    # model's null-padding variants) share that hue but differ in lightness so
    # they read as related while still being distinguishable.
    family_hues = np.linspace(0, 1, len(supported_models), endpoint=False)
    model_colors = {}
    for family_hue, model_name in zip(family_hues, supported_models):
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
    merge_computation_time(results_dir, supported_models, supported_sliding_windows,
                           missing_imputation_stategies,
                           http_codes, aggregations, graph_models,
                           null_padding_features,
                           null_padding_targets
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

    optimal_hyperparameters_df = load_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        proposed_model_detail,
        results_dir,
        supported_sliding_windows,
        http_codes,
        aggregations,
        missing_imputation_stategies,
    )
    shorten_optimal_hyperparameters_df = load_shorten_optimal_scoring_hyperparameter_of_the_proposed_model_on_each_subset_and_sliding_window(
        proposed_model_detail,
        results_dir,
        supported_sliding_windows,
        http_codes,
        aggregations,
        missing_imputation_stategies,
    )

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


if __name__ == '__main__':
    main()