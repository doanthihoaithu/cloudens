import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils import get_project_root


def plot_computation_time(results_dir, supported_models, supported_sliding_windows,
                          imputation_strategies, http_codes, aggregations, graph_models):
    index_columns = ['model_name', 'is_graph_model', 'sliding_window', 'http_code', 'aggregation', 'imputation_strategy']
    merged_df = pd.DataFrame()
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)

    # Collect times keyed by (http_code, agg) subset → model → window
    subsets = [(hc, agg) for hc in http_codes for agg in aggregations]
    train_data = {s: {m: {w: [] for w in supported_sliding_windows} for m in supported_models} for s in subsets}
    infer_data = {s: {m: {w: [] for w in supported_sliding_windows} for m in supported_models} for s in subsets}

    for window in supported_sliding_windows:
        for http_code in http_codes:
            for agg in aggregations:
                subset = (http_code, agg)
                for imputation in imputation_strategies:
                    for model in supported_models:
                        base = dict(
                            model_name=model,
                            is_graph_model=model in graph_models,
                            sliding_window=window,
                            http_code=http_code,
                            aggregation=agg,
                            imputation_strategy=imputation,
                        )
                        training_time_csv_path = os.path.join(
                            get_project_root(),
                            results_dir,
                            f'window_{window}',
                            f'no_group_{http_code}_{agg}',
                            f'fill_nan_with_{imputation}',
                            model,
                            f'{model}_training_time.csv'
                        )
                        inference_time_csv_path = os.path.join(
                            get_project_root(),
                            results_dir,
                            f'window_{window}',
                            f'no_group_{http_code}_{agg}',
                            f'fill_nan_with_{imputation}',
                            model,
                            f'inference_time.csv'
                        )

                        if os.path.exists(training_time_csv_path):
                            df = pd.read_csv(training_time_csv_path)
                            t = df['training_time'].values[0]
                            train_data[subset][model][window].append(t)
                            row_df = pd.DataFrame([{**base, 'training_time': t, 'epochs': df['epochs'].values[0]}])
                            row_df.set_index(index_columns, inplace=True)
                            merged_df = row_df.combine_first(merged_df)

                        if os.path.exists(inference_time_csv_path):
                            inference_df = pd.read_csv(inference_time_csv_path, index_col=0)
                            t = inference_df['inference_time'].values[0]
                            infer_data[subset][model][window].append(t)
                            row_df = pd.DataFrame([{**base, 'inference_time': t}])
                            row_df.set_index(index_columns, inplace=True)
                            merged_df = row_df.combine_first(merged_df)

    csv_saved_path = os.path.join(merged_results_dir, 'computation_time_comparision.csv')
    merged_df.to_csv(csv_saved_path, index=True)
    print(f'Merged CSV saved to {csv_saved_path}')

    # One row per (http_code, agg) subset; 2 columns: training time | inference time
    n_rows = len(subsets)
    n_windows = len(supported_sliding_windows)
    bar_width = 0.7 / n_windows
    colors = plt.cm.tab10(np.linspace(0, 0.45, n_windows))

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (http_code, agg) in enumerate(subsets):
        subset = (http_code, agg)
        ax_train = axes[row_idx, 0]
        ax_infer = axes[row_idx, 1]

        train_means = {
            m: {w: np.mean(vs) if vs else np.nan for w, vs in train_data[subset][m].items()}
            for m in supported_models
        }
        infer_means = {
            m: {w: np.mean(vs) if vs else np.nan for w, vs in infer_data[subset][m].items()}
            for m in supported_models
        }

        valid_models = [m for m in supported_models
                        if not all(np.isnan(v) for v in train_means[m].values())]
        x = np.arange(len(valid_models))

        for i, window in enumerate(supported_sliding_windows):
            offset = (i - n_windows / 2 + 0.5) * bar_width
            ax_train.bar(x + offset, [train_means[m][window] for m in valid_models],
                         bar_width, label=f'Win {window}', color=colors[i])
            ax_infer.bar(x + offset, [infer_means[m][window] for m in valid_models],
                         bar_width, label=f'Win {window}', color=colors[i])

        row_label = f'{http_code} / {agg}'
        for ax, ylabel in [(ax_train, 'Training Time (s)\n[log scale]'), (ax_infer, 'Inference Time (s)\n[log scale]')]:
            ax.set_xticks(x)
            ax.set_xticklabels(valid_models if valid_models else [], rotation=30, ha='right', fontsize=8)
            ax.set_yscale('log')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
            ax.tick_params(axis='y', labelsize=8)

        ax_train.set_title(f'Training Time — {row_label}', fontsize=9, fontweight='bold')
        ax_infer.set_title(f'Inference Time — {row_label}', fontsize=9, fontweight='bold')

    # Single shared legend at the top — same color palette visible once for all subplots
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f'Win {w}')
        for i, w in enumerate(supported_sliding_windows)
    ]
    fig.legend(handles=legend_handles, title='Window', fontsize=8, title_fontsize=8,
               loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=n_windows, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(merged_results_dir, 'computation_time_comparision.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Computation time plot saved to {out_path}')
    return out_path


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    supported_models  = cfg.supported_models
    supported_sliding_windows = cfg.supported_sliding_windows

    graph_models = ['T-GCN','ST-GCN','A3TGCN','GDN','MTAD-GAT','STformer']
    missing_imputation_stategies = ['zero','mean','median']
    http_codes = ['5xx','4xx','2xx']
    aggregations = ['count','avg','min','max']
    results_dir = cfg.evaluation.model_save_path
    results_dir = os.path.join(get_project_root(), results_dir)
    plot_computation_time(results_dir, supported_models, supported_sliding_windows, missing_imputation_stategies, http_codes, aggregations, graph_models)


if __name__ == '__main__':
    main()