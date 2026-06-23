import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils import get_project_root


def plot_computation_time(results_dir, supported_models, supported_sliding_windows,
                          imputation_strategies, http_codes, aggregations, graph_models):
    # Collect computation times keyed by (model, window)
    data = {model: {w: [] for w in supported_sliding_windows} for model in supported_models}
    inference_data = {model: {w: [] for w in supported_sliding_windows} for model in supported_models}

    index_columns = ['model_name', 'is_graph_model', 'sliding_window', 'http_code', 'aggregation', 'imputation_strategy']
    merged_df = pd.DataFrame()
    merged_results_dir = os.path.join(results_dir, 'merged_results')
    os.makedirs(merged_results_dir, exist_ok=True)

    for window in supported_sliding_windows:
        for http_code in http_codes:
            for agg in aggregations:
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
                            training_time = df['training_time'].values[0]
                            data[model][window].append(training_time)

                            row_df = pd.DataFrame([{**base, 'training_time': training_time, 'epochs': df['epochs'].values[0]}])
                            row_df.set_index(index_columns, inplace=True)
                            merged_df = row_df.combine_first(merged_df)

                        if os.path.exists(inference_time_csv_path):
                            inference_df = pd.read_csv(inference_time_csv_path, index_col=0)
                            inference_time = inference_df['inference_time'].values[0]
                            inference_data[model][window].append(inference_time)

                            row_df = pd.DataFrame([{**base, 'inference_time': inference_time}])
                            row_df.set_index(index_columns, inplace=True)
                            merged_df = row_df.combine_first(merged_df)

    csv_saved_path = os.path.join(merged_results_dir,'computation_time_comparision.csv')
    merged_df.to_csv(csv_saved_path, index=True)
    print(f'Training time plot saved to {csv_saved_path}')

    # Average over http_codes / aggregations / imputation strategies
    means = {
        model: {w: np.mean(times) if times else np.nan for w, times in windows.items()}
        for model, windows in data.items()
    }
    inference_means = {
        model: {w: np.mean(times) if times else np.nan for w, times in windows.items()}
        for model, windows in inference_data.items()
    }

    valid_models = [m for m in supported_models
                    if not all(np.isnan(v) for v in means[m].values())]

    n_windows = len(supported_sliding_windows)
    bar_width = 0.7 / n_windows
    x = np.arange(len(valid_models))
    colors = plt.cm.tab10(np.linspace(0, 0.45, n_windows))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5, 5))

    for i, window in enumerate(supported_sliding_windows):
        offset = (i - n_windows / 2 + 0.5) * bar_width
        ax.bar(x + offset, [means[m][window] for m in valid_models],
               bar_width, label=f'Window {window}', color=colors[i])
        ax2.bar(x + offset, [inference_means[m][window] for m in valid_models],
                bar_width, label=f'Window {window}', color=colors[i])

    for axis, title, ylabel in [
        
        (ax,  'Training Time',  'Training Time (s)'),
        (ax2, 'Inference Time', 'Inference Time (s)'),
    ]:
        axis.set_xticks(x)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_title(title, fontsize=10, fontweight='bold')
        axis.legend(title='Window', fontsize=7, title_fontsize=7,
                    loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
        axis.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
        axis.tick_params(axis='y', labelsize=8)
        axis.set_xticklabels(valid_models, rotation=30, ha='right', fontsize=8)

    fig.tight_layout()


    out_path = os.path.join(merged_results_dir, 'computation_time_comparision.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training time plot saved to {out_path}')
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