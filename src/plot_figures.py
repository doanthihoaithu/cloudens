import itertools
import os

import hydra
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from analysis_results import text_subset_wrapper_latex, text_subset_wrapper, SCORING_STRATEGY_DISPLAY_FULL_NAME_MAP, \
    MODEL_DISPLAY_NAME_MAP
from ibm_dataset_loader import IBMDatasetLoader
from run_training_single_model import label_reconstruction_errors
from utils import get_project_root

FONT_SIZE = 12

_ANOMALY_SOURCE_COLORS = {
    1: ('green',  'Issue Tracker'),
    2: ('orange', 'Instant Messenger'),
    3: ('violet', 'Test Log'),
}


def _plot_source_spans(ax, anomaly_windows_test, alpha=0.15):
    """Draw axvspan for each test anomaly window, colored by anomaly_source."""
    for _, win in anomaly_windows_test.iterrows():
        src = int(win['anomaly_source'])
        color, _ = _ANOMALY_SOURCE_COLORS.get(src, ('grey', 'Unknown'))
        ax.axvspan(win['anomaly_window_start'], win['anomaly_window_end'],
                   color=color, alpha=alpha, zorder=0)


def _source_legend_handles():
    return [
        mpatches.Patch(color=color, alpha=0.5, label=label)
        for color, label in _ANOMALY_SOURCE_COLORS.values()
    ]


def _anomaly_spans(labels, index):
    arr = np.array(labels).ravel().astype(float)
    arr = np.where(np.isnan(arr), 0.0, arr)
    spans, in_block, start = [], False, None
    for i, v in enumerate(arr):
        if v == 1 and not in_block:
            start, in_block = i, True
        elif v != 1 and in_block:
            spans.append((index[start], index[i - 1]))
            in_block = False
    if in_block:
        spans.append((index[start], index[-1]))
    return spans


def plot_reconstruction_errors(dataloader, results_dir, model_name,
                                http_code, aggregation, fill_nan, slide_win,
                                shown_nodes=None):
    """
    Load reconstruction_errors.npy saved during inference and plot the mean
    per-timestep reconstruction error together with ground-truth anomaly regions.

    Parameters
    ----------
    dataloader : IBMDatasetLoader
        Provides test_labels (binary Series) and test_index (DatetimeIndex).
    results_dir : str
        Root of the model-save tree (e.g. the absolute path to 'trained_models').
    model_name : str
    http_code : str   e.g. '5xx'
    aggregation : str e.g. 'count'
    fill_nan : str    e.g. 'zero'
    slide_win : int
    """
    model_dir = os.path.join(
        results_dir,
        f'window_{slide_win}',
        f'no_group_{http_code}_{aggregation}',
        f'fill_nan_with_{fill_nan}',
        model_name,
    )
    recon_path = os.path.join(model_dir, 'reconstruction_errors.npy')
    if not os.path.exists(recon_path):
        print(f'reconstruction_errors.npy not found at {recon_path}')
        return None

    # reconstruction_errors: [total, N, F] → collapse to 1-D score
    recon_errors = np.load(recon_path)             # [T, N, F]
    recon_score = recon_errors.mean(axis=-1).mean(axis=-1)  # [T]

    test_labels = np.array(dataloader.test_labels).ravel()
    test_index = np.array(dataloader.test_index)

    # Align lengths (windowing may produce one sample per test timestep)
    n = min(len(recon_score), len(test_labels), len(test_index))
    recon_errors = recon_errors[:n]
    recon_score  = recon_score[:n]
    test_labels  = test_labels[:n]
    test_index   = test_index[:n]

    # ── Anomalous-dimension CSV ───────────────────────────────────────────────
    node_ids   = dataloader.meta_data['node_ids']           # length N
    feat_names = dataloader.meta_data['node_feature_names'] # length F

    # For each timestep find the (node, feature) with the highest error
    T, N, F = recon_errors.shape
    flat      = recon_errors.reshape(T, -1)                   # [T, N*F]
    flat_idx  = flat.argmax(axis=1)                           # [T]
    node_idx  = flat_idx // F
    feat_idx  = flat_idx % F
    max_errors     = flat[np.arange(T), flat_idx]             # [T]
    total_errors   = flat.sum(axis=1)                         # [T]
    contribution_pct = np.where(
        total_errors > 0,
        max_errors / total_errors * 100.0,
        0.0,
    )

    dim_df = pd.DataFrame({
        'timestamp':              test_index,
        'is_anomaly':             test_labels.astype(int),
        'max_recon_error':        max_errors,
        'total_recon_error':      total_errors,
        'max_contribution_pct':   contribution_pct,
        'node':                   [node_ids[i]   for i in node_idx],
        'feature':                [feat_names[i] for i in feat_idx],
    })
    csv_path = os.path.join(model_dir, 'anomalous_dimension.csv')
    dim_df.to_csv(csv_path, index=False)
    print(f'Saved anomalous dimension CSV to {csv_path}')

    spans = _anomaly_spans(test_labels, test_index)

    # ── Resolve node time series for each requested node ─────────────────────
    node_ids_list = list(node_ids)
    # Inverse-transform the full scaled matrix once to get raw values [timestamps, N*F]
    X_test_raw = dataloader.scaler.inverse_transform(dataloader.X_test_scaled)

    valid_nodes = []   # list of (node_name, series [n, F])
    for node_name in (shown_nodes or []):
        if node_name in node_ids_list:
            ni = node_ids_list.index(node_name)
            series = X_test_raw[:n, ni * F:(ni + 1) * F]
            valid_nodes.append((node_name, series))
        else:
            print(f'shown_node "{node_name}" not found in node_ids; skipping')

    n_rows = 1 + len(valid_nodes)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 4 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    # Always index axes as a list for uniform access
    if n_rows == 1:
        axes = [axes]

    # ── Reconstruction error subplot ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(test_index, recon_score, color='steelblue', linewidth=0.8)
    ax.fill_between(test_index, 0, recon_score, alpha=0.2, color='steelblue')

    anomaly_patch = None
    for t_start, t_end in spans:
        anomaly_patch = ax.axvspan(t_start, t_end, color='red', alpha=0.18, zorder=0)

    ax.set_ylabel('Mean Reconstruction Error', fontsize=10)
    ax.set_title(
        f'{model_name} — Reconstruction Error on Test Set\n'
        f'({http_code} / {aggregation} / imputation={fill_nan} / win={slide_win})',
        fontsize=11, fontweight='bold',
    )
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    recon_handles = [mpatches.Patch(color='steelblue', alpha=0.7, label='Recon error (mean)')]
    if anomaly_patch is not None:
        recon_handles.append(mpatches.Patch(color='red', alpha=0.4, label='Ground-truth anomaly'))
    ax.legend(handles=recon_handles, loc='upper right', fontsize=9)

    # ── One subplot per node ──────────────────────────────────────────────────
    feat_colors = [plt.colormaps['tab10'](i / max(F, 1)) for i in range(F)]
    for row, (node_name, node_series) in enumerate(valid_nodes, start=1):
        axn = axes[row]
        for fi, (fname, color) in enumerate(zip(feat_names, feat_colors)):
            axn.plot(test_index, node_series[:, fi], color=color,
                     linewidth=0.8, label=fname)
        for t_start, t_end in spans:
            axn.axvspan(t_start, t_end, color='red', alpha=0.18, zorder=0)
        axn.set_ylabel('Value (raw)', fontsize=10)
        axn.set_title(f'Node: {node_name}', fontsize=10, fontweight='bold')
        axn.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        axn.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time', fontsize=10)
    axes[-1].tick_params(axis='x', rotation=30)

    fig.tight_layout()
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, f'{model_name}_reconstruction_errors.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved reconstruction error plot to {out_path}')
    return out_path



def plot_only_node_series(dataloader, shown_node, out_dir=None,
                          zoom_period=None):
    """
    Plot the raw time-series of a single node (all features) with anomalous
    regions highlighted.

    Parameters
    ----------
    dataloader : IBMDatasetLoader
        Must have been used with get_index_dataset() so that X_test_scaled,
        scaler, test_labels, test_index, and meta_data are populated.
    shown_node : str
        Node name as it appears in dataloader.meta_data['node_ids'].
    out_dir : str | None
        Directory to save the figure.  Defaults to the current directory.
    zoom_period : tuple(start_date, end_date) | None
        If given, a second subplot is added below showing only that date range.
        Values can be anything accepted by pd.Timestamp (str, datetime, etc.).
    """
    node_ids   = dataloader.meta_data['node_ids']
    feat_names = dataloader.meta_data['node_feature_names']
    node_ids_list = list(node_ids)

    if shown_node not in node_ids_list:
        print(f'plot_only_node_series: "{shown_node}" not found in node_ids')
        return None

    F = len(feat_names)
    ni = node_ids_list.index(shown_node)

    X_test_raw = dataloader.scaler.inverse_transform(dataloader.X_test_scaled)
    test_index = np.array(dataloader.test_index)
    test_labels = np.array(dataloader.test_labels).ravel()

    n = min(len(test_index), len(test_labels), X_test_raw.shape[0])
    test_index  = test_index[:n]
    test_labels = test_labels[:n]
    node_series = X_test_raw[:n, ni * F:(ni + 1) * F]   # [n, F]

    feat_colors = [plt.colormaps['tab10'](i / max(F, 1)) for i in range(F)]

    # Build shared legend handles once
    legend_handles = [
        plt.Line2D([0], [0], color=feat_colors[i], linewidth=1.5, label=feat_names[i])
        for i in range(F)
    ]
    anom_handle = mpatches.Patch(color='red', alpha=0.4, label='Anomaly')

    def _draw_series(ax, idx, series, title, xlabel=False):
        anomaly_patch = None
        for fi, (fname, color) in enumerate(zip(feat_names, feat_colors)):
            ax.plot(idx, series[:, fi], color=color, linewidth=0.8, label=fname)
        for t_start, t_end in _anomaly_spans(test_labels, test_index):
            # Only draw spans that overlap with this axis range
            if t_end >= idx[0] and t_start <= idx[-1]:
                anomaly_patch = ax.axvspan(
                    max(t_start, idx[0]), min(t_end, idx[-1]),
                    color='red', alpha=0.18, zorder=0,
                )
        ax.set_ylabel('Value (raw)', fontsize=10)
        if xlabel:
            ax.set_xlabel('Time', fontsize=10)
            ax.tick_params(axis='x', rotation=30)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        handles = legend_handles + ([anom_handle] if anomaly_patch is not None else [])
        ax.legend(handles=handles, loc='upper right', fontsize=8)

    # ── Resolve zoom mask ────────────────────────────────────────────────────
    zoom_mask = None
    if zoom_period is not None:
        z_start = pd.Timestamp(zoom_period[0])
        z_end   = pd.Timestamp(zoom_period[1])
        zoom_mask = (pd.DatetimeIndex(test_index) >= z_start) & \
                    (pd.DatetimeIndex(test_index) <= z_end)
        if not zoom_mask.any():
            print(f'zoom_period {zoom_period} has no data in test range; skipping zoom')
            zoom_mask = None

    n_rows = 2 if zoom_mask is not None else 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(5, 2 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    filt = dataloader.data_preparation_config.features_prep.filter
    http_codes_used   = ', '.join(filt.http_codes)
    aggregations_used = ', '.join(filt.aggregations)
    fig.suptitle(
        f'Node time series: {shown_node}\n'
        f'http_code: {http_codes_used}  |  aggregation: {aggregations_used}',
        fontsize=11, fontweight='bold',
    )

    _draw_series(axes[0], test_index, node_series,
                 title='Full test period', xlabel=(zoom_mask is None))

    if zoom_mask is not None:
        z_idx    = test_index[zoom_mask]
        z_series = node_series[zoom_mask]
        _draw_series(axes[1], z_idx, z_series,
                     title=f'Zoom: {zoom_period[0]} → {zoom_period[1]}',
                     xlabel=True)

    save_dir = out_dir or '.'
    os.makedirs(save_dir, exist_ok=True)
    safe_name = shown_node.replace('/', '_')
    out_path = os.path.join(save_dir, f'{aggregations_used}_{safe_name}_node_series.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved node series plot to {out_path}')
    return out_path


def plot_only_multiple_node_series_zoom_in(dataloader, shown_nodes, out_dir=None,
                                           zoom_period=None):
    """
    Plot multiple node raw time-series within a specific zoom period,
    one subplot per node, with anomalous regions highlighted.

    Parameters
    ----------
    dataloader : IBMDatasetLoader
        Must have been used with get_index_dataset() so that X_test_scaled,
        scaler, test_labels, test_index, and meta_data are populated.
    shown_nodes : str | list[str]
        One or more node names from dataloader.meta_data['node_ids'].
    out_dir : str | None
        Directory to save the figure.  Defaults to the current directory.
    zoom_period : tuple(start_date, end_date) | None
        Date range to show.  Values accepted by pd.Timestamp (str, datetime…).
        If None the full test period is shown.
    """
    if isinstance(shown_nodes, str):
        shown_nodes = [shown_nodes]

    node_ids   = dataloader.meta_data['node_ids']
    feat_names = dataloader.meta_data['node_feature_names']
    node_ids_list = list(node_ids)
    F = len(feat_names)

    X_test_raw  = dataloader.scaler.inverse_transform(dataloader.X_test_scaled)
    test_index  = np.array(dataloader.test_index)
    test_labels = np.array(dataloader.test_labels).ravel()

    n = min(len(test_index), len(test_labels), X_test_raw.shape[0])
    test_index  = test_index[:n]
    test_labels = test_labels[:n]

    # ── Apply zoom mask ───────────────────────────────────────────────────────
    if zoom_period is not None:
        z_start = pd.Timestamp(zoom_period[0])
        z_end   = pd.Timestamp(zoom_period[1])
        mask = (pd.DatetimeIndex(test_index) >= z_start) & \
               (pd.DatetimeIndex(test_index) <= z_end)
        if not mask.any():
            print(f'plot_only_multiple_node_series_zoom_in: '
                  f'zoom_period {zoom_period} has no data; plotting full range')
            mask = np.ones(n, dtype=bool)
    else:
        mask = np.ones(n, dtype=bool)

    plot_index  = test_index[mask]
    plot_labels = test_labels[mask]
    spans       = _anomaly_spans(plot_labels, plot_index)

    # ── Collect valid nodes ───────────────────────────────────────────────────
    valid_nodes = []
    for node_name in shown_nodes:
        if node_name in node_ids_list:
            ni     = node_ids_list.index(node_name)
            series = X_test_raw[:n, ni * F:(ni + 1) * F][mask]
            valid_nodes.append((node_name, series))
        else:
            print(f'plot_only_multiple_node_series_zoom_in: '
                  f'"{node_name}" not found in node_ids; skipping')

    if not valid_nodes:
        print('No valid nodes to plot.')
        return None

    feat_colors = [plt.colormaps['tab10'](i / max(F, 1)) for i in range(F)]
    anom_handle = mpatches.Patch(color='red', alpha=0.4, label='Ground-truth anomaly')

    filt             = dataloader.data_preparation_config.features_prep.filter
    http_codes_used  = ', '.join(filt.http_codes)
    aggregations_used = ', '.join(filt.aggregations)

    period_label = (f'{zoom_period[0]} → {zoom_period[1]}'
                    if zoom_period is not None else 'Full test period')

    n_rows = len(valid_nodes)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(5, 1.5 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        f'Multivariate Time Series\n',
        fontsize=FONT_SIZE + 2, fontweight='bold',
    )

    any_anomaly = False
    for row, (node_name, node_series) in enumerate(valid_nodes):
        ax = axes[row]
        for fi, (fname, color) in enumerate(zip(feat_names, feat_colors)):
            ax.plot(plot_index, node_series[:, fi], color=color, linewidth=0.8)
        for t_start, t_end in spans:
            ax.axvspan(t_start, t_end, color='red', alpha=0.18, zorder=0)
            any_anomaly = True
        y_label = 'Count' if aggregations_used == 'count' else f'{aggregations_used} response time'
        ax.set_ylabel(y_label, fontsize=FONT_SIZE)
        ax.set_title(node_name, fontsize=FONT_SIZE-1)
        # ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        ax.tick_params(axis='y', labelsize=FONT_SIZE-1)

    axes[-1].set_xlabel('Time', fontsize=FONT_SIZE)
    axes[-1].tick_params(axis='x', rotation=30, labelsize=FONT_SIZE)

    # Single shared legend at the bottom of the figure
    shared_handles = [
        plt.Line2D([0], [0], color=feat_colors[i], linewidth=1.5, label=feat_names[i])
        for i in range(F)
    ]
    if any_anomaly:
        shared_handles.append(anom_handle)

    fig.legend(handles=shared_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.97), ncol=len(shared_handles),
               fontsize=FONT_SIZE, frameon=False)

    save_dir = out_dir or '.'
    os.makedirs(save_dir, exist_ok=True)
    period_tag = (f'{zoom_period[0]}_{zoom_period[1]}'.replace(' ', 'T').replace(':', '')
                  if zoom_period is not None else 'full')
    out_path = os.path.join(save_dir,
                            f'{aggregations_used}_multiple_nodes_{period_tag}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved multiple node series plot to {out_path}')
    return out_path


def plot_only_time_series(dataloader, out_dir=None,
                          zoom_periods=None
                          ):
    anomaly_windows_test = dataloader.anomaly_windows_test.copy()
    anomaly_windows_test['anomaly_window_start'] = pd.to_datetime(
        anomaly_windows_test['anomaly_window_start'])
    anomaly_windows_test['anomaly_window_end'] = pd.to_datetime(
        anomaly_windows_test['anomaly_window_end'])
    index = dataloader.test_index

    X_raw = dataloader.scaler.inverse_transform(dataloader.X_test_scaled)
    mean_series = X_raw.mean(axis=1)
    _ms_min, _ms_max = mean_series.min(), mean_series.max()
    mean_series = (mean_series - _ms_min) / max(_ms_max - _ms_min, 1e-8)

    T0 = min(len(mean_series), len(index))
    plot_index = pd.to_datetime(index[:T0])
    plot_series = mean_series[:T0]

    test_labels = np.array(dataloader.test_labels).ravel()[:T0]
    n_anomalous = int((test_labels == 1).sum())
    print(f'Number of anomalous timestamps in testing data: {n_anomalous} / {T0}')

    fig, ax = plt.subplots(figsize=(12, 2.5), constrained_layout=True)

    ax.plot(plot_index, plot_series, color='steelblue', linewidth=0.8)
    _plot_source_spans(ax, anomaly_windows_test, alpha=0.3)

    filt = dataloader.data_preparation_config.features_prep.filter
    http_codes_used = ', '.join(filt.http_codes)
    agg_used = ', '.join(filt.aggregations)

    ax.set_xlabel('Timestamp', fontsize=FONT_SIZE)
    ax.set_ylabel(f'Normalized $\\mathtt{{count}}$ [0,1]', fontsize=FONT_SIZE)
    # ax.set_title(f'Sum of ${http_codes_used}\,{agg_used}$ subset accross all dimensions',
    #              fontsize=FONT_SIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=30, labelsize=FONT_SIZE - 1)
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 1)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    # zoom_periods: list of (start, end) tuples
    _zoom_list = list(zoom_periods) if zoom_periods else []

    for period_i, (zs, ze) in enumerate(_zoom_list):
        zs = pd.to_datetime(zs)
        ze = pd.to_datetime(ze)
        mid = zs + (ze - zs) / 2

        # Find anomaly windows that overlap with this period
        overlapping_ids = [
            str(aw_idx-6)
            for aw_idx, win in anomaly_windows_test.iterrows()
            if win['anomaly_window_start'] <= ze and win['anomaly_window_end'] >= zs
        ]
        annotation = (f'Anomaly {", ".join(overlapping_ids)}'
                      if overlapping_ids else f'Period {period_i}')

        ax.annotate(
            annotation,
            xy=(mid, 0.02),
            xycoords=('data', 'axes fraction'),
            xytext=(mid, 0.22),
            textcoords=('data', 'axes fraction'),
            fontsize=FONT_SIZE - 2,
            color='red',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
        )

    fig.legend(handles=_source_legend_handles(), loc='upper right',
               bbox_to_anchor=(1.0, 1.15), ncol=len(_ANOMALY_SOURCE_COLORS),
               fontsize=FONT_SIZE - 1, frameon=True)

    save_dir = out_dir or '.'
    os.makedirs(save_dir, exist_ok=True)
    safe_http = http_codes_used.replace(',', '').replace(' ', '_')
    safe_agg = agg_used.replace(',', '').replace(' ', '_')
    out_path = os.path.join(save_dir, f'time_series_{safe_http}_{safe_agg}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved time series plot to {out_path}')
    return out_path


def plot_detected_anomalies(dataloader, results_dir, models, imputation_strategies,
                            http_codes, aggregations,
                            null_padding_features,
                            null_padding_targets,
                            out_dir=None,
                            fix_scoring_parameters=None):

    slide_win = dataloader.data_preparation_config.slide_win
    index = dataloader.test_index
    labels = np.array(dataloader.test_labels).ravel()
    anomaly_windows_test = dataloader.anomaly_windows_test

    X_raw = dataloader.scaler.inverse_transform(dataloader.X_test_scaled)
    mean_series = X_raw.mean(axis=1)
    _ms_min, _ms_max = mean_series.min(), mean_series.max()
    mean_series = (mean_series - _ms_min) / max(_ms_max - _ms_min, 1e-8)

    for http_code in http_codes:
        for agg in aggregations:
            for fill_nan in imputation_strategies:
                for null_padding_feature, null_padding_target in itertools.product(null_padding_features, null_padding_targets):
                # Each entry is type='recon' or type='score', grouped per model
                    all_panels = []

                    for model in models:
                        model_folder = model

                        if model in ['GRU']:
                            model_folder = model
                        elif model in ['A3TGCN']:
                            if null_padding_feature == False and null_padding_target == False:
                                model_folder = model
                            elif null_padding_feature == True and null_padding_target == False:
                                model_folder = f'{model}_null_padding_feature'
                            elif null_padding_feature == False and null_padding_target == True:
                                model_folder = f'{model}_null_padding_target'
                            else:
                                model_folder = f'{model}_null_padding_both'

                        model_dir = os.path.join(
                            results_dir,
                            f'window_{slide_win}',
                            f'no_group_{http_code}_{agg}',
                            f'fill_nan_with_{fill_nan}',
                            model_folder,
                        )
                        recon_path = os.path.join(model_dir, 'reconstruction_errors.npy')
                        maha_path = os.path.join(model_dir, 'mahalanobis.npy')

                        required_exists = os.path.exists(recon_path)
                        if fix_scoring_parameters is None:
                            csv_path = os.path.join(model_dir, f'{model}_grid_search.csv')
                            required_exists = required_exists and os.path.exists(csv_path)

                        if not required_exists:
                            print(f'Skipping {model}: missing files in {model_dir}')
                            continue

                        reconstruction_errors = np.load(recon_path)
                        mahalanobis = np.load(maha_path) if os.path.exists(maha_path) else np.zeros(len(reconstruction_errors))

                        if fix_scoring_parameters is None:
                            grid_df = pd.read_csv(csv_path)

                        T = min(len(reconstruction_errors), len(index))
                        recon = reconstruction_errors[:T]
                        maha = mahalanobis[:T]
                        idx = index[:T]

                        # Reconstruction error panel: mean across all N×F dimensions, scaled to [0,1]
                        _rm = recon.mean(axis=(1, 2))
                        _rm = (_rm - _rm.min()) / max(_rm.max() - _rm.min(), 1e-8)
                        all_panels.append({
                            'type': 'recon',
                            'model': model,
                            'series': _rm,
                            'index': idx,
                        })

                        for strategy in ['likelihood', 'mahalanobis']:
                            if fix_scoring_parameters is not None:
                                if strategy not in fix_scoring_parameters:
                                    continue
                                params = fix_scoring_parameters[strategy]
                                anomaly_threshold = float(params['anomaly_threshold'])
                                topk = int(params['topk'])
                                long_window = params.get('long_window', None)
                                short_window = params.get('short_window', None)
                            else:
                                strat_df = grid_df[grid_df['post_processing_strategy'] == strategy]
                                if strat_df.empty:
                                    continue
                                best_row = strat_df.loc[strat_df['NAB_reward_fn_rank'].idxmin()]
                                anomaly_threshold = float(best_row['anomaly_threshold'])
                                topk = int(best_row['topk'])
                                lw = best_row.get('long_window', None)
                                sw = best_row.get('short_window', None)
                                long_window = None if pd.isna(lw) else int(lw)
                                short_window = None if pd.isna(sw) else int(sw)

                            is_anom, likelihoods, _, _ = label_reconstruction_errors(
                                idx, recon, maha, strategy, topk, anomaly_threshold,
                                long_window, short_window,
                            )

                            all_panels.append({
                                'type': 'score',
                                'model': model,
                                'strategy': strategy,
                                'scores': likelihoods,
                                'anomaly_threshold': anomaly_threshold,
                                'long_window': long_window,
                                'short_window': short_window,
                                'is_anomalies': is_anom.values,
                                'index': idx,
                            })

                if not all_panels:
                    print(f'No results to plot for {http_code}/{agg}/{fill_nan}')
                    continue

                # For each A3TGCN score panel, find timesteps not detected by any
                # other model using the same scoring strategy.
                for panel in all_panels:
                    if panel['type'] != 'score' or panel['model'] != 'A3TGCN':
                        continue
                    strategy = panel['strategy']
                    n_t = len(panel['is_anomalies'])
                    others_union = np.zeros(n_t, dtype=bool)
                    for other in all_panels:
                        if (other['type'] == 'score'
                                and other['model'] != 'A3TGCN_null_padding_feature'
                                and other['strategy'] == strategy):
                            o_len = min(len(other['is_anomalies']), n_t)
                            others_union[:o_len] |= other['is_anomalies'][:o_len].astype(bool)
                    panel['unique_mask'] = panel['is_anomalies'].astype(bool) & ~others_union

                n_panels = len(all_panels)
                fig, axes = plt.subplots(
                    n_panels + 1, 1,
                    figsize=(10, 2 * (n_panels + 1)),
                    sharex=True,
                    constrained_layout=True,
                )
                if n_panels + 1 == 1:
                    axes = [axes]

                # Row 0: averaged raw time series with source-colored anomaly spans
                ax0 = axes[0]
                T0 = min(len(mean_series), len(index))
                ax0.plot(index[:T0], mean_series[:T0], color='steelblue', linewidth=0.8)
                _plot_source_spans(ax0, anomaly_windows_test, alpha=0.3)
                ax0.set_ylabel('Mean Value [0,1]', fontsize=FONT_SIZE)
                ax0.set_title('Averaged Time Series', fontsize=FONT_SIZE)
                ax0.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                ax0.tick_params(axis='y', labelsize=FONT_SIZE - 1)

                # Remaining rows: recon error or score panels
                for pi, panel in enumerate(all_panels):
                    ax = axes[pi + 1]
                    idx = panel['index']

                    # Ground-truth anomaly spans on every subplot, colored by source
                    _plot_source_spans(ax, anomaly_windows_test, alpha=0.2)

                    if panel['type'] == 'recon':
                        ax.plot(idx, panel['series'], color='steelblue', linewidth=0.8)
                        ax.set_ylabel('Recon Error [0,1]', fontsize=FONT_SIZE)
                        ax.set_title(f'{panel["model"]} — Reconstruction Error (mean)',
                                     fontsize=FONT_SIZE)

                    else:
                        scores = panel['scores']
                        is_anom = panel['is_anomalies']

                        ax.plot(idx, scores, color='steelblue', linewidth=0.8)

                        gt_mask = labels[:len(idx)].astype(bool)
                        anom_mask = is_anom.astype(bool)
                        tp_mask = anom_mask & gt_mask
                        fp_mask = anom_mask & ~gt_mask
                        if tp_mask.any():
                            ax.scatter(idx[tp_mask], scores[tp_mask],
                                       color='red', s=8, zorder=3)
                        if fp_mask.any():
                            ax.scatter(idx[fp_mask], scores[fp_mask],
                                       color='darkorange', s=8, zorder=3)
                        unique_mask = panel.get('unique_mask')
                        if unique_mask is not None:
                            unique_mask = unique_mask & gt_mask  # true positives only
                        if unique_mask is not None and unique_mask.any():
                            ax.scatter(idx[unique_mask], scores[unique_mask],
                                       color='green', s=20, zorder=4, marker='*')

                        ax.set_ylabel('Score', fontsize=FONT_SIZE)
                        unique_count = int(unique_mask.sum()) if unique_mask is not None else 0
                        detection_str = (f'true={tp_mask.sum()}, false={fp_mask.sum()}'
                                         + (f', unique_true={unique_count}' if unique_mask is not None else ''))
                        if panel['strategy'] == 'likelihood':
                            param_str = (f'threshold={panel["anomaly_threshold"]:.5f}, '
                                         f'long_win={panel["long_window"]}, '
                                         f'short_win={panel["short_window"]}')
                            ax.set_title(f'{panel["model"]} — {panel["strategy"]}  '
                                         f'({param_str})  [{detection_str}]',
                                         fontsize=FONT_SIZE)
                        else:
                            ax.set_title(f'{panel["model"]} — {panel["strategy"]}  '
                                         f'(threshold_pct={panel["anomaly_threshold"]:.1f})'
                                         f'  [{detection_str}]',
                                         fontsize=FONT_SIZE)

                    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                    ax.tick_params(axis='y', labelsize=FONT_SIZE - 1)

                # Single shared legend to the right of all subplots
                legend_handles = [
                    plt.Line2D([0], [0], color='steelblue', linewidth=1.2, label='Score'),
                    plt.Line2D([0], [0], color='red', marker='o', markersize=4,
                               linestyle='None', label='True detected'),
                    plt.Line2D([0], [0], color='darkorange', marker='o', markersize=4,
                               linestyle='None', label='False detected'),
                    plt.Line2D([0], [0], color='green', marker='*', markersize=7,
                               linestyle='None', label='True unique to A3TGCN_null_padding_feature'),
                    *_source_legend_handles(),
                ]
                fig.legend(handles=legend_handles, loc='center left',
                           bbox_to_anchor=(1.01, 0.5), fontsize=FONT_SIZE - 1,
                           frameon=True)

                axes[-1].set_xlabel('Time', fontsize=FONT_SIZE)
                axes[-1].tick_params(axis='x', rotation=30, labelsize=FONT_SIZE - 1)

                fig.suptitle(
                    f'Detected Anomalies — {http_code} / {agg} / imputation={fill_nan}',
                    fontsize=FONT_SIZE + 2, fontweight='bold',
                )

                save_dir = out_dir or '.'
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(
                    save_dir,
                    f'detected_anomalies_{http_code}_{agg}_{fill_nan}.png',
                )
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved detected anomalies plot to {out_path}')

                # ── Zoom-in figures: one per (strategy × span), per-model subplots ─
                model_list = list(dict.fromkeys(
                    p['model'] for p in all_panels if p['type'] == 'score'))
                all_recon_panels = [p for p in all_panels if p['type'] == 'recon']

                for a3_panel in all_panels:
                    if a3_panel['type'] != 'score' or a3_panel['model'] != 'A3TGCN_null_padding_feature':
                        continue
                    strategy = a3_panel['strategy']
                    a3_idx = np.array(a3_panel['index'])
                    unique_mask_raw = a3_panel.get('unique_mask')
                    if unique_mask_raw is None:
                        continue
                    gt_mask_z = labels[:len(a3_idx)].astype(bool)
                    unique_true = unique_mask_raw & gt_mask_z
                    if not unique_true.any():
                        continue

                    strat_score_panels = [p for p in all_panels
                                          if p['type'] == 'score' and p['strategy'] == strategy]

                    for span_i, (t_start, t_end) in enumerate(
                            _anomaly_spans(unique_true.astype(float), a3_idx)):
                        pad = pd.Timedelta(hours=6)
                        zoom_start = pd.Timestamp(t_start) - pad
                        zoom_end = pd.Timestamp(t_end) + pad

                        def _zmask(arr_idx, _zs=zoom_start, _ze=zoom_end):
                            ai = np.array(arr_idx, dtype='datetime64[ns]')
                            return ((ai >= np.datetime64(_zs, 'ns')) &
                                    (ai <= np.datetime64(_ze, 'ns')))

                        # Build per-model row list: (model, kind, panel)
                        model_row_specs = []
                        for m in model_list:
                            rp = next((p for p in all_recon_panels if p['model'] == m), None)
                            sp = next((p for p in strat_score_panels if p['model'] == m), None)
                            if rp:
                                model_row_specs.append((m, 'recon', rp))
                            if sp:
                                model_row_specs.append((m, 'score', sp))

                        n_zoom_rows = 1 + len(model_row_specs)
                        fig_z, axes_z = plt.subplots(
                            n_zoom_rows, 1,
                            figsize=(14, 3 * n_zoom_rows),
                            sharex=True,
                            constrained_layout=True,
                        )
                        if n_zoom_rows == 1:
                            axes_z = [axes_z]

                        # Row 0: averaged time series
                        T0 = min(len(mean_series), len(index))
                        ts_idx = np.array(index[:T0])
                        zm0 = _zmask(ts_idx)
                        ax_z0 = axes_z[0]
                        ax_z0.plot(ts_idx[zm0], mean_series[:T0][zm0],
                                   color='steelblue', linewidth=0.8)
                        _plot_source_spans(ax_z0, anomaly_windows_test, alpha=0.3)
                        ax_z0.set_xlim(zoom_start, zoom_end)
                        ax_z0.set_ylabel('Mean Value [0,1]', fontsize=FONT_SIZE)
                        ax_z0.set_title('Averaged Time Series', fontsize=FONT_SIZE)
                        ax_z0.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                        ax_z0.tick_params(axis='y', labelsize=FONT_SIZE - 1)

                        # One subplot per (model, kind)
                        for row_z, (m, kind, panel) in enumerate(model_row_specs, start=1):
                            ax_z = axes_z[row_z]
                            p_idx = np.array(panel['index'])
                            zm = _zmask(p_idx)
                            _plot_source_spans(ax_z, anomaly_windows_test, alpha=0.2)
                            ax_z.set_xlim(zoom_start, zoom_end)

                            if kind == 'recon':
                                ax_z.plot(p_idx[zm], panel['series'][zm],
                                          color='steelblue', linewidth=0.9)
                                ax_z.set_ylabel('Recon Error [0,1]', fontsize=FONT_SIZE)
                                ax_z.set_title(f'{m} — Reconstruction Error',
                                               fontsize=FONT_SIZE)

                            else:  # score
                                sc = panel['scores']
                                det = panel['is_anomalies'].astype(bool)
                                gt = labels[:len(p_idx)].astype(bool)
                                ax_z.plot(p_idx[zm], sc[zm],
                                          color='steelblue', linewidth=0.9)
                                tp_z = det & gt
                                fp_z = det & ~gt
                                if (tp_z & zm).any():
                                    ax_z.scatter(p_idx[tp_z & zm], sc[tp_z & zm],
                                                 color='red', s=8, zorder=3)
                                if (fp_z & zm).any():
                                    ax_z.scatter(p_idx[fp_z & zm], sc[fp_z & zm],
                                                 color='darkorange', s=8, zorder=3)
                                if m == 'A3TGCN_null_padding_feature':
                                    u_raw = panel.get('unique_mask')
                                    if u_raw is not None:
                                        u_true_z = u_raw & gt
                                        if (u_true_z & zm).any():
                                            ax_z.scatter(p_idx[u_true_z & zm],
                                                         sc[u_true_z & zm],
                                                         color='green', s=20,
                                                         zorder=4, marker='*')
                                ax_z.set_ylabel('Score', fontsize=FONT_SIZE)
                                ax_z.set_title(f'{m} — {strategy}', fontsize=FONT_SIZE)

                            ax_z.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                            ax_z.tick_params(axis='y', labelsize=FONT_SIZE - 1)

                        axes_z[-1].set_xlabel('Time', fontsize=FONT_SIZE)
                        axes_z[-1].tick_params(axis='x', rotation=30,
                                               labelsize=FONT_SIZE - 1)
                        fig_z.suptitle(
                            f'Zoom-in — {strategy} — '
                            f'{http_code} / {agg} / imputation={fill_nan}\n'
                            f'{t_start} → {t_end}',
                            fontsize=FONT_SIZE, fontweight='bold',
                        )

                        zoom_legend = [
                            plt.Line2D([0], [0], color='steelblue', linewidth=1.2,
                                       label='Score / Recon Error'),
                            plt.Line2D([0], [0], color='red', marker='o',
                                       markersize=4, linestyle='None',
                                       label='True detected'),
                            plt.Line2D([0], [0], color='darkorange', marker='o',
                                       markersize=4, linestyle='None',
                                       label='False detected'),
                            plt.Line2D([0], [0], color='green', marker='*',
                                       markersize=7, linestyle='None',
                                       label='True unique to A3TGCN_null_padding_feature'),
                            *_source_legend_handles(),
                        ]
                        fig_z.legend(handles=zoom_legend, loc='center left',
                                     bbox_to_anchor=(1.01, 0.5),
                                     fontsize=FONT_SIZE - 1, frameon=True)

                        zoom_path = os.path.join(
                            save_dir,
                            f'detected_anomalies_{http_code}_{agg}_{fill_nan}'
                            f'_{strategy}_zoom_{span_i}.png',
                        )
                        fig_z.savefig(zoom_path, dpi=150, bbox_inches='tight')
                        plt.close(fig_z)
                        print(f'Saved zoom-in plot to {zoom_path}')

                        # ── Combined zoom: all models overlaid in one subplot ──
                        model_colors = {
                            m: plt.colormaps['tab10'](i / max(len(model_list), 1))
                            for i, m in enumerate(model_list)
                        }
                        n_comb_rows = 1 + (1 if all_recon_panels else 0) + 1
                        fig_c, axes_c = plt.subplots(
                            n_comb_rows, 1,
                            figsize=(14, 3 * n_comb_rows),
                            sharex=True,
                            constrained_layout=True,
                        )
                        if n_comb_rows == 1:
                            axes_c = [axes_c]

                        # Row 0: averaged time series
                        ax_c0 = axes_c[0]
                        ax_c0.plot(ts_idx[zm0], mean_series[:T0][zm0],
                                   color='steelblue', linewidth=0.8)
                        _plot_source_spans(ax_c0, anomaly_windows_test, alpha=0.3)
                        ax_c0.set_xlim(zoom_start, zoom_end)
                        ax_c0.set_ylabel('Mean Value [0,1]', fontsize=FONT_SIZE)
                        ax_c0.set_title('Averaged Time Series', fontsize=FONT_SIZE)
                        ax_c0.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                        ax_c0.tick_params(axis='y', labelsize=FONT_SIZE - 1)

                        row_c = 1
                        # Row 1: all models' recon overlaid
                        if all_recon_panels:
                            ax_cr = axes_c[row_c]
                            for rp in all_recon_panels:
                                rp_idx = np.array(rp['index'])
                                zm = _zmask(rp_idx)
                                ax_cr.plot(rp_idx[zm], rp['series'][zm],
                                           color=model_colors[rp['model']],
                                           linewidth=0.9, label=rp['model'])
                            _plot_source_spans(ax_cr, anomaly_windows_test, alpha=0.2)
                            ax_cr.set_xlim(zoom_start, zoom_end)
                            ax_cr.set_ylabel('Recon Error [0,1]', fontsize=FONT_SIZE)
                            ax_cr.set_title('Reconstruction Error',
                                            fontsize=FONT_SIZE)
                            ax_cr.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                            ax_cr.tick_params(axis='y', labelsize=FONT_SIZE - 1)
                            row_c += 1

                        # Last row: all models' scores overlaid
                        ax_cs = axes_c[row_c]
                        for sp in strat_score_panels:
                            sp_idx = np.array(sp['index'])
                            zm = _zmask(sp_idx)
                            sc = sp['scores']
                            gt = labels[:len(sp_idx)].astype(bool)
                            det = sp['is_anomalies'].astype(bool)
                            col = model_colors[sp['model']]
                            ax_cs.plot(sp_idx[zm], sc[zm], color=col,
                                       linewidth=0.9, label=sp['model'])
                            tp_z = det & gt
                            fp_z = det & ~gt
                            if (tp_z & zm).any():
                                ax_cs.scatter(sp_idx[tp_z & zm], sc[tp_z & zm],
                                              color='red', s=8, zorder=3)
                            if (fp_z & zm).any():
                                ax_cs.scatter(sp_idx[fp_z & zm], sc[fp_z & zm],
                                              color='darkorange', s=8, zorder=3)
                            if sp['model'] == 'A3TGCN_null_padding_feature':
                                u_raw = sp.get('unique_mask')
                                if u_raw is not None:
                                    u_true_z = u_raw & gt
                                    if (u_true_z & zm).any():
                                        ax_cs.scatter(sp_idx[u_true_z & zm],
                                                      sc[u_true_z & zm],
                                                      color='green', s=20,
                                                      zorder=4, marker='*')
                        _plot_source_spans(ax_cs, anomaly_windows_test, alpha=0.2)
                        ax_cs.set_xlim(zoom_start, zoom_end)
                        ax_cs.set_ylabel('Score', fontsize=FONT_SIZE)
                        ax_cs.set_title(f'Scores — {strategy}',
                                        fontsize=FONT_SIZE)
                        ax_cs.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                        ax_cs.tick_params(axis='y', labelsize=FONT_SIZE - 1)

                        axes_c[-1].set_xlabel('Time', fontsize=FONT_SIZE)
                        axes_c[-1].tick_params(axis='x', rotation=30,
                                               labelsize=FONT_SIZE - 1)
                        fig_c.suptitle(
                            f'Zoom-in (combined) — {strategy} — '
                            f'{http_code} / {agg} / imputation={fill_nan}\n'
                            f'{t_start} → {t_end}',
                            fontsize=FONT_SIZE, fontweight='bold',
                        )
                        comb_legend = (
                            [plt.Line2D([0], [0], color=model_colors[m],
                                        linewidth=1.2, label=m) for m in model_list]
                            + [
                                plt.Line2D([0], [0], color='red', marker='o',
                                           markersize=4, linestyle='None',
                                           label='True detected'),
                                plt.Line2D([0], [0], color='darkorange', marker='o',
                                           markersize=4, linestyle='None',
                                           label='False detected'),
                                plt.Line2D([0], [0], color='green', marker='*',
                                           markersize=7, linestyle='None',
                                           label='True unique to A3TGCN_null_padding_feature'),
                                *_source_legend_handles(),
                            ]
                        )
                        fig_c.legend(handles=comb_legend, loc='center left',
                                     bbox_to_anchor=(1.01, 0.5),
                                     fontsize=FONT_SIZE - 1, frameon=True)
                        comb_path = os.path.join(
                            save_dir,
                            f'detected_anomalies_{http_code}_{agg}_{fill_nan}'
                            f'_{strategy}_zoom_{span_i}_combined.png',
                        )
                        fig_c.savefig(comb_path, dpi=150, bbox_inches='tight')
                        plt.close(fig_c)
                        print(f'Saved combined zoom-in plot to {comb_path}')


def plot_detected_anomalies_for_specific_periods(
        dataloader, results_dir, models, imputation_strategies,
        http_codes, aggregations,
        null_padding_features,
        null_padding_targets,
        zoom_in_periods,
        out_dir=None,
        fix_scoring_parameters=None):

    slide_win = dataloader.data_preparation_config.slide_win
    index = dataloader.test_index
    labels = np.array(dataloader.test_labels).ravel()
    anomaly_windows_test = dataloader.anomaly_windows_test

    X_raw = dataloader.scaler.inverse_transform(dataloader.X_test_scaled)
    mean_series = X_raw.mean(axis=1)
    _ms_min, _ms_max = mean_series.min(), mean_series.max()
    mean_series = (mean_series - _ms_min) / max(_ms_max - _ms_min, 1e-8)

    _zoom_list = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in zoom_in_periods]
    n_periods = len(_zoom_list)
    if n_periods == 0:
        return

    for http_code in http_codes:
        for agg in aggregations:
            for fill_nan in imputation_strategies:
                for null_padding_feature, null_padding_target in itertools.product(
                        null_padding_features, null_padding_targets):
                    all_panels = []

                    for model in models:
                        model_folder = model
                        if model == 'A3TGCN':
                            if null_padding_feature and null_padding_target:
                                model_folder = f'{model}_null_padding_both'
                            elif null_padding_feature:
                                model_folder = f'{model}_null_padding_feature'
                            elif null_padding_target:
                                model_folder = f'{model}_null_padding_target'

                        model_dir = os.path.join(
                            results_dir,
                            f'window_{slide_win}',
                            f'no_group_{http_code}_{agg}',
                            f'fill_nan_with_{fill_nan}',
                            model_folder,
                        )
                        recon_path = os.path.join(model_dir, 'reconstruction_errors.npy')
                        maha_path = os.path.join(model_dir, 'mahalanobis.npy')
                        required_exists = os.path.exists(recon_path)
                        if fix_scoring_parameters is None:
                            csv_path = os.path.join(model_dir, f'{model}_grid_search.csv')
                            required_exists = required_exists and os.path.exists(csv_path)
                        if not required_exists:
                            print(f'Skipping {model}: missing files in {model_dir}')
                            continue

                        reconstruction_errors = np.load(recon_path)
                        mahalanobis_dist = (np.load(maha_path) if os.path.exists(maha_path)
                                            else np.zeros(len(reconstruction_errors)))
                        if fix_scoring_parameters is None:
                            grid_df = pd.read_csv(csv_path)

                        T = min(len(reconstruction_errors), len(index))
                        recon = reconstruction_errors[:T]
                        maha = mahalanobis_dist[:T]
                        idx = index[:T]

                        _rm = recon.mean(axis=(1, 2))
                        _rm = (_rm - _rm.min()) / max(_rm.max() - _rm.min(), 1e-8)
                        all_panels.append({'type': 'recon', 'model': model_folder,
                                           'series': _rm, 'index': idx})

                        for strategy in ['likelihood', 'mahalanobis']:
                            if fix_scoring_parameters is not None:
                                if strategy not in fix_scoring_parameters:
                                    continue
                                params = fix_scoring_parameters[strategy]
                                anomaly_threshold = float(params['anomaly_threshold'])
                                topk = int(params['topk'])
                                long_window = params.get('long_window', None)
                                short_window = params.get('short_window', None)
                            else:
                                strat_df = grid_df[grid_df['post_processing_strategy'] == strategy]
                                if strat_df.empty:
                                    continue
                                best_row = strat_df.loc[strat_df['NAB_reward_fn_rank'].idxmin()]
                                anomaly_threshold = float(best_row['anomaly_threshold'])
                                topk = int(best_row['topk'])
                                lw = best_row.get('long_window', None)
                                sw = best_row.get('short_window', None)
                                long_window = None if pd.isna(lw) else int(lw)
                                short_window = None if pd.isna(sw) else int(sw)

                            is_anom, likelihoods, _, _ = label_reconstruction_errors(
                                idx, recon, maha, strategy, topk,
                                anomaly_threshold, long_window, short_window,
                            )
                            all_panels.append({
                                'type': 'score', 'model': model_folder, 'strategy': strategy,
                                'scores': likelihoods, 'is_anomalies': is_anom.values,
                                'index': idx,
                            })

                    if not all_panels:
                        print(f'No results for {http_code}/{agg}/{fill_nan}')
                        continue

                    # A3TGCN unique mask per strategy
                    for panel in all_panels:
                        if panel['type'] != 'score' or panel['model'] != 'A3TGCN_null_padding_feature':
                            continue
                        strategy = panel['strategy']
                        n_t = len(panel['is_anomalies'])
                        others_union = np.zeros(n_t, dtype=bool)
                        for other in all_panels:
                            if (other['type'] == 'score' and other['model'] != 'A3TGCN_null_padding_feature'
                                    and other['strategy'] == strategy):
                                o_len = min(len(other['is_anomalies']), n_t)
                                others_union[:o_len] |= other['is_anomalies'][:o_len].astype(bool)
                        panel['unique_mask'] = panel['is_anomalies'].astype(bool) & ~others_union

                    # Build ordered row specs: time_series row is index 0;
                    # then per model: recon + per-strategy scores
                    model_list = list(dict.fromkeys(
                        p['model'] for p in all_panels if p['type'] == 'score'))
                    strategies = list(dict.fromkeys(
                        p['strategy'] for p in all_panels if p['type'] == 'score'))
                    row_specs = []
                    for m in model_list:
                        rp = next((p for p in all_panels
                                   if p['type'] == 'recon' and p['model'] == m), None)
                        if rp:
                            row_specs.append(('recon', m, rp))
                        for strat in strategies:
                            sp = next((p for p in all_panels
                                       if p['type'] == 'score' and p['model'] == m
                                       and p['strategy'] == strat), None)
                            if sp:
                                row_specs.append(('score', m, strat, sp))

                    # Fixed rows: time series | recon (all models) | per-strategy (all models)
                    model_colors = {
                        m: plt.colormaps['tab10'](i / max(len(model_list), 1))
                        for i, m in enumerate(model_list)
                    }
                    recon_panels = [p for p in all_panels if p['type'] == 'recon']
                    # Row labels: 0=ts, 1=recon, 2..=strategies
                    row_labels = (['Recon. Error [0,1]'] +
                                  [s.capitalize() for s in strategies])
                    n_rows = 1 + len(row_labels)
                    T0 = min(len(mean_series), len(index))
                    ts_idx = np.array(index[:T0])

                    fig, axes = plt.subplots(
                        n_rows, n_periods,
                        figsize=(4 * n_periods, 2 * n_rows),
                        sharex='col',
                        constrained_layout=True,
                    )
                    if n_rows == 1:
                        axes = axes[np.newaxis, :]
                    if n_periods == 1:
                        axes = axes[:, np.newaxis]

                    for col, (zoom_start, zoom_end) in enumerate(_zoom_list):
                        def _zmask(arr_idx, _zs=zoom_start, _ze=zoom_end):
                            ai = np.array(arr_idx, dtype='datetime64[ns]')
                            return ((ai >= np.datetime64(pd.Timestamp(_zs), 'ns')) &
                                    (ai <= np.datetime64(pd.Timestamp(_ze), 'ns')))

                        overlapping_ids = [
                            str(aw_idx-6)
                            for aw_idx, win in anomaly_windows_test.iterrows()
                            if (pd.Timestamp(win['anomaly_window_start']) <= zoom_end
                                and pd.Timestamp(win['anomaly_window_end']) >= zoom_start)
                        ]
                        period_label = (f'Anomaly {", ".join(overlapping_ids)}'
                                        if overlapping_ids else f'Period {col}')

                        # Choose formatter based on zoom span width
                        _span_hours = (zoom_end - zoom_start).total_seconds() / 3600
                        _fmt = (mdates.DateFormatter('%d %H:%M') if _span_hours <= 72
                                else mdates.DateFormatter('%m-%d'))

                        def _style(ax, ylabel=None, title=None,
                                   _f=_fmt, color='black'):
                            _plot_source_spans(ax, anomaly_windows_test, alpha=0.2)
                            ax.set_xlim(zoom_start, zoom_end)
                            ax.xaxis.set_major_formatter(_f)
                            ax.ticklabel_format(axis='y', style='sci',
                                                scilimits=(-2, 4), useMathText=False)
                            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                            ax.tick_params(axis='x', rotation=30, labelsize=FONT_SIZE - 2)
                            ax.tick_params(axis='y', labelsize=FONT_SIZE - 1)
                            if col == 0 and ylabel:
                                ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
                            if title:
                                ax.set_title(title, fontsize=FONT_SIZE - 1, color=color)

                        def _lbl(row, _c=col):
                            return f'({chr(ord("a") + row * n_periods + _c)}) '

                        # Row 0: averaged time series
                        ax0 = axes[0, col]
                        zm0 = _zmask(ts_idx)
                        ax0.plot(ts_idx[zm0], mean_series[:T0][zm0],
                                 color='steelblue', linewidth=0.8)
                        _style(ax0, ylabel=f'Norm. $\\mathtt{{{agg}}}$ [0,1]', title=_lbl(0) + period_label, color='red')

                        # Row 1: all models' recon overlaid
                        ax_r = axes[1, col]
                        for rp in recon_panels:
                            p_idx = np.array(rp['index'])
                            zm = _zmask(p_idx)
                            ax_r.plot(p_idx[zm], rp['series'][zm],
                                      color=model_colors[rp['model']],
                                      linewidth=0.8, label=rp['model'])
                        _style(ax_r, ylabel='Recon. Error [0,1]',
                               title=_lbl(1) + 'Avg. Reconstruction Errors')

                        # Rows 2+: one per strategy, all models overlaid
                        for strat_i, strat in enumerate(strategies):
                            ax_s = axes[2 + strat_i, col]
                            strat_panels = [p for p in all_panels
                                            if p['type'] == 'score' and p['strategy'] == strat]
                            for sp in strat_panels:
                                m = sp['model']
                                p_idx = np.array(sp['index'])
                                zm = _zmask(p_idx)
                                sc = sp['scores']
                                gt = labels[:len(p_idx)].astype(bool)
                                det = sp['is_anomalies'].astype(bool)
                                col_m = model_colors[m]
                                ax_s.plot(p_idx[zm], sc[zm], color=col_m,
                                          linewidth=0.8, label=m)
                                tp_z = det & gt
                                fp_z = det & ~gt
                                if (tp_z & zm).any():
                                    ax_s.scatter(p_idx[tp_z & zm], sc[tp_z & zm],
                                                 color='red', s=25, zorder=3)
                                if (fp_z & zm).any():
                                    ax_s.scatter(p_idx[fp_z & zm], sc[fp_z & zm],
                                                 color='darkorange', s=25, zorder=3)
                                if m == 'A3TGCN_null_padding_feature':
                                    u_raw = sp.get('unique_mask')
                                    if u_raw is not None:
                                        u_true_z = u_raw & gt
                                        if (u_true_z & zm).any():
                                            ax_s.scatter(p_idx[u_true_z & zm],
                                                         sc[u_true_z & zm],
                                                         color='green', s=50,
                                                         zorder=4, marker='*')
                            _style(ax_s, ylabel='Score [0,1]',
                                   title=_lbl(2 + strat_i) + f'{SCORING_STRATEGY_DISPLAY_FULL_NAME_MAP.get(strat,strat)}')

                    legend_handles = (
                        [plt.Line2D([0], [0], color=model_colors[m], linewidth=1.2, label=MODEL_DISPLAY_NAME_MAP.get(m,m))
                         for m in model_list]
                        + [
                            plt.Line2D([0], [0], color='red', marker='o', markersize=7,
                                       linestyle='None', label='True Positive'),
                            plt.Line2D([0], [0], color='darkorange', marker='o', markersize=7,
                                       linestyle='None', label='False Positive'),
                            plt.Line2D([0], [0], color='green', marker='*', markersize=10,
                                       linestyle='None', label='True Positive only captured by ClouDens'),
                        ]
                    )
                    fig.legend(handles=legend_handles, loc='lower center',
                               bbox_to_anchor=(0.5, -0.05),
                               ncol=len(legend_handles),
                               fontsize=FONT_SIZE - 1, frameon=True)
                    fig.suptitle(
                        f'Detected Anomalies — ${text_subset_wrapper(http_code,agg)}$ — {fill_nan} imputation',
                        fontsize=FONT_SIZE + 1, fontweight='bold',
                    )

                    save_dir = out_dir or '.'
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(
                        save_dir,
                        f'detected_anomalies_periods_{http_code}_{agg}_{fill_nan}.png',
                    )
                    fig.savefig(out_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Saved periods plot to {out_path}')


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):

    results_dir = cfg.evaluation.model_save_path
    shown_model = 'GRU'
    shown_http_code = '5xx'
    shown_aggregation = 'count'
    shown_fill_nan = 'zero'
    shown_slide_win = 6

    shown_nodes = ['datacenter7_CLIENT_component15_GET_500_endpoint805',
                   'datacenter2_CLIENT_component47_DELETE_502_endpoint915',
                   'datacenter4_CLIENT_component15_GET_500_endpoint643',
                   'datacenter3_CLIENT_component54_GET_504_endpoint447'
                   ]

    # shown_http_code = '5xx'
    # shown_aggregation = 'max'
    # shown_fill_nan = 'median'
    # shown_nodes = ['datacenter6_CLIENT_component34_GET_500_endpoint884',
    #                'datacenter4_SERVER_component26_POST_500_endpoint213',
    #                'datacenter3_SERVER_component26_PUT_500_endpoint172']

    # shown_http_code = '5xx'
    # shown_aggregation = 'avg'
    # shown_fill_nan = 'median'

    data_preparation_config = cfg.data_preparation_pipeline

    if shown_model in ('GRU', 'TranAD', 'AnomalyTransformer'):
        data_preparation_config.null_padding_feature = False
        data_preparation_config.null_padding_target = False

    data_preparation_config.slide_win = shown_slide_win
    data_preparation_config.fill_nan = shown_fill_nan
    data_preparation_config.feature_subsets.http_codes = [shown_http_code]
    data_preparation_config.feature_subsets.aggregations = [shown_aggregation]

    data_preparation_config.features_prep.filter.http_codes = [shown_http_code]
    data_preparation_config.features_prep.filter.aggregations = [shown_aggregation]

    dataloader = IBMDatasetLoader(data_preparation_config)

    train_loader, valid_loader, test_loader, edge_index = dataloader.get_index_dataset(
        window_size=data_preparation_config.slide_win,
        null_padding_feature=data_preparation_config.null_padding_feature,
        null_padding_target=data_preparation_config.null_padding_target,
        batch_size=16,
        device='cpu')
    print("Training dataset batches", len(train_loader))
    print("Validation dataset batches", len(valid_loader))
    print("Testing dataset batches", len(test_loader))

    results_dir = os.path.join(get_project_root(), results_dir)

    # plot_reconstruction_errors(dataloader, results_dir, shown_model, shown_http_code, shown_aggregation, shown_fill_nan, shown_slide_win, shown_nodes)

    shown_node = 'datacenter4_CLIENT_component15_GET_500_endpoint643'
    output_dir = os.path.join(get_project_root(), 'figures')
    start_date = '2024-03-07 00:00:00'
    end_date = '2024-03-08 00:00:00'

    start_date = '2024-04-14 06:00:00'
    end_date = '2024-04-14 18:00:00'

    # plot_only_node_series(dataloader, shown_node, output_dir, (start_date, end_date))

    shown_nodes = ['datacenter2_CLIENT_component15_GET_500_endpoint643',
                   'datacenter4_CLIENT_component15_GET_500_endpoint643',
                   'datacenter4_SERVER_component26_PUT_500_endpoint172'
                   ]

    # plot_only_multiple_node_series_zoom_in(dataloader, shown_nodes, output_dir, (start_date, end_date))

    shown_models = ['GRU','A3TGCN']
    missing_imputation_stategies = [shown_fill_nan]
    http_codes = [shown_http_code]
    aggregations = [shown_aggregation]
    null_padding_features = [True]
    null_padding_targets = [False]

    fix_scoring_parameters = {
        'likelihood': {
            'long_window':30,
            'short_window':2,
            'anomaly_threshold': 0.99975,
            'topk': 1,
        },
        'mahalanobis': {
            'long_window': 30,
            'short_window': 2,
            'anomaly_threshold': 99.8,
            'topk': 1,
        }

    }
    # plot_detected_anomalies(dataloader, results_dir, shown_models, missing_imputation_stategies, http_codes, aggregations,
    #                         null_padding_features,
    #                         null_padding_targets,
    #                         output_dir,
    #                         fix_scoring_parameters=fix_scoring_parameters)

    zoom_in_periods = [
        (pd.Timestamp('2024-03-07 00:00:00'), pd.Timestamp('2024-03-07 12:00:00')),
        # (pd.Timestamp('2024-03-13 00:00:00'), pd.Timestamp('2024-03-13 23:00:00')),
        (pd.Timestamp('2024-03-19 06:00:00'), pd.Timestamp('2024-03-19 16:00:00')),
        (pd.Timestamp('2024-05-31 10:00:00'), pd.Timestamp('2024-05-31 21:00:00')),
    ]


    plot_only_time_series(dataloader, output_dir, zoom_periods=zoom_in_periods)
    plot_detected_anomalies_for_specific_periods(dataloader, results_dir,
                                                 shown_models,
                                                 missing_imputation_stategies,
                                                 http_codes, aggregations,
                                                null_padding_features,
                                                null_padding_targets,
                                                zoom_in_periods,
                                                output_dir,
                                                fix_scoring_parameters=fix_scoring_parameters)


if __name__ == '__main__':
    main()