"""
Streamlit explorer of grid-search results, two pages showing the same views:
  • Results folder: the grid-search CSVs scanned under a results folder (default: trained_models_log1p),
    which can be saved as one merged file;
  • Merged file: a merged file (from merged_results/ or uploaded), e.g. merged on the HPC server with
        python streamlit_result_visualization/grid_results.py trained_models_log1p
    and copied to a PC: no results folder needed.
See grid_results.py for the folder layout and the merged files.

Run from the project root:
    streamlit run streamlit_result_visualization/results_explorer_app.py
The app reruns when this file is saved (.streamlit/config.toml next to it: runOnSave) and, with
'Auto-refresh' on, when grid-search files change on disk.
"""
import ast
import hashlib
import inspect
import io
import itertools
import os
import re
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from grid_results import (PROJECT_ROOT, TAG_COLUMNS, MERGED_DIR, MERGED_SUFFIXES, IGNORED_MODEL_FAMILIES,
                          IGNORED_MODEL_FOLDERS, find_grid_files, grid_signature, is_ignored_model,
                          read_grid_files, read_merged, write_merged, merged_bytes, find_merged_files,
                          default_merged_path)

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from figure_style import TICK_FONT_SIZE, AXIS_LABEL_FONT_SIZE, LEGEND_FONT_SIZE  # noqa: E402

NAB_COLUMNS = {'reward_fn': 'reward_fn_normalized', 'standard': 'standard_normalized'}
# Test measures the charts can show: label -> (column, axis title, number format)
MEASURES = {
    'NAB reward_fn': ('reward_fn_normalized', 'NAB reward_fn (normalized)', '.2f'),
    'NAB standard': ('standard_normalized', 'NAB standard (normalized)', '.2f'),
    'AUC-PR': ('test_auc_pr', 'Test AUC-PR', '.4f'),
    'VUS-PR': ('test_vus_pr', 'Test VUS-PR', '.4f'),
    'F1': ('f1', 'Test F1', '.4f'),
    'Precision': ('precision', 'Test precision', '.4f'),
    'Recall': ('recall', 'Test recall', '.4f'),
}
# Metrics a setting can be selected by: NAB on test, or metrics on train / valid (no test labels)
SELECTION_METRICS = [
    'reward_fn_normalized', 'standard_normalized', 'f1',
    'train_valid_reward_fn_normalized', 'train_valid_standard_normalized',
    'train_valid_vus_pr', 'train_valid_auc_pr', 'train_valid_f1',
    'valid_vus_pr', 'valid_auc_pr', 'valid_f1',
    'test_vus_pr', 'test_auc_pr',
]
DIMENSIONS = {
    'window': 'Window size',
    'subset': 'Subset',
    'fill_nan': 'Fill NaN',
    'model_folder': 'Model',
    'score_normalization': 'Score normalization',
    'post_processing_strategy': 'Strategy',
    'threshold_type': 'Threshold type',
}
AUTO_REFRESH_INTERVALS = ['10s', '30s', '1m', '5m']
MAX_SHOWN_ROWS = 10_000   # rows of a table sent to the browser at most (All settings tab)
PARAMETER_COLUMNS = ['post_processing_strategy', 'threshold_type', 'topk', 'long_window', 'short_window', 'anomaly_threshold']
# A configuration: everything about a setting but the model, so that it can be applied to every model
CONFIG_COLUMNS = ['window', 'subset', 'fill_nan', 'score_normalization', *PARAMETER_COLUMNS]
# Strategies thresholded on the score itself: their absolute thresholds are each model's own highest-F1
# thresholds on train_valid, so they differ from one model to another for the same configuration; they are
# matched across models by their F1 rank (#1 = best F1 on train_valid)
SCORE_STRATEGIES = ('mahalanobis', 'mean_reconstruction_errors')
# Parts of the configuration the target reference can be fixed on, in the 'Target model' tab
TARGET_FIXED_COLUMNS = ['subset', 'window', 'score_normalization', 'fill_nan', 'post_processing_strategy', 'topk']
ANY = 'Any'
# One color per model family in every chart, whatever the filters: the hues of the paper figures
# (src/analysis_results.py: A3TGCN blue, GRU green) for known families, then free palette colors
FAMILY_COLORS = {'A3TGCN': '#4c78a8', 'GRU': '#54a24b', 'AnomalyTransformer': '#f58518'}
EXTRA_FAMILY_COLORS = ['#e45756', '#b279a2', '#72b7b2', '#eeca3b', '#9d755d', '#ff9da6', '#bab0ac']
NO_FAMILY_COLOR = '#8a8a8a'   # groups mixing several families


def family_color_scale(families):
    """Fixed family -> color scale, built from every family found on disk so filters do not shift it."""
    families = sorted(families)
    free = iter(c for c in EXTRA_FAMILY_COLORS if c not in FAMILY_COLORS.values())
    colors = {f: FAMILY_COLORS.get(f) or next(free, NO_FAMILY_COLOR) for f in families}
    return alt.Scale(domain=families, range=[colors[f] for f in families])


def family_colors(family_scale):
    """{family: color} of a family_color_scale."""
    return dict(zip(family_scale.domain, family_scale.range))
# Groups are colored by model family; Vega-Lite has no hatch patterns, so up to two other group
# dimensions are told apart by the fill shade and by the border style of the bars / boxes
FILL_SHADES = [1.0, 0.55, 0.25, 0.8, 0.4]
BORDER_DASHES = [[1, 0], [6, 3], [2, 2], [8, 3, 2, 3]]
# Lines (Strategies by window tab): strategies by their point shape and dash, models of a family by their shade
STRATEGY_SHAPES = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'triangle-down']
LINE_SHADES = [1.0, 0.55, 0.3]
LINE_DASHES = {False: [1, 0], True: [6, 3]}   # window charts of the results table: solid / dashed
CHART_BORDER = '#9e9e9e'   # border of the plots of the results-table charts
# Window charts of the results table sized as a figure of the paper: chart units are points (1/72 in), so the
# fonts have their size in points (src/figure_style.py) once the PNG is shown FIGURE_WIDTH_INCHES wide
FIGURE_WIDTH_INCHES = 6.5
POINTS_PER_INCH = 72
FIGURE_WIDTH = FIGURE_WIDTH_INCHES * POINTS_PER_INCH
PNG_DPI = 600
PANEL_COLUMNS = 4          # panels per row (one panel per group)
PANEL_SPACING = 4          # between the panels
PANEL_ASPECT = 0.8         # height / width of a panel
CHART_ASPECT = 0.6         # height / width of a single chart (no panels)
FONT_SIZE_INCREASE = 3     # points added to the font sizes of src/figure_style.py (the matplotlib figures)
Y_TICK_FONT_SIZE_DECREASE = 2   # y tick labels smaller than the other text (long numbers, e.g. 0.005)
LEGEND_OFFSET = 4          # between a legend and the plots
LEGEND_PADDING = 4         # between the frame of a legend and its entries
PANEL_AXIS_SPACE = 28 + 4 * (FONT_SIZE_INCREASE - Y_TICK_FONT_SIZE_DECREASE)   # y tick labels left of a panel, about
PANEL_TITLE_OFFSET = 4     # between the measure (left of the panels) and their y axes
CHART_GROUPINGS = ['One panel per group', 'All groups in one chart']
# Point shapes of the groups drawn in one chart: enough for the 8 subsets
GROUP_SHAPES = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'triangle-down', 'triangle-left',
                'triangle-right', 'wedge', 'arrow']


# STGformer pooled over the whole window (temporal_kernel_size = slide_win) is saved as <model>_pool<window>,
# a different folder name for every window: shown as one model, <model>_poolwin; other pool sizes keep theirs
POOL_SUFFIX = re.compile(r'^(?P<model>.+)_pool(?P<size>\d+)$')
WHOLE_WINDOW_POOL = '_poolwin'


def model_variant(model_folder, window):
    """'STGformer_pool12' at window 12 -> 'STGformer_poolwin': the same variant at every window."""
    match = POOL_SUFFIX.match(str(model_folder))
    if match and int(match['size']) == int(window):
        return match['model'] + WHOLE_WINDOW_POOL
    return model_folder


# Short model names of the results table: the model, '*' for the graph models with null padding (the sparsity
# indicator), and a short tag for the other variants (e.g. 'STGformer_null_padding_feature_poolwin' -> 'STGformer*')
# Whole folder names shown as given: Anomaly Transformer scored by its forecasting error (folder without suffix,
# as every other model) as the model itself, the association-based score of its paper tagged
FOLDER_SHORT_NAMES = {'AnomalyTransformer': 'Anom.Trans.', 'AnomalyTransformer_association': 'Anom.Trans.-assoc'}
# ClouDens backbones: A3TGCN (propagating on the operational-context graph) and BigST (learned graph; with the context
# graph as its prior, folders <model>..._spatial: '-ctx'); '*' with the sparsity indicator
CONTEXT_GRAPH_NAMES = {('A3TGCN', False): 'ClouDens-A3T', ('BigST', False): 'ClouDens-BigST',
                       ('BigST', True): 'ClouDens-BigST-ctx'}
PROPOSED_MODEL_PREFIX = 'ClouDens'
NULL_PADDING_SUFFIX = re.compile(r'_null_padding_(feature|target|both)')
CONTEXT_GRAPH_SUFFIX = '_spatial'
VARIANT_TAGS = {WHOLE_WINDOW_POOL: '', '_association': '-assoc', '_forecast': ''}


def short_model_name(model_folder):
    """'A3TGCN' -> 'ClouDens-A3T', 'A3TGCN_null_padding_feature' -> 'ClouDens-A3T*',
    'BigST_null_padding_feature' -> 'ClouDens-BigST*',
    'BigST_null_padding_feature_spatial' -> 'ClouDens-BigST-ctx*'."""
    name = str(model_folder)
    if name in FOLDER_SHORT_NAMES:
        return FOLDER_SHORT_NAMES[name]
    padded = bool(NULL_PADDING_SUFFIX.search(name))
    name = NULL_PADDING_SUFFIX.sub('', name)
    context = name.endswith(CONTEXT_GRAPH_SUFFIX)
    name = name.removesuffix(CONTEXT_GRAPH_SUFFIX)
    tags = []
    for suffix, tag in VARIANT_TAGS.items():
        if name.endswith(suffix):
            name, tags = name[:-len(suffix)], [tag] + tags
    match = POOL_SUFFIX.match(name)   # other pool sizes than the whole window: '-pool<k>'
    if match:
        name, tags = match['model'], [f'-pool{match["size"]}'] + tags
    if (name, context) in CONTEXT_GRAPH_NAMES:
        name = CONTEXT_GRAPH_NAMES[(name, context)]
    elif context:   # a context-graph variant of another model
        tags.append('+ctx')
    return name + ('*' if padded else '') + ''.join(tags)


def short_subset_name(subset):
    """'no_group_5xx_count' -> '5xx count' (as the paper's subset names, src/analysis_results.py)."""
    return ' '.join(str(subset).removeprefix('no_group_').split('_'))


# Families of the graph-based models (conf graph_models), listed after the other models in the results table
GRAPH_MODEL_FAMILIES = {'A3TGCN', 'GDN', 'GST', 'STGformer', 'MAGE', 'BigST'}


def model_order_key(model_folder):
    """Sort key of a model: the other models first, then the graph models, then ClouDens; by short name without
    the null-padding mark, each model followed by its variant with it"""
    name = short_model_name(model_folder)
    return (model_family(model_folder) in GRAPH_MODEL_FAMILIES, name.startswith(PROPOSED_MODEL_PREFIX),
            name.replace('*', ''), '*' in name)


def model_family(model_folder):
    """'A3TGCN_null_padding_feature' -> 'A3TGCN': the model without its variant suffix."""
    return str(model_folder).split('_')[0]


def _cycled(values, n):
    return [values[i % len(values)] for i in range(n)]


def group_style(data, group_by, family_scale):
    """
    Encodings shared by the group charts: color = model family; fill shade and border dash = the first
    two other group dimensions (the model folder itself counts when a family has several folders).
    Returns ({channel: encoding}, shade legend layer factory, dimensions without a visual channel).
    Vega-Lite draws no legend for fillOpacity: add shade_legend(data, x, y) as a layer to get one.
    """
    style = {}
    families = list(data['model_family'].unique()) if 'model_family' in data else []
    if 'model_folder' in group_by:
        style['color'] = alt.Color('model_family:N', title='Model family', scale=family_scale,
                                   legend=alt.Legend(orient='top', direction='vertical', labelLimit=0))
    elif len(families) == 1 and families[0] in family_scale.domain:   # one family selected: its color
        style['color'] = alt.value(family_scale.range[list(family_scale.domain).index(families[0])])
    else:
        style['color'] = alt.value(NO_FAMILY_COLOR)
    pattern_dims = [d for d in group_by if d != 'model_folder']
    if 'model_folder' in group_by and data.groupby('model_family')['model_folder'].nunique().max() > 1:
        pattern_dims = ['model_folder'] + pattern_dims
    shade_legend = None
    for channel, dim, values in zip(('fillOpacity', 'strokeDash'), pattern_dims, (FILL_SHADES, BORDER_DASHES)):
        domain = sorted(data[dim].astype(str).unique())
        scale = alt.Scale(domain=domain, range=_cycled(values, len(domain)))
        if channel == 'strokeDash':
            style[channel] = alt.StrokeDash(f'{dim}:N', title=DIMENSIONS[dim], scale=scale, legend=alt.Legend(
                orient='top', direction='vertical', labelLimit=0, symbolType='square', symbolFillColor='#bbbbbb',
                symbolStrokeColor='black'))
            continue
        style[channel] = alt.FillOpacity(f'{dim}:N', scale=scale, legend=None)
        legend = alt.Legend(orient='top', direction='vertical', labelLimit=0, symbolType='square', symbolSize=150,
                            symbolStrokeColor='black', symbolStrokeWidth=1)
        legend_channels = {'opacity': alt.Opacity(f'{dim}:N', title=DIMENSIONS[dim], scale=scale, legend=legend)}
        if dim == 'model_folder':
            # Each model in its family color: a fill scale on the same field merges with the opacity legend
            colors = family_colors(family_scale)
            legend_channels['fill'] = alt.Fill(f'{dim}:N', title=DIMENSIONS[dim], legend=legend, scale=alt.Scale(
                domain=domain, range=[colors.get(model_family(m), NO_FAMILY_COLOR) for m in domain]))

        def shade_legend(legend_data, x, y, legend_channels=legend_channels):
            # Invisible points carrying the same scale on the opacity channel, which has a legend; they
            # are grey (not Vega's default blue) unless they carry the family colors of the models
            return alt.Chart(legend_data).mark_point(size=0, filled=True, color='#555555').encode(
                x=x, y=y, **legend_channels)
    # Legends sit side by side above the chart, each listing its entries vertically: the tallest sets
    # the room they need (the 'mean' legend has one entry)
    legend_rows = max([1] + [len(data[d].astype(str).unique()) for d in pattern_dims[:2]] +
                      [len(families) if 'model_folder' in group_by else 0])
    return style, shade_legend, pattern_dims[2:], legend_rows


def group_x(order):
    """Groups along x, in the given order. Their labels (e.g. 'AnomalyTransformer_association ·
    likelihood_mahalanobis') are long: rotated, never truncated, and all shown (Vega would hide every
    other one where they overlap)."""
    return alt.X('group:N', sort=order, title=None, axis=alt.Axis(labelAngle=-45, labelLimit=0, labelOverlap=False))


def side_by_side(charts, n_groups, per_row, px_per_group=36):
    """One chart per measure, side by side and wrapped every per_row charts (so that many measures do
    not overflow the page), each with its own value scale and its own order of the groups."""
    width = max(200, px_per_group * n_groups)
    return alt.concat(*(c.properties(width=width, height=280) for c in charts), columns=per_row).resolve_scale(
        x='independent', y='independent')


def mean_dots(stats, x, fmt='.2f'):
    """Black dot at the mean measure of every group, with a 'mean' legend entry."""
    return alt.Chart(stats[['group', 'mean', 'median', 'min', 'max', 'settings']].assign(statistic='mean')).mark_point(
        filled=True, size=30, color='black', stroke='white', strokeWidth=0.5, opacity=1,
    ).encode(
        x=x,
        y=alt.Y('mean:Q'),
        # On the shape channel, not color: the model-family colors of the other layers stay one scale
        shape=alt.Shape('statistic:N', scale=alt.Scale(domain=['mean'], range=['circle']), title=None,
                        legend=alt.Legend(orient='top', direction='vertical', labelLimit=0, symbolFillColor='black')),
        tooltip=['group', alt.Tooltip('mean:Q', format=fmt), alt.Tooltip('median:Q', format=fmt),
                 alt.Tooltip('min:Q', format=fmt), alt.Tooltip('max:Q', format=fmt), 'settings'],
    )


def configuration_keys(rows, columns):
    """Configuration columns of the rows as text, for matching settings across models: NaN windows (strategies
    without them) match each other; the own absolute thresholds of the score strategies (different values for
    every model) match by their F1 rank on train_valid."""
    keys = rows[columns].astype(str)
    if 'anomaly_threshold' in columns:
        own_threshold = rows['f1_rank'].notna()
        keys.loc[own_threshold, 'anomaly_threshold'] = [f'own best-F1 threshold #{int(r)} (train_valid)'
                                                        for r in rows.loc[own_threshold, 'f1_rank']]
    return keys


def best_per_group(frame, keys, by):
    """The row with the highest `by` of every group of frame (by keys)."""
    return frame.loc[frame.groupby(keys)[by].idxmax()]


def join_columns(frame, columns, sep=' · '):
    """Values of the columns as text joined by sep, row by row: frame[columns].astype(str).agg(sep.join, axis=1),
    vectorised (no Python call per row of tables of millions of rows)."""
    if not columns:
        return pd.Series('', index=frame.index)
    text = frame[columns[0]].astype(str)
    for column in columns[1:]:
        text = text + sep + frame[column].astype(str)
    return text


def measure_format(measure):
    """Python number format of a measure, e.g. '{:.4f}'."""
    return '{:' + MEASURES[measure][2] + '}'


def measure_column_config(measures, prefix='', label=''):
    """st.dataframe number columns of the measures (columns prefix + measure column, titled label + measure)."""
    return {f'{prefix}{MEASURES[m][0]}': st.column_config.NumberColumn(f'{label}{m}', format=f'%{MEASURES[m][2]}')
            for m in measures}


def dimension_filters(container_of, options, key_prefix=''):
    """One multiselect per dimension, empty by default (= all its values, 'All' as placeholder): every value
    chosen by default made each rerun of the app filter and show the whole table. container_of(i) gives the
    container of the i-th filter. Returns {column: chosen values}."""
    return {column: container_of(i).multiselect(label, options[column][0], default=[], placeholder='All',
                                                key=f'{key_prefix}{column}' if key_prefix else None)
            for i, (column, label) in enumerate(DIMENSIONS.items())}


def dimension_options(results):
    """Filter options of every dimension (its values, sorted) and whether it has missing values."""
    return {column: (sorted(results[column].dropna().unique().tolist(), key=str), bool(results[column].isna().any()))
            for column in DIMENSIONS}


def filter_rows(results, options, chosen):
    """Rows whose dimensions take chosen values; nothing chosen for a dimension = all its values. A dimension
    with all its values chosen and no missing value restricts nothing: its (costly, on millions of rows) isin
    test is skipped, and results is returned as is when no dimension restricts it."""
    mask = None
    for column, values in chosen.items():
        all_values, has_missing = options[column]
        values = values or all_values
        if len(values) == len(all_values) and not has_missing:
            continue
        keep = results[column].isin(values)
        mask = keep if mask is None else mask & keep
    return results if mask is None else results[mask]


# ── Loading ───────────────────────────────────────────────────────────────────

def _topk_label(topk, strategy):
    """Top-k as text: the number of sensors, 'mean' (likelihood over all sensors) or '-' (not used)."""
    if pd.notna(topk) and str(topk) not in ('', 'None'):
        return str(int(float(topk)))
    return 'mean' if strategy == 'likelihood' else '-'


# Sources of the test anomalies in detection_counters: (key prefix, column of the detected anomalies)
ANOMALY_SOURCES = {'issue': 'issue_tracker', 'im': 'instant_messenger', 'TestLog': 'test_log'}
DETECTION_COLUMNS = ['detected', 'detected_ids', *ANOMALY_SOURCES.values()]
DETECTION_COLUMN_CONFIG = {
    'detected': st.column_config.TextColumn('Detected', help='Test anomalies detected (NAB TP) / all test anomalies'),
    'detected_ids': st.column_config.ListColumn('Detected anomaly ids'),
    'issue_tracker': st.column_config.TextColumn('Issue tracker', help='Detected / total: ids of the detected ones'),
    'instant_messenger': st.column_config.TextColumn('Instant messenger', help='Detected / total: ids of the detected ones'),
    'test_log': st.column_config.TextColumn('Test log', help='Detected / total: ids of the detected ones'),
}


def _parse_counters(counters):
    try:
        parsed = ast.literal_eval(counters)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _detection_summary(counters):
    """Detected test anomalies of a grid-search row, overall and per anomaly source."""
    summary, detected_ids, total = {}, [], 0
    for source, column in ANOMALY_SOURCES.items():
        ids = sorted(counters.get(f'{source}_detected_ids', []))
        n_source = len(counters.get(f'gt_{source}_ids', []))
        summary[column] = f'{len(ids)}/{n_source}' + (f': {", ".join(map(str, ids))}' if ids else '')
        detected_ids += ids
        total += n_source
    summary['detected'] = f'{len(detected_ids)}/{total}'
    summary['detected_ids'] = sorted(detected_ids)
    return summary


def prepare_results(results):
    """Explorer columns of a merged table (read_grid_files / read_merged): defaults for older grid searches,
    the F1 rank of the own absolute thresholds, the NAB detections, the top-k labels and model families; the
    rows of the ignored models (grid_results.is_ignored_model, e.g. from a merged file made before) are left out."""
    results = results.copy()
    # Older grid searches have no threshold_type: their likelihood thresholds were absolute values
    if 'threshold_type' not in results:
        results['threshold_type'] = None
    is_likelihood = results['post_processing_strategy'].str.startswith('likelihood')
    results['threshold_type'] = results['threshold_type'].fillna(
        is_likelihood.map({True: 'absolute', False: 'percentile'}))
    # Rank of the absolute thresholds of the score strategies by their F1 on train_valid, within each
    # file and strategy (#1 = best F1): the thresholds are the model's own, the rank is comparable
    own = results['post_processing_strategy'].isin(SCORE_STRATEGIES) & (results['threshold_type'] == 'absolute')
    results['f1_rank'] = float('nan')
    if own.any() and 'train_valid_f1' in results:
        results.loc[own, 'f1_rank'] = (results[own].groupby(['file', 'post_processing_strategy'])['train_valid_f1']
                                       .rank(ascending=False, method='first'))
    # Few distinct detection counters (~3% of the rows of a large merged file): each is parsed once
    codes, distinct = pd.factorize(results['detection_counters'])
    counters = [_parse_counters(c) for c in distinct] + [{}]   # code -1 (missing counters) -> {}
    per_counters = pd.DataFrame([_detection_summary(c) for c in counters])
    for key in ('tp', 'fp', 'fn'):
        per_counters[f'nab_{key}'] = [c.get(key) for c in counters]
    detections = per_counters.iloc[codes].set_axis(results.index)
    results = pd.concat([results, detections[['nab_tp', 'nab_fp', 'nab_fn', *DETECTION_COLUMNS]]], axis=1)
    results['topk'] = [_topk_label(k, s) for k, s in zip(results['topk'], results['post_processing_strategy'])]
    results['model_folder'] = [model_variant(m, w) for m, w in zip(results['model_folder'], results['window'])]
    results['model_family'] = results['model_folder'].map(model_family)
    # Ignored models: not merged any more, but a merged file written before may still hold them
    ignored = {m for m in results['model_folder'].unique() if is_ignored_model(m)}
    return results[~results['model_folder'].isin(ignored)] if ignored else results


# Source of the code reading and shaping the rows, passed to every cached loader (its cache key holds the
# arguments given, not the defaults): st.cache_data only hashes the source of the cached function itself, so without it a change to prepare_results (picked up by
# runOnSave) would keep serving the rows prepared by the previous version until the cache is cleared
LOADING_CODE = hashlib.md5((''.join(inspect.getsource(f) for f in (
    prepare_results, model_variant, model_family, _topk_label, _parse_counters, _detection_summary,
    read_grid_files, read_merged, is_ignored_model)) + repr(sorted(IGNORED_MODEL_FAMILIES))
    + repr(sorted(IGNORED_MODEL_FOLDERS))).encode()).hexdigest()


@st.cache_data(show_spinner='Reading grid-search files…', max_entries=8)
def merge_folder(files, loading_code):
    """Merged table of the grid-search files of a folder (what a merged file holds). `files`: (path, mtime)
    pairs, so that the cache is refreshed when a file changes."""
    return read_grid_files([path for path, _ in files])


@st.cache_data(show_spinner='Loading grid-search results…', max_entries=8)
def load_folder(files, loading_code):
    """Rows of the grid-search files of a folder (see merge_folder)."""
    return prepare_results(merge_folder(files, loading_code))


@st.cache_data(show_spinner='Loading merged file…', max_entries=8)
def load_merged_path(path, mtime, loading_code):
    """Rows of a merged file on disk; mtime refreshes the cache when it is rewritten."""
    return prepare_results(read_merged(path))


@st.cache_data(show_spinner='Loading merged file…', max_entries=4)
def load_merged_upload(_uploaded, file_id, name, loading_code):
    """Rows of an uploaded merged file. Cached by the upload's file_id: its bytes (hundreds of MB) are not
    hashed on every rerun (the leading underscore keeps _uploaded out of the cache key)."""
    return prepare_results(read_merged(io.BytesIO(_uploaded.getvalue()), name))


# ── Views ─────────────────────────────────────────────────────────────────────

# Detected test anomalies (NAB true positives) of the models, by the source of the anomaly
ANOMALY_SOURCE_NAMES = {'issue': 'Issue tracker', 'im': 'Instant messenger', 'TestLog': 'Test log'}
ANOMALY_SOURCE_SHORT = {'Issue tracker': 'Issue', 'Instant messenger': 'IM', 'Test log': 'Log'}
# Colors of the sources as in the detected-anomalies table of the paper (src/analysis_results.py)
ANOMALY_SOURCE_COLORS = {'Issue tracker': 'green', 'Instant messenger': 'orange', 'Test log': 'violet'}
CONFIGURATION_COLUMNS = ['window', 'subset', 'fill_nan', 'model_folder']   # one setting picked for each
# y axis (labels and title) left of the first column of panels, about: the panels share it
HEATMAP_AXIS_SPACE = 52
SCATTER_AXIS_SPACE = 40
BAR_AXIS_SPACE = 40
HEATMAP_ROW_HEIGHT = 7      # of an anomaly
BAR_Y_TITLE = 'Detected (mean)'   # the same in both layers: one title; short: one per row of panels


def anomaly_sources(rows):
    """{anomaly id: source name} of the test anomalies, from the ground truth of the detection counters."""
    counters = _parse_counters(rows['detection_counters'].dropna().iloc[0])
    return {i: name for source, name in ANOMALY_SOURCE_NAMES.items() for i in counters.get(f'gt_{source}_ids', [])}


def detection_data(picked, short_names):
    """The picked setting of every configuration with its model and subset as shown, models in table order."""
    data = picked.copy()
    models = sorted(data['model_folder'].unique(), key=model_order_key)
    name = short_model_name if short_names else str
    data['model'] = data['model_folder'].map(name)
    data['family'] = data['model_folder'].map(model_family)
    data['subset_shown'] = data['subset'].map(short_subset_name if short_names else str)
    subsets = sorted(data['subset_shown'].unique())
    return data, [name(m) for m in models], [model_family(m) for m in models], subsets


def model_scales(models, families, family_scale):
    """Color (family) and shape (rank in the family) scales of the models, and their legend layout."""
    colors = family_colors(family_scale)
    rank = pd.Series(families).groupby(families).cumcount().tolist()
    domain, columns = family_legend_layout(models, families)
    color_of = dict(zip(models, [colors.get(f, NO_FAMILY_COLOR) for f in families]))
    shape_of = dict(zip(models, [STRATEGY_SHAPES[r % len(STRATEGY_SHAPES)] for r in rank]))
    color = alt.Scale(domain=domain, range=[color_of.get(m, 'transparent') for m in domain])
    shape = alt.Scale(domain=domain, range=[shape_of.get(m, 'circle') for m in domain])
    legend = alt.Legend(orient='top', direction='horizontal', columns=columns, labelLimit=0, symbolSize=40,
                        symbolOpacity=1, columnPadding=8, rowPadding=1, titleOrient='left', titleAnchor='start',
                        titleBaseline='top', offset=LEGEND_OFFSET)
    return color, shape, legend


def facet_panels(layers, subsets, axis_space, height=None, columns=PANEL_COLUMNS):
    """One panel per subset, FIGURE_WIDTH wide in all; their shared y axis (axis_space wide) left of the first
    column only. height: of a panel (default: PANEL_ASPECT times its width)."""
    columns = min(columns, len(subsets))
    width = (FIGURE_WIDTH - axis_space - PANEL_SPACING * (columns - 1)) / columns
    return layers.properties(width=width, height=height or width * PANEL_ASPECT).facet(
        facet=alt.Facet('subset_shown:N', sort=subsets, title=None), columns=columns, spacing=PANEL_SPACING)


def detection_heatmap(data, models, subsets, sources):
    """(A) Share of the configurations of each subset (windows x fill NaN) in which each model detects each test
    anomaly: rows the anomalies by source, columns the models, one panel per subset."""
    n_configurations = data.groupby(['subset_shown', 'model']).size().rename('configurations')
    hits = (data[['subset_shown', 'model', 'detected_ids']].explode('detected_ids').dropna()
            .groupby(['subset_shown', 'model', 'detected_ids']).size().rename('hits'))
    ids = sorted(sources, key=lambda i: (list(ANOMALY_SOURCE_NAMES.values()).index(sources[i]), i))
    grid = pd.MultiIndex.from_product([subsets, models, ids], names=['subset_shown', 'model', 'detected_ids'])
    rate = hits.reindex(grid, fill_value=0).reset_index().join(n_configurations, on=['subset_shown', 'model'])
    rate = rate.dropna(subset=['configurations'])
    rate['detected (%)'] = 100 * rate['hits'] / rate['configurations']
    rate['anomaly'] = [f'{ANOMALY_SOURCE_SHORT[sources[i]]} #{i}' for i in rate['detected_ids']]
    anomalies = [f'{ANOMALY_SOURCE_SHORT[sources[i]]} #{i}' for i in ids]
    heat = alt.Chart().mark_rect().encode(
        x=alt.X('model:N', sort=models, title=None, axis=alt.Axis(labelAngle=-90, labelLimit=0, domain=False)),
        y=alt.Y('anomaly:N', sort=anomalies, title=None, axis=alt.Axis(domain=False, labelFontSize=TICK_FONT_SIZE)),
        color=alt.Color('detected (%):Q', scale=alt.Scale(scheme='blues', domain=[0, 100]),
                        legend=alt.Legend(orient='top', direction='horizontal', titleOrient='left',
                                          titleAnchor='start', titleBaseline='top', offset=LEGEND_OFFSET,
                                          gradientLength=160, title='Configurations detecting it (%)')),
        tooltip=['subset_shown', 'model', 'anomaly', alt.Tooltip('detected (%):Q', format='.0f'), 'hits',
                 'configurations'])
    return paper_style(facet_panels(alt.layer(heat, data=rate), subsets, HEATMAP_AXIS_SPACE,
                                    height=HEATMAP_ROW_HEIGHT * len(anomalies)))


def detection_scatter(data, models, families, subsets, family_scale):
    """(B) Detected anomalies against false positives of the picked setting of every configuration, with each
    model's Pareto front (most anomalies for at most so many false positives); one panel per subset."""
    color, shape, legend = model_scales(models, families, family_scale)
    points = data[['subset_shown', 'model', 'window', 'fill_nan', 'nab_tp', 'nab_fp']].dropna(subset=['nab_tp', 'nab_fp'])
    points = points.sort_values(['subset_shown', 'model', 'nab_fp', 'nab_tp'], ascending=[True, True, True, False])
    best_so_far = points.groupby(['subset_shown', 'model'])['nab_tp'].cummax()
    previous = best_so_far.groupby([points['subset_shown'], points['model']]).shift(fill_value=-1)
    points['front'] = points['nab_tp'] > previous   # the facet needs one data set: the front is a filter of it
    x = alt.X('nab_fp:Q', title='False positives', axis=alt.Axis(domain=False, tickCount=3, format='~s'))
    y = alt.Y('nab_tp:Q', title='Detected anomalies', scale=alt.Scale(domainMin=0),
              axis=alt.Axis(domain=False, tickMinStep=1))
    tooltip = ['model', 'window', 'fill_nan', alt.Tooltip('nab_tp:Q', title='detected'),
               alt.Tooltip('nab_fp:Q', title='false positives')]
    model_color = alt.Color('model:N', scale=color, title='Model', legend=legend)
    layers = alt.layer(
        alt.Chart().mark_line(interpolate='step-after', strokeWidth=1.2).encode(x=x, y=y, color=model_color)
        .transform_filter('datum.front'),
        alt.Chart().mark_point(filled=True, size=22, opacity=0.8).encode(
            x=x, y=y, color=model_color, shape=alt.Shape('model:N', scale=shape, title='Model', legend=legend),
            tooltip=tooltip),
        data=points)
    return paper_style(facet_panels(layers, subsets, SCATTER_AXIS_SPACE).resolve_scale(x='independent'))


def detection_bars(data, models, subsets, sources, nab_measure):
    """(C) Mean number of detected anomalies of every model over the configurations of a subset, stacked by
    source, with the mean NAB score written above the bar (Vega-Lite has no second y axis in panels); one panel
    per subset."""
    source_of = pd.Series(sources)
    counts = data[['subset_shown', 'model', 'detected_ids']].explode('detected_ids')
    counts['source'] = counts['detected_ids'].map(source_of)
    per_row = counts.dropna(subset=['source']).groupby([counts.dropna(subset=['source']).index, 'source']).size()
    per_row = per_row.unstack(fill_value=0).reindex(columns=list(ANOMALY_SOURCE_NAMES.values()), fill_value=0)
    per_row = per_row.reindex(data.index, fill_value=0)
    means = (pd.concat([data[['subset_shown', 'model']], per_row], axis=1)
             .groupby(['subset_shown', 'model']).mean()
             .reset_index().melt(id_vars=['subset_shown', 'model'], var_name='source', value_name='detected'))
    nab_column = MEASURES[nab_measure][0]
    nab = data.groupby(['subset_shown', 'model'])[nab_column].mean().rename('nab').reset_index()
    nab = nab.join(means.groupby(['subset_shown', 'model'])['detected'].sum().rename('total'),
                   on=['subset_shown', 'model'])
    both = pd.concat([means, nab], ignore_index=True)   # the facet needs one data set: bars and points filter it
    sources_order = list(ANOMALY_SOURCE_NAMES.values())
    x = alt.X('model:N', sort=models, title=None, axis=alt.Axis(labelAngle=-90, labelLimit=0, domain=False))
    bars = alt.Chart().mark_bar().encode(
        x=x, y=alt.Y('sum(detected):Q', title=BAR_Y_TITLE, axis=alt.Axis(domain=False)),
        color=alt.Color('source:N', title='Source', scale=alt.Scale(
            domain=sources_order, range=[ANOMALY_SOURCE_COLORS[s] for s in sources_order]),
            legend=alt.Legend(orient='top', direction='horizontal', titleOrient='left', titleAnchor='start',
                              titleBaseline='top', offset=LEGEND_OFFSET, symbolSize=40)),
        order=alt.Order('source_rank:Q'),
        tooltip=['model', 'source', alt.Tooltip('detected:Q', format='.2f')]
    ).transform_filter('isValid(datum.source)').transform_calculate(
        source_rank=f'indexof({sources_order}, datum.source)')
    labels = alt.Chart().transform_filter('isValid(datum.nab)').mark_text(
        angle=270, align='left', baseline='middle', dx=2, fontSize=TICK_FONT_SIZE).encode(
        x=x, y=alt.Y('total:Q', title=BAR_Y_TITLE, axis=alt.Axis(domain=False)),
        text=alt.Text('nab:Q', format='.1f'),
        tooltip=['model', alt.Tooltip('nab:Q', title=nab_measure, format='.1f')])
    layers = alt.layer(bars, labels, data=both)
    return paper_style(facet_panels(layers, subsets, BAR_AXIS_SPACE))


# Bubble chart of the likelihood windows: the measures it can show (label -> column, number format)
BUBBLE_MEASURES = {
    'NAB reward_fn': ('reward_fn_normalized', '.1f'),
    'NAB standard': ('standard_normalized', '.1f'),
    'Detected anomalies': ('nab_tp', '.0f'),
    'F1': ('f1', '.3f'),
}
BUBBLE_STATISTICS = ['Best setting', 'Max', 'Mean', 'Median']
BUBBLE_SIZES = [15, 280]          # area of the smallest / largest bubble
BUBBLE_AXIS_SPACE = 30            # y axis (labels and title) left of each panel, about
LIKELIHOOD_WINDOW_COLUMNS = ['short_window', 'long_window']


def bubble_charts(values, measure, panel_dims, short_names):
    """
    Bubble chart of the likelihood windows: x the short window, y the long window, the bubble (area and color)
    the value of the measure; one panel per value of panel_dims (e.g. the model), all on the same scale.
    values: one row per (panel_dims..., short_window, long_window), column 'value'.
    """
    column, fmt = BUBBLE_MEASURES[measure]
    data = values.reset_index()
    # Panels in the order of the results table: models by model_order_key, then the other dimensions
    order_key = data['model_folder'].map(model_order_key) if 'model_folder' in data else pd.Series(0, index=data.index)
    if short_names and 'model_folder' in data:
        data['model_folder'] = data['model_folder'].map(short_model_name)
    if short_names and 'subset' in data:
        data['subset'] = data['subset'].map(short_subset_name)
    data['panel'] = join_columns(data, panel_dims) if panel_dims else ''
    panels = list(dict.fromkeys(data.assign(_order=order_key).sort_values(['_order', 'panel'], kind='stable')['panel']))
    shorts = sorted(data['short_window'].unique())
    longs = sorted(data['long_window'].unique(), reverse=True)
    encodings = dict(
        x=alt.X('short_window:O', sort=shorts, title='Short window', axis=alt.Axis(labelAngle=0, domain=False)),
        y=alt.Y('long_window:O', sort=longs, title='Long window', axis=alt.Axis(domain=False)),
        size=alt.Size('value:Q', title=measure, scale=alt.Scale(zero=False, nice=False, range=BUBBLE_SIZES),
                      legend=alt.Legend(orient='top', direction='horizontal', titleOrient='left',
                                        titleAnchor='start', titleBaseline='top', offset=LEGEND_OFFSET,
                                        format=fmt)),
        color=alt.Color('value:Q', title=measure, scale=alt.Scale(scheme='viridis', zero=False, nice=False),
                        legend=alt.Legend(orient='top', direction='horizontal', titleOrient='left',
                                          titleAnchor='start', titleBaseline='top', offset=LEGEND_OFFSET,
                                          format=fmt, gradientLength=120)),
        tooltip=['panel', 'short_window', 'long_window', alt.Tooltip('value:Q', title=measure, format=fmt),
                 alt.Tooltip('settings:Q', title='settings')])
    base = alt.Chart().mark_circle(opacity=0.85, stroke='white', strokeWidth=0.5).encode(**encodings)
    layers = alt.layer(base, data=data)
    if panel_dims:
        columns = min(PANEL_COLUMNS, len(panels))
        width = panel_width(columns, BUBBLE_AXIS_SPACE)
        chart = layers.properties(width=width, height=width).facet(
            facet=alt.Facet('panel:N', sort=panels, title=None), columns=columns, spacing=PANEL_SPACING)
    else:
        width = FIGURE_WIDTH * 0.6
        chart = layers.properties(width=width, height=width)
    return paper_style(chart)


# Detection matrix (as Table IV of the paper): one row per model (and strategy), the test anomalies as columns
STRATEGY_SHORT_NAMES = {'likelihood': 'LF', 'likelihood_mahalanobis': 'LF-MD', 'mahalanobis': 'MD',
                        'mean_reconstruction_errors': 'MRE'}
NOT_DETECTED = 'Not detected'
MATRIX_LABEL_SPACE = 95     # row labels left of the matrix, about
MATRIX_ROW_HEIGHT = 11
NAB_BAR_WIDTH = 62
COUNTS_WIDTH = 46           # TP / FP column
# Two strategies in one cell: the triangles of a unit square (Vega symbol paths in [-1, 1]), as in the paper
TRIANGLE_PATHS = ('M -1 -1 L 1 -1 L -1 1 Z', 'M 1 -1 L 1 1 L -1 1 Z')   # upper-left, lower-right
STRATEGY_BAR_COLORS = ['#616161', '#bdbdbd']   # bars of the first / second strategy (or of the only one)


def confusion_counts(confusion_matrix):
    """(TP, FP) of the point-based confusion matrix [TN, FP, FN, TP] of a grid-search row."""
    try:
        tn, fp, fn, tp = ast.literal_eval(confusion_matrix)
        return int(tp), int(fp)
    except (ValueError, SyntaxError, TypeError):
        return None, None


def detection_matrix(table, sources, row_order, blocks_order, strategies=None):
    """
    (D) The test anomalies detected by each row (a model, or a model and strategy) as a matrix colored by the
    source of the anomaly, with its NAB scores (Standard, Low FN) and point-based TP / FP: Table IV of the
    paper as a figure. One block per value of the column 'block' (e.g. the subset, when the rows are models),
    in blocks_order. table: one row per (block, row[, strategy_shown]).
    strategies: two strategies (strategy_shown values) drawn in the same cell, split along its diagonal (the
    first one the upper-left triangle, as in the paper's detected-anomalies table), with two bars / counts per
    row; None: one strategy per row.
    """
    ids = sorted(sources, key=lambda i: (list(ANOMALY_SOURCE_NAMES.values()).index(sources[i]), i))
    anomalies = [f'#{i}' for i in ids]
    source_names = list(ANOMALY_SOURCE_NAMES.values())
    split = bool(strategies)
    row_height = MATRIX_ROW_HEIGHT * (1.5 if split else 1)   # room for two bars per row
    matrix_width = FIGURE_WIDTH - MATRIX_LABEL_SPACE - 2 * NAB_BAR_WIDTH - COUNTS_WIDTH - 3 * PANEL_SPACING
    source_scale = alt.Scale(domain=source_names + [NOT_DETECTED],
                             range=[ANOMALY_SOURCE_COLORS[s] for s in source_names] + ['white'])
    # Short labels: the legends must not be wider than the matrix (they would push the bars away)
    source_legend = alt.Legend(
        orient='top', direction='horizontal', columns=2, titleOrient='left', titleAnchor='start',
        titleBaseline='top', offset=LEGEND_OFFSET, symbolStrokeColor='#bdbdbd', symbolStrokeWidth=0.5,
        labelExpr=' || '.join(f"(datum.label == '{name}' && '{short}')" for name, short in
                              {**ANOMALY_SOURCE_SHORT, NOT_DETECTED: 'Missed'}.items()) + ' || datum.label')
    blocks = []
    for block_name in blocks_order:
        rows = table[table['block'] == block_name]
        if rows.empty:
            continue
        order = [r for r in row_order if r in set(rows['row'])]
        height = row_height * len(order)
        first = not blocks   # the legends once, above the first block
        source_color = alt.Color('shown:N', title='Source', scale=source_scale,
                                 legend=source_legend if first else None)
        x = alt.X('anomaly:N', sort=anomalies, title='Ground-truth anomaly',
                  axis=alt.Axis(labelAngle=-90, domain=False, ticks=False))
        y = alt.Y('row:N', sort=order, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=0))
        no_y = alt.Y('row:N', sort=order, axis=None)
        keys = ['row', 'strategy_shown'] if split else ['row']
        detected = {(*k, i) for *k, found in zip(*(rows[c] for c in keys), rows['detected_ids'])
                    for i in (found or [])}
        cells = pd.DataFrame([(r, i) for r in order for i in ids], columns=['row', 'id'])
        cells['anomaly'] = [f'#{i}' for i in cells['id']]
        cells['source'] = [sources[i] for i in cells['id']]
        if split:
            halves = pd.DataFrame([(r, st_, i) for r in order for st_ in strategies for i in ids],
                                  columns=['row', 'strategy_shown', 'id'])
            halves = halves[[(r, st_, i) in detected for r, st_, i in
                             zip(halves['row'], halves['strategy_shown'], halves['id'])]]
            halves['anomaly'] = [f'#{i}' for i in halves['id']]
            halves['shown'] = [sources[i] for i in halves['id']]
            side = min(matrix_width / len(ids), row_height)
            grid = alt.Chart(cells).mark_rect(fill='white', stroke='#bdbdbd', strokeWidth=0.5).encode(
                x=x, y=y, tooltip=['row', 'anomaly', 'source'])
            triangles = alt.Chart(halves).mark_point(filled=True, opacity=1, strokeWidth=0, size=side ** 2).encode(
                x=x, y=y, color=source_color,
                shape=alt.Shape('strategy_shown:N', title='Strategy', scale=alt.Scale(
                    domain=strategies, range=list(TRIANGLE_PATHS)),
                    legend=alt.Legend(orient='top', direction='horizontal', titleOrient='left',
                                      titleAnchor='start', titleBaseline='top', offset=LEGEND_OFFSET,
                                      symbolFillColor='#555555', symbolStrokeWidth=0) if first else None),
                tooltip=['row', 'strategy_shown', 'anomaly', alt.Tooltip('shown:N', title='source')])
            matrix = alt.layer(grid, triangles)
        else:
            cells['shown'] = [s if (r, i) in detected else NOT_DETECTED for r, i, s in
                              zip(cells['row'], cells['id'], cells['source'])]
            matrix = alt.Chart(cells).mark_rect(stroke='#bdbdbd', strokeWidth=0.5).encode(
                x=x, y=y, color=source_color,
                tooltip=['row', 'anomaly', 'source', alt.Tooltip('shown:N', title='detected as')])
        parts = [matrix.properties(width=matrix_width, height=height)]
        # Two strategies: two bars / counts per row, the first one darker (yOffset)
        offset = dict(yOffset=alt.YOffset('strategy_shown:N', sort=strategies)) if split else {}
        bar_color = (alt.Color('strategy_shown:N', scale=alt.Scale(domain=strategies, range=STRATEGY_BAR_COLORS),
                               legend=None) if split else alt.value(STRATEGY_BAR_COLORS[1]))
        for title, column in (('NAB Std.', 'standard_normalized'), ('NAB Low FN', 'reward_fn_normalized')):
            base = alt.Chart(rows).encode(y=no_y, **offset)
            bar = base.mark_bar(height=(row_height - 3) / (2 if split else 1)).encode(
                x=alt.X(f'{column}:Q', title=title, axis=alt.Axis(domain=False, tickCount=2)), color=bar_color,
                tooltip=['row', *(['strategy_shown'] if split else []),
                         alt.Tooltip(f'{column}:Q', title=title, format='.2f')])
            label = base.mark_text(align='left', dx=2, fontSize=TICK_FONT_SIZE - (2 if split else 0)).encode(
                x=alt.X(f'{column}:Q'), text=alt.Text(f'{column}:Q', format='.1f'))
            parts.append(alt.layer(bar, label).properties(width=NAB_BAR_WIDTH, height=height))
        parts.append(alt.Chart(rows).mark_text(fontSize=TICK_FONT_SIZE - (2 if split else 0)).encode(
            y=no_y, **offset, text='tp_fp:N').properties(
            width=COUNTS_WIDTH, height=height,
            title=alt.TitleParams('TP / FP', orient='bottom', fontSize=AXIS_LABEL_FONT_SIZE + FONT_SIZE_INCREASE,
                                  fontWeight='bold')))
        # The grey bars of the strategies must not share the color scale of the sources
        block = alt.hconcat(*parts, spacing=PANEL_SPACING).resolve_scale(color='independent')
        blocks.append(block.properties(title=alt.TitleParams(block_name, anchor='start'))
                      if len(blocks_order) > 1 else block)
    if not blocks:
        return None
    return paper_style(alt.vconcat(*blocks, spacing=2 * PANEL_SPACING) if len(blocks) > 1 else blocks[0])


DETECTION_ROWS = ['Models', 'Subsets']


def render_detection_matrix(rows, select_by, short_names):
    """(D) Detection matrix of chosen configurations (a window and fill NaN): one row per model (and strategy), one
    block per subset; or one row per subset, one block per model (as the paper's detected-anomalies table)."""
    st.markdown('**(D) Detected anomalies, NAB and TP / FP of one configuration (Table IV of the paper)**: each '
                'row with its setting picked as above.')
    rows_by = st.segmented_control('Rows', DETECTION_ROWS, default='Models', required=True, key='det_matrix_rows',
                                   help='Models: one row per model, one block per subset. Subsets: one row per '
                                        'subset, one block per model.')
    by_subset = rows_by == 'Subsets'
    col_window, col_fill, col_subsets, col_models, col_strategies = st.columns([1, 1, 2, 2, 2])
    windows = sorted(rows['window'].dropna().unique())
    window = col_window.selectbox('Window size', windows, key='det_matrix_window')
    fills = sorted(rows['fill_nan'].dropna().unique())
    fill = col_fill.selectbox('Fill NaN', fills, key='det_matrix_fill',
                              index=fills.index('zero') if 'zero' in fills else 0)
    subsets = sorted(rows['subset'].dropna().unique())
    chosen_subsets = col_subsets.multiselect(
        'Subsets', subsets, key=f'det_matrix_subsets_{rows_by}', format_func=short_subset_name,
        default=subsets if by_subset else
        (['no_group_5xx_count'] if 'no_group_5xx_count' in subsets else subsets[:1]))
    models = sorted(rows['model_folder'].dropna().unique(), key=model_order_key)
    proposed = [m for m in models if short_model_name(m).startswith(PROPOSED_MODEL_PREFIX)]
    chosen_models = col_models.multiselect(
        'Models', models, key=f'det_matrix_models_{rows_by}', format_func=short_model_name,
        default=proposed[:1] if by_subset else models, placeholder='Choose models')
    strategies = sorted(rows['post_processing_strategy'].dropna().unique())
    chosen_strategies = col_strategies.multiselect(
        'Strategies (one row each)', strategies, key='det_matrix_strategies', placeholder='Best of all strategies',
        format_func=lambda s: STRATEGY_SHORT_NAMES.get(s, s),
        help='Empty: one row per model (or subset), its best setting over all the strategies.')
    split = len(chosen_strategies) == 2 and st.toggle(
        'Both strategies in one cell', value=True, key='det_matrix_split',
        help='Each cell split along its diagonal: the first strategy the upper-left triangle, the second one the '
             'lower-right one; two bars and counts per row.')
    part = rows[(rows['window'] == window) & (rows['fill_nan'] == fill) & rows['subset'].isin(chosen_subsets)
                & rows['model_folder'].isin(chosen_models)]
    if chosen_strategies:
        part = part[part['post_processing_strategy'].isin(chosen_strategies)]
    keys = ['subset', 'model_folder'] + (['post_processing_strategy'] if chosen_strategies else [])
    table = best_per_group(part.dropna(subset=[select_by]), keys, select_by)
    if table.empty:
        st.info('No setting for this configuration.')
        return
    model_name = short_model_name if short_names else str
    subset_name = short_subset_name if short_names else str
    table['strategy_shown'] = table['post_processing_strategy'].map(lambda s: STRATEGY_SHORT_NAMES.get(s, s))
    if by_subset:
        table['row'], table['block'] = table['subset'].map(subset_name), table['model_folder'].map(model_name)
        row_order = [subset_name(s) for s in chosen_subsets]
        blocks_order = [model_name(m) for m in models if m in chosen_models]
    else:
        table['row'], table['block'] = table['model_folder'].map(model_name), table['subset'].map(subset_name)
        order_key = table['model_folder'].map(model_order_key)
        row_order = list(dict.fromkeys(table.assign(_o=order_key).sort_values(['_o', 'row'], kind='stable')['row']))
        blocks_order = [subset_name(s) for s in chosen_subsets]
    if chosen_strategies and not split:
        table['row'] = table['row'] + ' · ' + table['strategy_shown']
        row_order = [f'{r} · {STRATEGY_SHORT_NAMES.get(s, s)}' for r in row_order for s in chosen_strategies]
    counts = [confusion_counts(c) for c in table['confusion_matrix']]
    table['tp_fp'] = [f'{tp} / {fp}' if tp is not None else '' for tp, fp in counts]
    chart = detection_matrix(table, anomaly_sources(table), row_order, blocks_order,
                             [STRATEGY_SHORT_NAMES.get(s, s) for s in chosen_strategies] if split else None)
    if chart is None:
        st.info('No setting for this configuration.')
        return
    st.altair_chart(chart, width='content')
    export_buttons(chart, f'detection_matrix_by_{rows_by.lower()}_w{window}_{fill}', 'det_matrix')
    with st.expander('Settings of the rows'):
        shown = ['block', 'row', 'post_processing_strategy', 'threshold_type', 'topk', 'long_window',
                 'short_window', 'anomaly_threshold', 'tp_fp', 'standard_normalized', 'reward_fn_normalized',
                 'detected', select_by]
        st.dataframe(table[[c for c in dict.fromkeys(shown) if c in table]], hide_index=True)


def render_detections(rows, family_scale):
    """Detected test anomalies of the models, one panel per subset: (A) heatmap anomaly x model, (B) detected
    anomalies against false positives, (C) detected anomalies by source with the NAB score."""
    st.subheader('Detected anomalies', divider='gray')
    col_select, col_nab, col_short = st.columns([2, 2, 1])
    available_metrics = [m for m in SELECTION_METRICS if m in rows.columns and rows[m].notna().any()]
    select_by = col_select.selectbox(
        'Setting of each configuration picked by', available_metrics, key='det_select_by',
        index=available_metrics.index('train_valid_f1') if 'train_valid_f1' in available_metrics else 0,
        help='A configuration is a window, subset, fill NaN and model: its setting (strategy, threshold, ...) with '
             'the highest value of this metric. train_valid_* / valid_* pick without the test labels; NAB / F1 '
             'pick on the test split itself.')
    nab_measures = [m for m in ('NAB reward_fn', 'NAB standard') if MEASURES[m][0] in rows]
    nab_measure = col_nab.selectbox('NAB score (C)', nab_measures, key='det_nab')
    short_names = col_short.toggle('Short names', value=True, key='det_short_names')
    picked = best_per_group(rows.dropna(subset=[select_by]), CONFIGURATION_COLUMNS, select_by)
    if picked.empty or picked['detection_counters'].isna().all():
        st.info('No detection counters for these settings.')
        return
    sources = anomaly_sources(picked)
    data, models, families, subsets = detection_data(picked, short_names)
    n_configurations = data.groupby(['subset_shown', 'model']).size()
    st.caption(f'One setting per configuration (window x fill NaN x model of a subset), picked by `{select_by}`: '
               f'{n_configurations.min()}–{n_configurations.max()} configurations per model and subset.')

    st.markdown('**(A) Which anomalies each model detects**: share of the configurations of the subset in '
                'which the model detects the anomaly (rows grouped by source).')
    chart = detection_heatmap(data, models, subsets, sources)
    st.altair_chart(chart, width='content')
    export_buttons(chart, 'detected_anomalies_heatmap', 'det_heatmap')

    st.markdown('**(B) Detected anomalies against false positives**: one point per configuration; line: the '
                'Pareto front of the model (most anomalies detected for at most so many false positives).')
    chart = detection_scatter(data, models, families, subsets, family_scale)
    st.altair_chart(chart, width='content')
    export_buttons(chart, 'detected_anomalies_vs_false_positives', 'det_scatter')

    st.markdown(f'**(C) Detected anomalies by source**: mean over the configurations of the subset; above each '
                f'bar, the mean {nab_measure}.')
    chart = detection_bars(data, models, subsets, sources, nab_measure)
    st.altair_chart(chart, width='content')
    export_buttons(chart, 'detected_anomalies_by_source', 'det_bars')

    render_detection_matrix(rows, select_by, short_names)


def render_likelihood_windows(results, options, family_scale):
    """
    'Likelihood windows' tab: the detected test anomalies of the models (render_detections), then the measures
    of the likelihood strategies over their short and long windows, as bubbles. Its filters are its own: the
    sidebar filters do not apply.
    """
    st.caption('This tab has its own filters: the filters of the sidebar do not apply here.')
    with st.expander('Filters', expanded=True, icon=':material/filter_alt:'):
        filter_columns = st.columns(4)
        st.caption('Empty filter: all its values. Choose at least one filter to draw the charts.')
        chosen = dimension_filters(lambda i: filter_columns[i % 4], options, key_prefix='bubble_filter_')
    if not any(chosen.values()):
        st.info('Choose at least one filter (e.g. a subset and a window size).')
        return
    rows = filter_rows(results, options, chosen)
    if rows.empty:
        st.info('No setting matches the filters.')
        return
    render_detections(rows, family_scale)

    st.subheader('Likelihood windows', divider='gray')
    st.caption('Likelihood strategies only (the others have no short / long window).')
    if not set(LIKELIHOOD_WINDOW_COLUMNS) <= set(rows.columns):
        st.info('These results have no short / long window columns.')
        return
    rows = rows[rows['post_processing_strategy'].str.startswith('likelihood')
                & rows['short_window'].gt(0) & rows['long_window'].gt(0)]
    if rows.empty:
        st.info('No likelihood setting matches the filters.')
        return

    col_measure, col_statistic, col_select = st.columns(3)
    available = [m for m, (c, _) in BUBBLE_MEASURES.items() if c in rows and rows[c].notna().any()]
    measure = col_measure.selectbox('Bubble', available, key='bubble_measure',
                                    help='Detected anomalies: test anomalies detected (NAB true positives).')
    statistic = col_statistic.segmented_control(
        'Value of a bubble', BUBBLE_STATISTICS, default='Best setting', required=True, key='bubble_statistic',
        help='Over the other settings of the bubble (thresholds, top-k, ...): the measure of the best one, '
             'or the max / mean / median of the measure.')
    available_metrics = [m for m in SELECTION_METRICS if m in rows.columns and rows[m].notna().any()]
    select_by = col_select.selectbox(
        'Best setting by', available_metrics, key='bubble_select_by', disabled=statistic != 'Best setting',
        index=available_metrics.index('train_valid_f1') if 'train_valid_f1' in available_metrics else 0)
    col_panels, col_short = st.columns([3, 1])
    panel_dims = col_panels.multiselect('One panel per', list(DIMENSIONS), default=['model_folder'],
                                        format_func=DIMENSIONS.get, key='bubble_panels')
    short_names = col_short.toggle('Short names', value=True, key='bubble_short_names')

    column = BUBBLE_MEASURES[measure][0]
    keys = [*panel_dims, *LIKELIHOOD_WINDOW_COLUMNS]
    groups = rows.groupby(keys)
    if statistic == 'Best setting':
        picked = best_per_group(rows.dropna(subset=[select_by]), keys, select_by)
        values = picked.set_index(keys)[[column]].rename(columns={column: 'value'})
    else:
        values = groups[column].agg(statistic.lower()).to_frame('value')
    values['settings'] = groups.size()
    if values['value'].notna().sum() == 0:
        st.info(f'No value of {measure} for these settings.')
        return
    if statistic == 'Max' and measure == 'Detected anomalies':
        st.caption('The most anomalies detected by any threshold: low thresholds detect many, with many false '
                   'positives too. *Best setting* picks one setting per bubble without the test labels.')
    chart = bubble_charts(values, measure, panel_dims, short_names)
    st.altair_chart(chart, width='content')
    export_buttons(chart, f'likelihood_windows_{measure}'.replace(' ', '_'), 'bubble')
    with st.expander('Values of the bubbles'):
        st.dataframe(values.reset_index(), hide_index=True)


def render_explorer(results):
    """Filters (sidebar) and tabs of the explorer over the rows of a merged table, whatever its source."""
    family_scale = family_color_scale(results['model_family'].unique())

    # ── Filters ──
    with st.sidebar:
        st.header('Filters')
        options = dimension_options(results)
        st.caption('Empty filter: all its values. Choose at least one filter to explore the results.')
        selected = dimension_filters(lambda i: st, options)

        st.header('Comparison')
        available_measures = [m for m, (c, _, _) in MEASURES.items() if c in results and results[c].notna().any()]
        measures = st.multiselect(
            'Measures (test)', available_measures,
            default=[m for m in ('NAB reward_fn', 'AUC-PR', 'VUS-PR') if m in available_measures],
            help='Test measures shown by the charts (one chart each) and the tables. The first one sorts the tables.')
        if not measures:
            measures = available_measures[:1]
            st.caption(f'No measure chosen: showing {measures[0]}.')
        measure_columns = [MEASURES[m][0] for m in measures]
        charts_per_row = st.slider('Charts per row', 1, 6, 3,
                                   help='Charts of the measures wrap onto new rows past this number.')
        measure = measures[0]   # sorts the tables and sets the default selection metric
        measure_column, measure_title, _ = MEASURES[measure]
        available_metrics = [m for m in SELECTION_METRICS if m in results.columns]
        default_select_by = measure_column if measure_column in available_metrics else NAB_COLUMNS['reward_fn']
        select_by = st.selectbox(
            'Pick the best setting of each group by', available_metrics,
            index=available_metrics.index(default_select_by), key=f'select_by_{measure}',
            help='NAB / F1 select on the test split itself; train_valid_* / valid_* select without the test labels.')
        group_by = st.multiselect('Compare groups of', list(DIMENSIONS), default=['model_folder', 'post_processing_strategy'],
                                  format_func=DIMENSIONS.get)

    # Nothing chosen: the whole table (millions of rows) would be filtered and charted on every rerun
    filtered = filter_rows(results, options, selected) if any(selected.values()) else None

    tab_overview, tab_compare, tab_target, tab_windows, tab_rows, tab_table, tab_bubbles = st.tabs(
        ['Overview', 'Compare', 'Target model', 'Strategies by window', 'All settings', 'Results table',
         'Likelihood windows'])

    # ── Results table and likelihood windows: their own filters, shown whatever the sidebar filters ──
    with tab_table:
        render_results_table(results, family_scale, options)
    with tab_bubbles:
        render_likelihood_windows(results, options, family_scale)

    if filtered is None or filtered.empty:
        for tab in (tab_overview, tab_compare, tab_target, tab_windows, tab_rows):
            tab.info('Choose at least one filter in the sidebar (e.g. a subset and a window size) to explore '
                     f'the {len(results):,} grid-search settings.' if filtered is None else
                     'No grid-search row matches the filters of the sidebar.')
        return

    # ── Overview: what has been run ──
    with tab_overview:
        cols = st.columns(5)
        for col, key in zip(cols, ('window', 'subset', 'fill_nan', 'model_folder', 'score_normalization')):
            values = sorted(results[key].unique().tolist(), key=str)
            col.metric(DIMENSIONS[key], len(values))
            col.caption(', '.join(map(str, values)))

        st.subheader('Grid-search files')
        inventory = results.groupby('file').agg(
            **{key: (key, 'first') for key in TAG_COLUMNS if key != 'file'},
            settings=('post_processing_strategy', 'size'),
            strategies=('post_processing_strategy', lambda s: ', '.join(sorted(s.unique()))),
            **{f'best_{c}': (c, 'max') for c in measure_columns}).reset_index()
        split_columns = results.filter(like='train_valid_').notna().any(axis=1)
        inventory['has_split_metrics'] = inventory['file'].map(split_columns.groupby(results['file']).any())
        st.dataframe(
            inventory[['window', 'subset', 'fill_nan', 'model_folder', 'score_normalization', 'settings', 'strategies',
                       *[f'best_{c}' for c in measure_columns], 'has_split_metrics', 'modified', 'file']],
            hide_index=True,
            column_config={
                **measure_column_config(measures, prefix='best_', label='Best '),
                'has_split_metrics': st.column_config.CheckboxColumn('Train/valid metrics'),
                'modified': st.column_config.DatetimeColumn('Modified', format='YYYY-MM-DD HH:mm'),
            })

    # ── Compare: best setting of every group ──
    with tab_compare:
        if not group_by:
            st.info('Choose at least one dimension to compare in the sidebar.')
        else:
            ranked = filtered.dropna(subset=[select_by])
            if ranked.empty:
                st.info(f'No row has `{select_by}` (older grid searches lack the train/valid metrics).')
            else:
                best = best_per_group(ranked, group_by, select_by).copy()
                best['group'] = join_columns(best, group_by)
                best = best.sort_values(measure_column, ascending=False)

                st.caption(f'Best setting of each group by **{select_by}**; one chart per test measure, '
                           'groups by decreasing value.')
                best_data = best.astype({d: str for d in group_by})
                style, shade_legend, unstyled, _ = group_style(best_data, group_by, family_scale)
                charts = []
                for m in measures:
                    column, title, fmt = MEASURES[m]
                    x = group_x(best.sort_values(column, ascending=False)['group'].tolist())
                    chart = alt.Chart(best_data).mark_bar(stroke='black', strokeWidth=1).encode(
                        x=x, y=alt.Y(f'{column}:Q', title=title), **style,
                        tooltip=['group', alt.Tooltip(f'{column}:Q', format=fmt), alt.Tooltip(f'{select_by}:Q', format='.4f'),
                                 *PARAMETER_COLUMNS, 'nab_tp', 'nab_fp', 'nab_fn', 'detected',
                                 *ANOMALY_SOURCES.values()],
                    )
                    if shade_legend:
                        chart = chart + shade_legend(best_data, x, f'{column}:Q')
                    charts.append(chart.properties(title=m))
                st.altair_chart(side_by_side(charts, len(best), charts_per_row))
                if unstyled:
                    st.caption(f'{", ".join(DIMENSIONS[d] for d in unstyled)}: shown in the labels only '
                               '(color, shade and border already encode other dimensions).')

                shown = group_by + [c for c in PARAMETER_COLUMNS if c not in group_by] + \
                    [select_by] * (select_by not in NAB_COLUMNS.values()) + measure_columns + \
                    ['reward_fn_normalized', 'standard_normalized', *DETECTION_COLUMNS, 'nab_tp', 'nab_fp', 'nab_fn',
                     'f1', 'test_auc_pr', 'test_vus_pr']
                shown = [c for c in dict.fromkeys(shown) if c in best.columns]
                st.dataframe(best[shown], hide_index=True,
                             column_config={**{c: st.column_config.NumberColumn(format='%.4f')
                                               for c in shown if c.endswith(('_pr', 'f1'))},
                                            **DETECTION_COLUMN_CONFIG})

            st.subheader('Over all settings: median and mean')
            spread = filtered.copy()
            spread['group'] = join_columns(spread, group_by)
            # Family of each group; '' when a group mixes several families (not grouped by model)
            group_family = spread.groupby('group')['model_family'].agg(lambda f: f.iloc[0] if f.nunique() == 1 else '')

            def group_stats(column):
                """Spread of a measure over the settings of every group, by decreasing median."""
                stats = (spread.groupby(group_by + ['group'])[column]
                         .agg(median='median', mean='mean', min='min', max='max', settings='size',
                              q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75))
                         .sort_values('median', ascending=False).reset_index())
                stats['model_family'] = stats['group'].map(group_family)
                return stats

            all_stats = {m: group_stats(MEASURES[m][0]) for m in measures}
            st.caption('Bar: median of the test measure over every setting of each group; dot: mean. '
                       'Groups by decreasing median.')
            style, shade_legend, unstyled, _ = group_style(all_stats[measure].astype({d: str for d in group_by}),
                                                           group_by, family_scale)
            charts = []
            for m, stats in all_stats.items():
                _, title, fmt = MEASURES[m]
                stats_data = stats.astype({d: str for d in group_by})
                x = group_x(stats['group'].tolist())
                stats_tooltip = ['group', *(alt.Tooltip(f'{c}:Q', format=fmt) for c in ('median', 'mean', 'min', 'max')),
                                 'settings']
                layers = [alt.Chart(stats_data).mark_bar(stroke='black', strokeWidth=1).encode(
                              x=x, y=alt.Y('median:Q', title=title), tooltip=stats_tooltip, **style),
                          mean_dots(stats, x, fmt)]
                if shade_legend:
                    layers.append(shade_legend(stats_data, x, 'median:Q'))
                charts.append(alt.layer(*layers).properties(title=m))
            st.altair_chart(side_by_side(charts, len(group_family), charts_per_row))

            with st.expander('Distribution in each group', icon=':material/candlestick_chart:'):
                st.caption('Line: min to max; box: quartiles; black bar in the box: median; dot: mean. '
                           'Color: model family; shade and border: the other group dimensions.')
                charts = []
                for m, stats in all_stats.items():
                    _, title, fmt = MEASURES[m]
                    box_data = stats.astype({d: str for d in group_by})
                    x = group_x(stats['group'].tolist())
                    tooltip = ['group', *(alt.Tooltip(f'{c}:Q', format=fmt)
                                          for c in ('median', 'mean', 'q1', 'q3', 'min', 'max')), 'settings']
                    base = alt.Chart(box_data)
                    whiskers = base.mark_rule(color='#666').encode(
                        x=x, y=alt.Y('min:Q', title=title), y2='max:Q', tooltip=tooltip)
                    # Boxes narrower than the band leave a gap between groups
                    boxes = base.mark_bar(width={'band': 0.6}, stroke='black', strokeWidth=1.2).encode(
                        x=x, y='q1:Q', y2='q3:Q', tooltip=tooltip, **style)
                    medians = base.mark_tick(color='black', thickness=2.5, width={'band': 0.6}).encode(
                        x=x, y='median:Q', tooltip=tooltip)
                    layers = [whiskers, boxes, medians, mean_dots(stats, x, fmt)]
                    if shade_legend:
                        layers.append(shade_legend(box_data, x, 'median:Q'))
                    charts.append(alt.layer(*layers).properties(title=m))
                st.altair_chart(side_by_side(charts, len(group_family), charts_per_row))
                if unstyled:
                    st.caption(f'{", ".join(DIMENSIONS[d] for d in unstyled)}: shown in the labels only.')
                table = pd.concat([stats.assign(measure=m) for m, stats in all_stats.items()])
                st.dataframe(table[['measure', 'group', 'median', 'mean', 'q1', 'q3', 'min', 'max', 'settings']],
                             hide_index=True,
                             column_config={c: st.column_config.NumberColumn(format='%.4f')
                                            for c in ('median', 'mean', 'q1', 'q3', 'min', 'max')})

    # ── Target model: its best configuration, applied to the other models ──
    with tab_target:
        models = sorted(filtered['model_folder'].unique())
        config_columns = [c for c in CONFIG_COLUMNS if c in filtered]
        col_target, col_match = st.columns([1, 3])
        target = col_target.selectbox('Target model', models,
                                      index=next((i for i, m in enumerate(models) if model_family(m) == 'A3TGCN'), 0))
        match_on = col_match.multiselect(
            'Other models use the same', config_columns, default=config_columns, format_func=lambda c: DIMENSIONS.get(c, c),
            help='Values taken from the best configuration of the target model. Dimensions left out are free: '
                 f'every model takes its best value of them by {select_by}.')
        ranked = filtered.dropna(subset=[select_by])
        target_rows = ranked[ranked['model_folder'] == target]
        # Fixed parts of the target reference: its best configuration is searched among the other parts only
        for col, column in zip(st.columns(len(TARGET_FIXED_COLUMNS)), TARGET_FIXED_COLUMNS):
            options = sorted(target_rows[column].dropna().unique().tolist(), key=str)
            value = col.selectbox(DIMENSIONS.get(column, 'Top-k'), [ANY] + options, key=f'target_{column}',
                                  help=f'"{ANY}": the best value for the target.')
            if value != ANY:
                target_rows = target_rows[target_rows[column] == value]
        if target_rows.empty:
            st.info(f'No {target} setting has `{select_by}` with these values.')
        else:
            target_index = target_rows[select_by].idxmax()
            keys = configuration_keys(ranked, match_on)
            same = (keys == keys.loc[target_index]).all(axis=1)
            matched = ranked[same]
            best_matched = best_per_group(matched, 'model_folder', select_by).copy()
            best_matched['role'] = ['target' if i == target_index else 'same configuration' for i in best_matched.index]
            # Target first, so that every other row can be checked against it
            best_matched = pd.concat([best_matched.loc[[target_index]],
                                      best_matched.drop(index=target_index).sort_values(select_by, ascending=False)])

            st.caption(f'Best `{target}` configuration by **{select_by}** (first row), and the setting each other '
                       'model is compared with:')
            table_columns = ['model_family', 'model_folder', 'role', *config_columns, select_by, *measure_columns,
                             *DETECTION_COLUMNS, 'nab_tp', 'nab_fp', 'nab_fn', 'file']
            st.dataframe(best_matched[list(dict.fromkeys(table_columns))], hide_index=True,
                         column_config={**measure_column_config(measures), **DETECTION_COLUMN_CONFIG})
            missing = sorted(set(ranked['model_folder']) - set(best_matched['model_folder']))
            if missing:
                st.warning(f'No setting with this configuration for: {", ".join(missing)}.')

            st.caption('Bar: each measure of every model with this configuration (target outlined in black); '
                       'tick: the best value of the measure for the model over all its settings.')
            # One row per (model, measure), with the best value of the measure for the model over all its settings
            own_best = filtered.groupby('model_folder')[measure_columns].max()
            long = best_matched.melt(id_vars=['model_folder', 'model_family', 'role'], value_vars=measure_columns,
                                     var_name='column', value_name='value')
            long['own_best'] = [own_best.at[f, c] for f, c in zip(long['model_folder'], long['column'])]
            long['gap_to_own_best'] = long['value'] - long['own_best']
            long['measure'] = long['column'].map({MEASURES[m][0]: m for m in measures})
            family_tooltip = ['model_folder', 'role', 'measure', alt.Tooltip('value:Q', format='.4f'),
                              alt.Tooltip('own_best:Q', format='.4f'), alt.Tooltip('gap_to_own_best:Q', format='.4f')]
            charts = []
            for m in measures:   # one chart per measure, side by side, models by decreasing value
                data = long[long['measure'] == m].sort_values('value', ascending=False)
                x = alt.X('model_folder:N', sort=data['model_folder'].tolist(), title=None,
                          # Long model names overlap once rotated: Vega would hide every other one
                          axis=alt.Axis(labelAngle=-45, labelLimit=0, labelOverlap=False))
                bars = alt.Chart(data).mark_bar(width=28).encode(
                    x=x, y=alt.Y('value:Q', title=None),
                    color=alt.Color('model_family:N', title='Model family', scale=family_scale,
                                    legend=alt.Legend(orient='top', direction='horizontal')),
                    stroke=alt.condition(alt.datum.role == 'target', alt.value('black'), alt.value(None)),
                    strokeWidth=alt.value(2), tooltip=family_tooltip)
                ticks = alt.Chart(data).mark_tick(color='black', thickness=2.5, size=34).encode(
                    x=x, y='own_best:Q', tooltip=family_tooltip)
                charts.append((bars + ticks).properties(title=m))
            chart = side_by_side(charts, len(best_matched), charts_per_row, px_per_group=80)
            st.altair_chart(chart)

    # ── Strategies by window: the post-processing strategies of the chosen models along the window size ──
    with tab_windows:
        models = sorted(filtered['model_folder'].unique())
        col_model, col_split, col_value, col_layout = st.columns([3, 2, 2, 2])
        chosen = col_model.multiselect('Models', models, key='windows_models',
                                       default=[m for m in models if model_family(m) == 'A3TGCN'][:1] or models[:1])
        split_by = col_split.multiselect(
            'Strategies split by', ['threshold_type', 'score_normalization', 'fill_nan'], default=['threshold_type'],
            format_func=DIMENSIONS.get, key='windows_split',
            help='Each combination of the strategy and these dimensions is one line.')
        statistic = col_value.segmented_control(
            'Value at each window', ['Target configuration', 'Best of each model', 'Median over settings'],
            default='Target configuration', required=True, key='windows_statistic',
            help=f'Target configuration: at each window and strategy, the best configuration of the target model by '
                 f'`{select_by}` (sidebar), and for every other model its best setting with the same configuration; '
                 f'best of each model: every model its own setting with the best `{select_by}` among its settings of '
                 'the strategy at the window (any subset, fill NaN, threshold…); median: median of each test measure '
                 'over these settings, band from the first to the third quartile.')
        layout = col_layout.segmented_control(
            'Layout', ['One row per model', 'Models together'], default='One row per model', required=True,
            key='windows_layout',
            help='One row per model: a row of charts (one per measure) for every model, same value scale for a '
                 'measure in every row; models together: one chart per measure, line color = strategy, '
                 'line dash = model.')
        strategy_keys = ['post_processing_strategy', *split_by]
        by_target = statistic == 'Target configuration'
        if by_target:
            col_target, col_match = st.columns([1, 3])
            target = col_target.selectbox(
                'Target model', models, key='windows_target',
                index=next((i for i, m in enumerate(models) if model_family(m) == 'A3TGCN'), 0),
                help='Shown with the chosen models, even when not among them.')
            # The window and the strategy are those of each point of a line already
            match_options = [c for c in CONFIG_COLUMNS if c in filtered and c not in ('window', *strategy_keys)]
            window_match_on = col_match.multiselect(
                'Other models use the same', match_options, default=match_options, key='windows_match_on',
                format_func=lambda c: DIMENSIONS.get(c, c),
                help='Values taken from the best configuration of the target model at the window and strategy. '
                     f'Dimensions left out are free: every model takes its best value of them by {select_by}.')
            chosen = [target] + [m for m in chosen if m != target]
        rows = filtered[filtered['model_folder'].isin(chosen)]
        if statistic != 'Median over settings':
            rows = rows.dropna(subset=[select_by])
        rows = rows.assign(strategy=join_columns(rows, strategy_keys) if len(rows) else '')
        if not chosen:
            st.info('Choose at least one model.')
        elif rows.empty:
            st.info('No setting of these models matches the filters' +
                    (f' with `{select_by}`.' if statistic != 'Median over settings' else '.'))
        else:
            keys = ['model_folder', 'model_family', 'window', 'strategy', *strategy_keys]
            unmatched = []
            if by_target:
                config = configuration_keys(rows, window_match_on)
                target_best = rows[rows['model_folder'] == target].groupby(['window', 'strategy'])[select_by].idxmax()
                picked = []
                for (window, strategy), target_index in target_best.items():
                    candidates = rows[(rows['window'] == window) & (rows['strategy'] == strategy)]
                    same = (config.loc[candidates.index] == config.loc[target_index]).all(axis=1)
                    matched = candidates[same]
                    picked.append(best_per_group(matched, 'model_folder', select_by))
                    matched_models = set(matched['model_folder'])
                    unmatched += [(m, window, strategy) for m in chosen if m not in matched_models]
                per_window = (pd.concat(picked) if picked else rows.iloc[:0]).copy()
                per_window['role'] = ['target' if m == target else 'same configuration'
                                      for m in per_window['model_folder']]
                long = per_window.melt(id_vars=keys, value_vars=measure_columns, var_name='column', value_name='value')
            elif statistic == 'Best of each model':
                per_window = best_per_group(rows, keys, select_by)
                long = per_window.melt(id_vars=keys, value_vars=measure_columns, var_name='column', value_name='value')
            else:
                long = rows.melt(id_vars=keys, value_vars=measure_columns, var_name='column', value_name='value')
                long = (long.groupby([*keys, 'column'])['value']
                        .agg(value='median', q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75),
                             settings='size').reset_index())
                per_window = long
            long['measure'] = long['column'].map({MEASURES[m][0]: m for m in measures})
            windows = sorted(rows['window'].unique(), key=float)
            long['window'] = long['window'].astype(str)
            x = alt.X('window:O', sort=[str(w) for w in windows], title='Window size', axis=alt.Axis(labelAngle=0))
            strategies = sorted(long['strategy'].unique())
            shown_models = [m for m in chosen if m in set(long['model_folder'])]
            together = layout == 'Models together'
            legend = dict(orient='top', direction='vertical', labelLimit=0)
            # As in the other tabs: color = model family; the strategies by their line dash and point shape
            color = alt.Color('model_family:N', title='Model family', scale=family_scale, legend=alt.Legend(**legend))
            strategy_channels = {
                'strokeDash': alt.StrokeDash('strategy:N', title='Strategy', scale=alt.Scale(
                    domain=strategies, range=_cycled(BORDER_DASHES, len(strategies))),
                    legend=alt.Legend(**legend, symbolStrokeColor='black', symbolType='stroke', symbolSize=400)),
                'shape': alt.Shape('strategy:N', title='Strategy', scale=alt.Scale(
                    domain=strategies, range=_cycled(STRATEGY_SHAPES, len(strategies))),
                    legend=alt.Legend(**legend, symbolFillColor='black')),
            }
            # Models of the same family (shown together) by their shade, as the bars of the other tabs
            shade, shade_legend = {}, None
            shown_families = pd.Series([model_family(m) for m in shown_models])
            if together and shown_families.duplicated().any():
                # Shade by the rank of the model within its family: the first model of every family is plain
                rank = shown_families.groupby(shown_families).cumcount()
                shade_scale = alt.Scale(domain=shown_models, range=[LINE_SHADES[r % len(LINE_SHADES)] for r in rank])
                shade['opacity'] = alt.Opacity('model_folder:N', scale=shade_scale, legend=None)
                colors = family_colors(family_scale)
                shade_legend = dict(
                    opacity=alt.Opacity('model_folder:N', title='Model', scale=shade_scale, legend=alt.Legend(
                        **legend, symbolType='square', symbolSize=150, symbolStrokeColor='black', symbolStrokeWidth=1)),
                    fill=alt.Fill('model_folder:N', title='Model', scale=alt.Scale(
                        domain=shown_models,
                        range=[colors.get(model_family(m), NO_FAMILY_COLOR) for m in shown_models])))

            def window_chart(data, m, title):
                """Lines of the strategies (of one model, or of every model when together) for measure m."""
                _, axis_title, fmt = MEASURES[m]
                # Same value scale for a measure in every row of models
                values = long.loc[long['measure'] == m, ['value', *(['q1', 'q3'] if 'q1' in long else [])]]
                low, high = values.min().min(), values.max().max()
                # No value of the measure (e.g. a column empty for these models): Vega-Lite rejects a NaN domain
                y = alt.Y('value:Q', title=axis_title, scale=alt.Scale(
                    zero=False, **({} if pd.isna(low) or pd.isna(high) else {'domain': [float(low), float(high)]})))
                tooltip = ['model_folder', 'strategy', 'window', alt.Tooltip('value:Q', title=m, format=fmt)]
                if statistic == 'Median over settings':
                    tooltip += [alt.Tooltip('q1:Q', format=fmt), alt.Tooltip('q3:Q', format=fmt), 'settings']
                base = alt.Chart(data)
                detail = alt.Detail(['model_folder:N', 'strategy:N'])
                layers = [
                    base.mark_line(strokeWidth=2).encode(x=x, y=y, color=color, detail=detail, tooltip=tooltip,
                                                         strokeDash=strategy_channels['strokeDash'], **shade),
                    base.mark_point(filled=True, size=70).encode(x=x, y=y, color=color, tooltip=tooltip,
                                                                 shape=strategy_channels['shape'], **shade)]
                if statistic == 'Median over settings':
                    layers.insert(0, base.mark_area(opacity=0.12).encode(x=x, y='q1:Q', y2='q3:Q', color=color,
                                                                         detail=detail))
                if shade_legend:   # Vega-Lite draws the shade legend in grey: invisible points in the family colors
                    layers.append(base.mark_point(size=0, filled=True).encode(x=x, y=y, **shade_legend))
                return alt.layer(*layers).properties(title=title, width=max(260, 70 * len(windows)), height=280)

            st.caption('One line per strategy' + (' and model' if together else '') + ': ' + {
                'Target configuration': f'at each window, the best configuration of **{target if by_target else ""}** '
                                        f'by `{select_by}`, and each other model\'s best setting with the same '
                                        'configuration.',
                'Best of each model': f'the setting with the best `{select_by}` at each window.',
                'Median over settings': 'median over the settings at each window (band: quartiles).'}[statistic])
            if together:
                charts = [window_chart(long[long['measure'] == m], m, m) for m in measures]
                st.altair_chart(alt.concat(*charts, columns=charts_per_row).resolve_scale(y='independent'))
            else:
                for model in shown_models:
                    name = f'{model} (target)' if by_target and model == target else model
                    charts = [window_chart(long[(long['measure'] == m) & (long['model_folder'] == model)], m,
                                           f'{name} · {m}') for m in measures]
                    st.altair_chart(alt.concat(*charts, columns=charts_per_row).resolve_scale(y='independent'))
            missing_models = sorted(set(chosen) - set(shown_models))
            if missing_models:
                st.warning(f'No setting matches the filters' + (' and the target configurations' if by_target else '')
                           + f' for: {", ".join(missing_models)}.')
            unmatched = [u for u in unmatched if u[0] not in missing_models]
            if unmatched:
                with st.expander(f'{len(unmatched)} points without the target configuration (gaps in the lines)',
                                 icon=':material/warning:'):
                    st.dataframe(pd.DataFrame(unmatched, columns=['model_folder', 'window', 'strategy']), hide_index=True)
            # Windows a strategy has no setting at: a gap in its line
            counts = rows.groupby(['model_folder', 'strategy', 'window']).size().unstack('window').reindex(columns=windows)
            counts.columns = [str(w) for w in counts.columns]
            st.caption('Settings of each model and strategy at each window (empty: not run, or filtered out):')
            st.dataframe(counts, column_config={str(w): st.column_config.NumberColumn(format='%d') for w in windows})
            if statistic != 'Median over settings':
                shown = ['model_folder', *(['role'] if by_target else []), 'window', 'strategy',
                         *[c for c in CONFIG_COLUMNS if c not in ('window', *strategy_keys)],
                         select_by, *measure_columns, 'nab_tp', 'nab_fp', 'nab_fn', 'detected', 'file']
                shown = [c for c in dict.fromkeys(shown) if c in per_window.columns]
                # Each point of the target first, followed by the other models
                order = per_window.assign(not_target=per_window['model_folder'] != (target if by_target else None))
                st.dataframe(order.sort_values(['strategy', 'window', 'not_target', 'model_folder'])[shown],
                             hide_index=True,
                             column_config={**measure_column_config(measures), **DETECTION_COLUMN_CONFIG})

    # ── All settings ──
    with tab_rows:
        leading = list(DIMENSIONS) + ['topk', 'long_window', 'short_window', 'anomaly_threshold',
                                      'reward_fn_normalized', 'standard_normalized', *DETECTION_COLUMNS,
                                      'nab_tp', 'nab_fp', 'nab_fn', 'precision', 'recall', 'f1']
        leading = [c for c in leading if c in filtered.columns]
        columns = leading + [c for c in filtered.columns if c not in leading and c not in ('detection_counters',)]
        # Only the best MAX_SHOWN_ROWS rows are sent to the browser: the whole table (millions of rows) exceeded
        # the 2 GB message limit of Streamlit. Sorting the one measure column picks them (NaN last, as before)
        order = filtered[measure_column].sort_values(ascending=False).index
        st.caption(f'{len(filtered):,} settings match the filters (sorted by {measure_title})' +
                   (f'; the best {MAX_SHOWN_ROWS:,} are shown, all of them are in the CSV.'
                    if len(filtered) > MAX_SHOWN_ROWS else '.'))
        st.dataframe(filtered.loc[order[:MAX_SHOWN_ROWS], columns], hide_index=True, height=600,
                     column_config=DETECTION_COLUMN_CONFIG)
        # The CSV of up to millions of rows is written on click only, not on every rerun
        st.download_button('Download as CSV', lambda: filtered.loc[order, columns].to_csv(index=False).encode(),
                           file_name='grid_search_filtered.csv', mime='text/csv')


TABLE_STATISTICS = ['Best setting', 'Median', 'Mean', 'Max']
TABLE_LAYOUTS = ['Measures under each column value', 'Column values under each measure', 'Measures as rows']
TABLE_HIGHLIGHT = {'Per column': 'the two best models / groups of every column',
                   'Per row': 'the two best columns of every row, for each measure', 'None': ''}
# Top-1 / top-2 cells of the results table: darker / lighter shade (LaTeX: bold / underlined)
RANK_CELL_STYLES = {0: '', 1: 'font-weight: bold; background-color: rgba(76, 120, 168, 0.45)',
                    2: 'background-color: rgba(76, 120, 168, 0.18)'}
LATEX_RANK_FORMATS = {0: '%s', 1: '\\textbf{%s}', 2: '\\underline{%s}'}
HIGHLIGHT_AXES = {'Per column': 0, 'Per row': 1}   # axis of the top-1 / top-2 ranks of a highlight mode
ROW_LABELS = {**DIMENSIONS, 'measure': 'Measure'}   # headers of the row columns of the results table
# Static results table with merged cells (st.html): readable in the light and dark themes, scrolls when wide
MERGED_TABLE_CSS = '''<style>
div.merged-table { overflow-x: auto; }
table.merged-table { border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
table.merged-table th, table.merged-table td { border: 1px solid rgba(128, 128, 128, 0.35); padding: 4px 10px; }
table.merged-table thead th { text-align: center; }
table.merged-table tbody th { text-align: left; vertical-align: middle; font-weight: 600; }
table.merged-table td { text-align: right; }
</style>'''


def _latex_text(value):
    return str(value).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


PANEL_TITLE_SIZE = AXIS_LABEL_FONT_SIZE + FONT_SIZE_INCREASE   # the measure on the left of the panels


def panel_width(columns, axis_space=PANEL_AXIS_SPACE):
    """Width of the panels of a figure FIGURE_WIDTH wide with `columns` panels per row, the measure on their left
    and a y axis axis_space wide (estimated) on the left of each."""
    return ((FIGURE_WIDTH - PANEL_TITLE_SIZE - PANEL_TITLE_OFFSET - PANEL_SPACING * (columns - 1)) / columns
            - axis_space)


def paper_style(chart):
    """The text sizes (src/figure_style.py, plus FONT_SIZE_INCREASE) and frames of a figure of the paper."""
    return (chart.configure_view(stroke=CHART_BORDER, strokeWidth=1)   # a border around each plot
            .configure_axis(labelFontSize=TICK_FONT_SIZE + FONT_SIZE_INCREASE,
                            titleFontSize=AXIS_LABEL_FONT_SIZE + FONT_SIZE_INCREASE)
            .configure_axisY(labelFontSize=TICK_FONT_SIZE + FONT_SIZE_INCREASE - Y_TICK_FONT_SIZE_DECREASE)
            .configure_legend(labelFontSize=LEGEND_FONT_SIZE + FONT_SIZE_INCREASE,
                              titleFontSize=LEGEND_FONT_SIZE + FONT_SIZE_INCREASE,
                              strokeColor=CHART_BORDER, padding=LEGEND_PADDING)   # a frame around each legend
            .configure_header(labelFontSize=AXIS_LABEL_FONT_SIZE + FONT_SIZE_INCREASE,   # the panel titles
                              labelPadding=2))


def export_buttons(chart, name, key):
    """PNG and PDF downloads of a chart, made only when clicked."""
    col_png, col_pdf, _ = st.columns([1, 1, 3])
    col_png.download_button('Export as PNG', lambda: chart_png(chart), file_name=f'{name}.png', mime='image/png',
                            key=f'png_{key}', icon=':material/image:', on_click='ignore')
    col_pdf.download_button('Export as PDF', lambda: chart_pdf(chart), file_name=f'{name}.pdf',
                            mime='application/pdf', key=f'pdf_{key}', icon=':material/picture_as_pdf:',
                            on_click='ignore', help='Vector figure (sharp at any zoom), for LaTeX')


def chart_pdf(chart):
    """PDF bytes of an Altair chart: vector, its units in points, so FIGURE_WIDTH_INCHES wide as it is."""
    buffer = io.BytesIO()
    chart.save(buffer, format='pdf')
    return buffer.getvalue()


def chart_png(chart):
    """PNG bytes of an Altair chart at PNG_DPI, its resolution set so that it is FIGURE_WIDTH_INCHES wide."""
    buffer = io.BytesIO()
    chart.save(buffer, format='png', scale_factor=PNG_DPI / POINTS_PER_INCH)
    image = Image.open(io.BytesIO(buffer.getvalue()))
    dpi = image.width / FIGURE_WIDTH_INCHES   # PNG_DPI, up to the width of the panel axes (estimated)
    buffer = io.BytesIO()
    image.save(buffer, format='png', dpi=(dpi, dpi))
    return buffer.getvalue()


def family_legend_layout(series, families):
    """
    Legend entries (domain) of series laid out in columns: one per model family with several series, the other
    series together in the first one. Vega-Lite fills a legend row by row: the columns are laid out with blank
    entries (no label, no symbol, whatever the scales give them). Returns (domain, number of columns).
    """
    family_series = pd.Series(list(series)).groupby(list(families), sort=False).agg(list)
    singles = [s for f in family_series if len(f) == 1 for s in f]
    columns = [singles] * bool(singles) + [f for f in family_series if len(f) > 1]
    n_rows = max(map(len, columns))
    blanks = iter('\u200b' * (i + 1) for i in range(n_rows * len(columns)))
    return [column[row] if row < len(column) else next(blanks) for row in range(n_rows) for column in columns], \
        len(columns)


def window_line_charts(values, measures, series_dims, facet_dims, family_scale, short_names, overlay_dims=()):
    """
    Line charts of the results table along the window size, one per measure: one line per series (the rows of
    the table, e.g. a model, and the other column dimensions), in its model family's color; ClouDens models are
    solid with the sparsity indicator (*) and dashed without it, the other models solid. One panel per group of
    rows of facet_dims (Group rows by, e.g. the subset), or the groups of overlay_dims all in the same chart, told
    apart by their point shape. The series that still share a color and a dash are told apart by their point
    shape (one panel per group) or by the shade of their color (groups in one chart).
    values: the cells of the table, one row per (row, column) key, one column per measure.
    """
    overlay_dims = list(overlay_dims)
    data = values.reset_index()
    order_key = data['model_folder'].map(model_order_key) if 'model_folder' in data else pd.Series(0, index=data.index)
    family = data['model_folder'].map(model_family) if 'model_folder' in data else pd.Series('', index=data.index)
    dashed = (data['model_folder'].map(lambda f: short_model_name(f).startswith(PROPOSED_MODEL_PREFIX)
                                       and not NULL_PADDING_SUFFIX.search(str(f)))
              if 'model_folder' in data else pd.Series(False, index=data.index))
    if short_names and 'model_folder' in data:
        data['model_folder'] = data['model_folder'].map(short_model_name)
    if short_names and 'subset' in data:
        data['subset'] = data['subset'].map(short_subset_name)
    data['series'] = join_columns(data, series_dims) if series_dims else 'value'
    data['panel'] = join_columns(data, facet_dims) if facet_dims else ''
    data['group'] = join_columns(data, overlay_dims) if overlay_dims else ''
    data['line'] = data['series'] + ' · ' + data['group']
    data['window'] = data['window'].astype(str)
    windows = sorted(data['window'].unique(), key=float)
    # Series in the order of the table, each in its family's color and its model's dash
    first = data.assign(_order=order_key, _family=family, _dashed=dashed).sort_values(['_order', 'series'],
                                                                                      kind='stable')
    first = first.drop_duplicates('series')
    colors = family_colors(family_scale)
    series = first['series'].tolist()
    rank = first.groupby(['_family', '_dashed']).cumcount()
    legend_domain, legend_columns = family_legend_layout(first['series'], first['_family'])
    color_of = dict(zip(series, [colors.get(f, NO_FAMILY_COLOR) for f in first['_family']]))
    dash_of = dict(zip(series, [LINE_DASHES[d] for d in first['_dashed']]))
    shape_of = dict(zip(series, [STRATEGY_SHAPES[r % len(STRATEGY_SHAPES)] for r in rank]))
    color_scale = alt.Scale(domain=legend_domain, range=[color_of.get(s, 'transparent') for s in legend_domain])
    dash_scale = alt.Scale(domain=legend_domain, range=[dash_of.get(s, LINE_DASHES[False]) for s in legend_domain])
    panels = list(dict.fromkeys(data['panel']))
    groups = list(dict.fromkeys(data['group']))
    long = data.melt(id_vars=['series', 'panel', 'group', 'line', 'window'], value_vars=measures, var_name='measure')
    # Legends compact (e.g. for a figure of a paper): the series above the chart, the groups drawn in one chart
    # (point shapes) below it
    legend = alt.Legend(orient='top', direction='horizontal', columns=legend_columns, labelLimit=0,
                        symbolStrokeWidth=1.5, symbolSize=40, symbolOpacity=1, columnPadding=8, rowPadding=1,
                        titleOrient='left', titleAnchor='start', titleBaseline='top',   # title in the top left corner
                        offset=LEGEND_OFFSET)
    title = ' · '.join(DIMENSIONS[d] for d in series_dims) or None
    if overlay_dims:
        # The shape tells apart the groups, the same for every series; the shade the series of a family and dash
        shape = alt.Shape('group:N', scale=alt.Scale(domain=groups, range=_cycled(GROUP_SHAPES, len(groups))),
                          title=' · '.join(DIMENSIONS[d] for d in overlay_dims),
                          legend=alt.Legend(orient='bottom', direction='horizontal', columns=4, labelLimit=0,
                                            symbolSize=40, columnPadding=8, rowPadding=1, titleOrient='left',
                                            titleAnchor='start', titleBaseline='top', offset=LEGEND_OFFSET))
        shade = alt.Opacity('series:N', scale=alt.Scale(domain=series, range=[LINE_SHADES[r % len(LINE_SHADES)]
                                                                               for r in rank]), legend=None)
        tooltip = ['series', 'group', 'window']
    else:
        shape = alt.Shape('series:N', scale=alt.Scale(domain=legend_domain,
                                                      range=[shape_of.get(s, 'circle') for s in legend_domain]),
                          title=title, legend=legend)
        shade = alt.value(1)
        tooltip = ['series', 'panel', 'window']
    charts = []
    for m in measures:
        fmt = MEASURES[m][2]
        encodings = dict(
            x=alt.X('window:O', sort=windows, title='Window size', axis=alt.Axis(labelAngle=0, domain=False, tickSize=3, labelPadding=2)),
            # padding: the top / bottom grid lines inside the border instead of over it
            # panels: the measure once on the left of all of them (the chart title) instead of on every y axis
            y=alt.Y('value:Q', title=None if facet_dims else m, scale=alt.Scale(zero=False, nice=False, padding=8),
                    axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),   # the border as axis line
            color=alt.Color('series:N', scale=color_scale, title=title, legend=legend),
            opacity=shade,
            tooltip=[*tooltip, alt.Tooltip('value:Q', title=m, format=fmt)])
        layers = alt.layer(
            alt.Chart().mark_line(strokeWidth=1.5).encode(
                strokeDash=alt.StrokeDash('series:N', scale=dash_scale, title=title, legend=legend),
                detail='line:N', **encodings),
            alt.Chart().mark_point(filled=True, size=20).encode(shape=shape, **encodings),
            data=long[long['measure'] == m])
        # FIGURE_WIDTH wide in all: a single chart fits it exactly, the panels share it (their axes estimated)
        if facet_dims:
            columns = min(PANEL_COLUMNS, len(panels))
            width = panel_width(columns)
            chart = layers.properties(width=width, height=width * PANEL_ASPECT).facet(
                facet=alt.Facet('panel:N', sort=panels, title=None), columns=columns, spacing=PANEL_SPACING
            ).resolve_scale(y='independent').properties(   # each panel its own y range
                title=alt.TitleParams(m, orient='left', anchor='middle', fontSize=PANEL_TITLE_SIZE,
                                      fontWeight='bold', offset=PANEL_TITLE_OFFSET))
        else:
            chart = layers.properties(width=FIGURE_WIDTH, height=FIGURE_WIDTH * CHART_ASPECT,
                                      autosize=alt.AutoSizeParams(type='fit-x', contains='padding'))
        charts.append(paper_style(chart))
    return charts


def latex_tabular(row_headers, header, cells, rule_before=()):
    """
    booktabs tabular: row_headers (left columns), header the (outer, inner) labels of the value columns, cells
    the rows (LaTeX text), with a rule before the rows rule_before. Consecutive equal outer labels span their
    columns (multicolumn and cmidrule).
    """
    n_rows = len(row_headers)
    lines = ['\\begin{tabular}{' + 'l' * n_rows + 'r' * len(header) + '}', '\\toprule']
    if any(outer is not None for outer, _ in header):
        spans, rules, start = [], [], n_rows + 1
        for outer, group in itertools.groupby(o for o, _ in header):
            width = len(list(group))
            spans.append(f'\\multicolumn{{{width}}}{{c}}{{{outer or ""}}}' if width > 1 else (outer or ''))
            if outer is not None:
                rules.append(f'\\cmidrule(lr){{{start}-{start + width - 1}}}')
            start += width
        lines += [' & '.join([''] * n_rows + spans) + ' \\\\', ' '.join(rules)]
    lines += [' & '.join(row_headers + [inner for _, inner in header]) + ' \\\\', '\\midrule']
    for i, row in enumerate(cells):
        if i in rule_before:   # e.g. between two groups of rows
            lines.append('\\midrule')
        lines.append(' & '.join(map(str, row)) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def render_results_table(results, family_scale, options):
    """
    'Results table' tab: one cell per group of rows (e.g. models) and of columns (e.g. windows), with the
    measures of the best setting of the group or a statistic over its settings. Its filters are its own:
    the sidebar filters do not apply.
    """
    st.caption('This tab has its own filters: the filters of the sidebar do not apply here.')
    with st.expander('Filters of the table', expanded=True, icon=':material/filter_alt:'):
        filter_columns = st.columns(4)
        st.caption('Empty filter: all its values. Choose at least one filter to build the table.')
        chosen = dimension_filters(lambda i: filter_columns[i % 4], options, key_prefix='table_filter_')
    if not any(chosen.values()):   # nothing chosen: the table of the whole results would be heavy
        st.info(f'Choose at least one filter of the table (e.g. a subset and a window size): '
                f'{len(results):,} grid-search settings in all.')
        return
    rows = filter_rows(results, options, chosen)

    col_rows, col_columns, col_measures = st.columns(3)
    row_dims = col_rows.multiselect('Rows', list(DIMENSIONS), default=['model_folder'], format_func=DIMENSIONS.get,
                                    key='table_rows')
    column_dims = col_columns.multiselect('Columns', list(DIMENSIONS), default=['window'], format_func=DIMENSIONS.get,
                                          key='table_columns', help='Each measure is split into one column per value.')
    column_dims = [d for d in column_dims if d not in row_dims]
    available_measures = [m for m, (c, _, _) in MEASURES.items() if c in results and results[c].notna().any()]
    measures = col_measures.multiselect('Measures (test)', available_measures, key='table_measures',
                                        default=[m for m in ('F1', 'NAB standard', 'NAB reward_fn')
                                                 if m in available_measures])
    col_statistic, col_select, col_highlight = st.columns(3)
    statistic = col_statistic.segmented_control(
        'Value of a cell', TABLE_STATISTICS, default='Best setting', required=True, key='table_statistic',
        help='Best setting: the measures of the setting with the best selection metric in the cell (one setting '
             'for all the measures). Median / Mean / Max: over the settings of the cell, measure by measure.')
    available_metrics = [m for m in SELECTION_METRICS if m in results.columns]
    select_by = col_select.selectbox(
        'Best setting by', available_metrics, key='table_select_by', disabled=statistic != 'Best setting',
        index=available_metrics.index('train_valid_f1') if 'train_valid_f1' in available_metrics else 0,
        help='train_valid_* / valid_* select without the test labels; f1 / NAB select on the test split itself.')
    highlight = col_highlight.segmented_control('Highlight top 1 / top 2', list(TABLE_HIGHLIGHT), default='Per column',
                                                required=True, key='table_highlight')
    layout = st.segmented_control(
        'Layout', TABLE_LAYOUTS, default=TABLE_LAYOUTS[0], required=True, key='table_layout',
        help='Measures under each column value: e.g. Window 6 · F1, Window 6 · NAB, Window 12 · F1, … '
             'Column values under each measure: F1 · Window 6, F1 · Window 12, …, NAB · Window 6, … '
             'Measures as rows: one row per model and measure, one column per window.')

    group_dims = st.multiselect(
        'Group rows by', list(DIMENSIONS), default=['subset'], format_func=DIMENSIONS.get, key='table_group',
        help='Leading columns of the table, e.g. the subset: its rows (models) are listed under every value and '
             'the best cells are those of each group. Dimensions already in the rows or columns are ignored.')
    group_dims = [d for d in group_dims if d not in row_dims + column_dims]
    row_dims = group_dims + row_dims
    short_names = st.toggle(
        'Short names', value=True, key='table_short_names',
        help="Subsets: 'no_group_5xx_count' → '5xx count'. Models (*: with the sparsity indicator, null padding): "
             "'A3TGCN' / 'A3TGCN_null_padding_feature' → 'ClouDens-A3T' / 'ClouDens-A3T*', "
             "'BigST_spatial' / 'BigST_null_padding_feature_spatial' (context graph) → 'ClouDens-BigST-ctx' / "
             "'ClouDens-BigST-ctx*', 'BigST' / 'BigST_null_padding_feature' → 'ClouDens-BigST' / 'ClouDens-BigST*', "
             "'STGformer_null_padding_feature_poolwin' "
             "→ 'STGformer*'.")
    merge_cells = st.toggle(
        'Merge equal cells', value=True, key='table_merge', disabled=not group_dims,
        help='One merged cell for each group (e.g. a subset over its models), as in the LaTeX table: a static '
             'table. Off: the interactive table (sortable, one value per row).') and bool(group_dims)
    if not row_dims or not measures:
        st.info('Choose at least one dimension for the rows and one measure.')
        return
    keys = row_dims + column_dims
    measure_columns = [MEASURES[m][0] for m in measures]
    if statistic == 'Best setting':
        rows = rows.dropna(subset=[select_by])
    if rows.empty:
        st.info('No grid-search row matches the filters of the table' +
                (f' with `{select_by}`.' if statistic == 'Best setting' else '.'))
        return

    def top_ranks(frame, axis):
        """1 / 2 for the highest / second highest values (ties share a rank) of every column (axis 0, within
        each group of rows of Group rows by) or of every row (axis 1) of frame; 0 elsewhere."""
        if axis == 0 and group_dims:
            ranks = frame.groupby(level=group_dims).rank(method='dense', ascending=False)
        else:
            ranks = frame.rank(axis=axis, method='dense', ascending=False)
        return ranks.where(ranks <= 2, 0).astype(int)   # NaN <= 2 is False: missing values get 0 too

    if statistic == 'Best setting':
        picked = best_per_group(rows, keys, select_by)
        values = picked.set_index(keys)[measure_columns]
    else:
        values = rows.groupby(keys)[measure_columns].agg(statistic.lower())
    values.columns = measures

    # Value columns in the order of the layout, each with its (outer, inner) header labels: the outer label
    # (e.g. 'w=6') groups the columns under it in the LaTeX header; best cells within each measure
    wide = values.unstack(column_dims) if column_dims else values
    wide.columns = pd.MultiIndex.from_tuples([(c[0], tuple(c[1:])) for c in wide.columns] if column_dims
                                             else [(m, ()) for m in wide.columns])
    col_values = list(dict.fromkeys(c[1] for c in wide.columns)) if column_dims else [()]

    def value_label(value):
        return ' · '.join((f'w={v}' if d == 'window' else str(v)) for d, v in zip(column_dims, value))

    axis = HIGHLIGHT_AXES.get(highlight)   # None: no highlight
    if layout == 'Measures as rows':
        # One row per (row group, measure), one column per value of the column dimensions
        labels = [value_label(v) or 'Value' for v in col_values]
        frames = []
        for m in measures:
            frame = wide[[(m, v) for v in col_values]].set_axis(labels, axis=1)
            frames.append(frame.assign(measure=m).set_index('measure', append=True))
        wide = pd.concat(frames).sort_index(level=list(range(len(row_dims))), sort_remaining=False, kind='stable')
        header = [(None, label) for label in labels]
        measure_index = wide.index.get_level_values('measure')
        best = pd.DataFrame(0, index=wide.index, columns=labels)
        if axis is not None:
            for m in measures:
                best[measure_index == m] = top_ranks(wide[measure_index == m], axis)
        row_columns = row_dims + ['measure']
    else:
        if layout == 'Measures under each column value':
            order = [(m, v) for v in col_values for m in measures]
            header = [(value_label(v) or None, m) for m, v in order]
        else:
            order = [(m, v) for m in measures for v in col_values]
            header = [(m, value_label(v)) if column_dims else (None, m) for m, v in order]
        if len(measures) == 1 and column_dims:   # the measure over its column values, e.g. F1 over w=6, w=12, …
            header = [(m, value_label(v)) for m, v in order]
        wide = wide[order]
        wide.columns = [inner if outer is None else f'{outer} · {inner}' for outer, inner in header]
        measure_of_column = dict(zip(wide.columns, (m for m, _ in order)))
        best = pd.DataFrame(0, index=wide.index, columns=wide.columns)
        if axis is not None:
            for m in measures:
                group = [c for c in wide.columns if measure_of_column[c] == m]
                best[group] = top_ranks(wide[group], axis)
        row_columns = row_dims
    value_columns = list(wide.columns)
    table = wide.reset_index()
    best = best.reset_index(drop=True)
    # Model family of every shown model name (its color), before shortening the names
    family_of = {}
    if 'model_folder' in table:
        # Same order of the models in every group: by model, each followed by its null-padding variant
        # ('BigST', 'BigST*', ..., 'ClouDens-A3T', 'ClouDens-A3T*'); the rows keep their order otherwise
        model_order = table['model_folder'].map(model_order_key)
        positions = table.assign(__model_order=model_order).sort_values([*group_dims, '__model_order'],
                                                                        kind='stable').index
        table, best = table.loc[positions].reset_index(drop=True), best.loc[positions].reset_index(drop=True)
        names = {m: short_model_name(m) if short_names else m for m in table['model_folder'].unique()}
        family_of = {name: model_family(m) for m, name in names.items()}
        table['model_folder'] = table['model_folder'].map(names)
    if short_names and 'subset' in table:
        table['subset'] = table['subset'].map(short_subset_name)

    def cell_measure(row, column):
        return table.at[row, 'measure'] if layout == 'Measures as rows' else measure_of_column[column]

    colors = family_colors(family_scale)

    def model_style(model):
        return f'color: {colors.get(family_of[model], NO_FAMILY_COLOR)}; font-weight: 600'

    def styled(frame, ranks, shown_of, row_measures):
        """Styler of the shown table frame: top-1 / top-2 shades from ranks and the number format of every value
        cell; shown_of maps each value column to its column in frame, row_measures the measure of each row."""
        shown = list(shown_of.values())
        styler = frame.style.apply(lambda column: ranks[column.name].map(RANK_CELL_STYLES), subset=shown)
        if layout == 'Measures as rows':
            for m in measures:
                styler = styler.format(measure_format(m), na_rep='–', subset=pd.IndexSlice[row_measures == m, shown])
            return styler
        return styler.format({shown_of[c]: measure_format(measure_of_column[c]) for c in value_columns}, na_rep='–')

    caption = {'Best setting': f'measures of the setting with the best `{select_by}` in each cell',
               'Median': 'median over the settings of each cell', 'Mean': 'mean over the settings of each cell',
               'Max': 'highest value over the settings of each cell (measure by measure)'}[statistic]
    st.caption(f'{len(rows):,} settings, {len(table)} rows: {caption}.' +
               (f' Dark / light: top 1 / top 2, {TABLE_HIGHLIGHT[highlight]}.' if TABLE_HIGHLIGHT[highlight] else '') +
               ' –: no value (e.g. AUC-PR / VUS-PR, computed for the score strategies only).' +
               (' *: graph model with null padding.' if short_names and 'model_folder' in row_dims else ''))
    if merge_cells:
        # Static HTML table: the row columns as a (sparsified) index, so equal consecutive values, e.g. a
        # subset over its models, are one merged cell (st.dataframe has no merged cells)
        indexed = table.set_index(row_columns)
        indexed.index.names = [ROW_LABELS.get(c, c) for c in row_columns]
        best_indexed = best.set_axis(indexed.index)
        # Grouped columns (e.g. w=6 over its measures): two header rows, the outer label one merged cell
        shown_of = dict(zip(value_columns, value_columns))
        if all(outer is not None for outer, _ in header):
            indexed.columns = best_indexed.columns = pd.MultiIndex.from_tuples(header)
            shown_of = dict(zip(value_columns, indexed.columns))
        merged = styled(indexed, best_indexed, shown_of,
                        indexed.index.get_level_values(ROW_LABELS['measure']) if layout == 'Measures as rows' else None)
        if 'model_folder' in row_dims:
            merged = merged.map_index(model_style, level=DIMENSIONS['model_folder'])
        st.html(MERGED_TABLE_CSS + '<div class="merged-table">' +
                merged.set_table_attributes('class="merged-table"').to_html() + '</div>')
    else:
        styler = styled(table, best, dict(zip(value_columns, value_columns)),
                        table['measure'].to_numpy() if layout == 'Measures as rows' else None)
        if 'model_folder' in row_dims:
            styler = styler.map(model_style, subset=['model_folder'])
        st.dataframe(styler, hide_index=True,
                     column_config={c: st.column_config.Column(ROW_LABELS[c]) for c in [*row_dims, 'measure']})

    # Line charts along the window size, one per measure, of the cells of the table
    if 'window' in column_dims:
        st.subheader('By window size', divider='gray')
        series_dims = [d for d in row_dims + column_dims if d not in group_dims and d != 'window']
        panel_dims, overlay_dims = group_dims, []
        if group_dims:
            grouping = st.segmented_control(
                'Groups', CHART_GROUPINGS, default=CHART_GROUPINGS[0], required=True, key='table_chart_groups',
                help='One panel per group of rows, or the lines of every group in the same chart (one point shape '
                     'per group).')
            if grouping == CHART_GROUPINGS[1]:
                panel_dims, overlay_dims = [], group_dims
        charts = window_line_charts(values, measures, series_dims, panel_dims, family_scale, short_names,
                                    overlay_dims)
        for measure, chart in zip(measures, charts):
            st.altair_chart(chart, width='content')   # the widths set in window_line_charts, not the page width
            export_buttons(chart, f'results_by_window_{measure}'.replace(' ', '_'), measure)
    else:
        st.caption('Line charts along the window size: put *Window size* in the Columns.')

    # Exports: the values, and a LaTeX table (two header rows when the columns are grouped; top 1 in bold,
    # top 2 underlined). Subsets in typewriter, as in the paper (\texttt{5xx count})
    cells = [[(f'\\texttt{{{_latex_text(table.at[i, c])}}}' if c == 'subset' else _latex_text(table.at[i, c]))
              for c in row_columns] +
             [(LATEX_RANK_FORMATS[best.at[i, c]] % measure_format(cell_measure(i, c)).format(table.at[i, c])
               if pd.notna(table.at[i, c]) else '--') for c in value_columns] for i in table.index]
    # Groups (Group rows by): their values in one merged cell (multirow) over their rows, a rule between groups.
    # The table has a 0..n-1 index: its labels are the positions of the rows
    group_starts = set()
    if group_dims:
        group_keys = join_columns(table, group_dims)
        first = group_keys.ne(group_keys.shift())
        group_starts = set(table.index[first][1:])
        group_sizes = group_keys.groupby(first.cumsum()).transform('size')
        for i in table.index:
            cells[i][:len(group_dims)] = ([f'\\multirow{{{group_sizes[i]}}}{{*}}{{{c}}}' for c in cells[i][:len(group_dims)]]
                                          if first[i] else [''] * len(group_dims))
    latex = latex_tabular([_latex_text(ROW_LABELS.get(c, c)) for c in row_columns],
                          [(None if o is None else _latex_text(o), _latex_text(i)) for o, i in header], cells,
                          rule_before=group_starts)
    if group_dims:
        latex = '% Merged cells: \\usepackage{multirow}\n' + latex
    col_csv, col_tex = st.columns(2)
    col_csv.download_button('Download as CSV', table.to_csv(index=False).encode(), file_name='results_table.csv',
                            mime='text/csv', icon=':material/download:', width='stretch')
    col_tex.download_button('Download as LaTeX', latex.encode(), file_name='results_table.tex', mime='text/plain',
                            icon=':material/download:', width='stretch')
    with st.expander('LaTeX', icon=':material/code:'):
        st.code(latex, language='latex')
    if statistic == 'Best setting':
        with st.expander('Settings of the cells', icon=':material/table_rows:'):
            shown = [*keys, *[c for c in CONFIG_COLUMNS if c not in keys], select_by, *measure_columns,
                     'nab_tp', 'nab_fp', 'nab_fn', 'detected', 'file']
            shown = [c for c in dict.fromkeys(shown) if c in picked.columns]
            st.dataframe(picked[shown].sort_values(keys), hide_index=True,
                         column_config={**measure_column_config(measures), **DETECTION_COLUMN_CONFIG})



# ── Pages ─────────────────────────────────────────────────────────────────────

def folder_page():
    """Grid-search files scanned under a results folder, refreshed when they change; can be merged."""
    with st.sidebar:
        st.header('Data')
        roots = sorted(p.name for p in PROJECT_ROOT.glob('trained_models*') if p.is_dir()) or ['']
        root_name = st.selectbox('Results folder', roots, index=roots.index('trained_models_log1p')
                                 if 'trained_models_log1p' in roots else 0)
        if st.button('Rescan folder', icon=':material/refresh:', width='stretch'):
            merge_folder.clear()
            load_folder.clear()
        auto_refresh = st.toggle('Auto-refresh', value=True,
                                 help='Reload the app when a grid-search file is added, removed or rewritten.')
        refresh_interval = st.segmented_control('Check every', AUTO_REFRESH_INTERVALS, default='30s', required=True,
                                                disabled=not auto_refresh)

    grid_files = find_grid_files(PROJECT_ROOT / root_name)
    # Files this run shows; the watcher reruns the app when the folder no longer matches
    signature = st.session_state['grid_signature'] = grid_signature(grid_files)

    def watch_results_folder():
        """Fragment rerun on a timer: cheap check of the folder, full rerun only when it changed."""
        if grid_signature(find_grid_files(PROJECT_ROOT / root_name)) != st.session_state.get('grid_signature'):
            st.rerun()   # whole app: reloads the changed files (the cache is keyed by their modification time)
        st.caption(f':material/schedule: Last checked {datetime.now():%H:%M:%S}')

    with st.sidebar:
        st.fragment(watch_results_folder, run_every=refresh_interval if auto_refresh else None)()

    if not grid_files:
        st.warning(f'No `*_grid_search_*.csv` found under `{root_name}`.')
        return
    results = load_folder(signature, LOADING_CODE)

    with st.sidebar:
        st.header('Merge')
        st.caption(f'{len(grid_files)} grid-search files, {len(results):,} rows: one merged file, '
                   "to open in the 'Merged file' page (also on another machine).")
        # The merged table (a cached copy of millions of rows) only when saved or downloaded, not on every rerun
        if st.button('Save to merged_results/', icon=':material/save:', width='stretch'):
            path = write_merged(merge_folder(signature, LOADING_CODE), default_merged_path(PROJECT_ROOT / root_name))
            st.success(f'Saved `{os.path.relpath(path, PROJECT_ROOT)}`')
        st.download_button('Download merged file', lambda: merged_bytes(merge_folder(signature, LOADING_CODE)),
                           icon=':material/download:',
                           file_name=default_merged_path(PROJECT_ROOT / root_name).name,
                           mime='application/gzip', width='stretch')
    render_explorer(results)


def merged_page():
    """A merged file: one of merged_results/ or an uploaded one, e.g. merged on another machine."""
    with st.sidebar:
        st.header('Data')
        source = st.segmented_control('Merged file from', ['merged_results/', 'Upload'], default='merged_results/',
                                      required=True)
        if source == 'Upload':
            uploaded = st.file_uploader('Merged file', type=['gz', 'csv', 'parquet'],
                                        help=f'A file merged by grid_results.py ({", ".join(MERGED_SUFFIXES)}).')
            if uploaded is None:
                st.info('Upload a merged file in the sidebar, e.g. one merged on the HPC server with '
                        '`python streamlit_result_visualization/grid_results.py trained_models_log1p`.')
                return
            load, args, label = load_merged_upload, (uploaded, uploaded.file_id, uploaded.name, LOADING_CODE), uploaded.name
        else:
            files = find_merged_files()
            if not files:
                st.info(f'No merged file in `{os.path.relpath(MERGED_DIR, PROJECT_ROOT)}/`: save one from the '
                        "'Results folder' page, run `grid_results.py`, or upload one.")
                return
            path = st.selectbox('Merged file', files, format_func=lambda p: p.name,
                                help='Most recent first.')
            if st.button('Reload file', icon=':material/refresh:', width='stretch'):
                load_merged_path.clear()
            load, args, label = load_merged_path, (str(path), os.path.getmtime(path), LOADING_CODE), path.name
    try:
        results = load(*args)
    except ValueError as e:
        st.error(f'`{label}`: {e}')
        return
    st.caption(f':material/description: `{label}`: {results["file"].nunique()} grid-search files, '
               f'{len(results):,} rows, last modified {results["modified"].max():%Y-%m-%d %H:%M}')
    render_explorer(results)


st.set_page_config(page_title='Grid-search explorer', page_icon=':material/monitoring:', layout='wide')
st.title('Grid-search results explorer')
st.navigation([
    st.Page(folder_page, title='Results folder', icon=':material/folder_open:', url_path='folder', default=True),
    st.Page(merged_page, title='Merged file', icon=':material/merge:', url_path='merged'),
], position='top').run()
