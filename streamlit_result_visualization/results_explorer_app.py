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
import json
import math
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
                          default_merged_path, component_subsets, COST_KEYS, COST_COLUMNS, COSTS_SUFFIX,
                          read_costs)

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from figure_style import TICK_FONT_SIZE, AXIS_LABEL_FONT_SIZE, LEGEND_FONT_SIZE  # noqa: E402

NAB_COLUMNS = {'reward_fn': 'reward_fn_normalized', 'standard': 'standard_normalized'}
# Names of the measures shown in the app and the figures (column -> name); the NAB profiles as in the paper
# (src/analysis_results.py NAB_PROFILE_DISPLAY_NAME_MAP: reward_fn -> Low FN)
MEASURE_NAMES = {
    'reward_fn_normalized': 'NAB Low FN',
    'standard_normalized': 'NAB Standard',
    'test_auc_pr': 'AUC-PR',
    'test_vus_pr': 'VUS-PR',
    'f1': 'F1',
    'precision': 'Precision',
    'recall': 'Recall',
    'nab_tp': 'Detected anomalies',
}
NAB_LOW_FN, NAB_STANDARD = MEASURE_NAMES['reward_fn_normalized'], MEASURE_NAMES['standard_normalized']
# Test measures the charts can show: name -> (column, axis title, number format)
MEASURES = {MEASURE_NAMES[column]: (column, title, fmt) for column, title, fmt in [
    ('reward_fn_normalized', f'{NAB_LOW_FN} (normalized)', '.2f'),
    ('standard_normalized', f'{NAB_STANDARD} (normalized)', '.2f'),
    ('test_auc_pr', 'Test AUC-PR', '.4f'),
    ('test_vus_pr', 'Test VUS-PR', '.4f'),
    ('f1', 'Test F1', '.4f'),
    ('precision', 'Test precision', '.4f'),
    ('recall', 'Test recall', '.4f'),
]}
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
# Bars of the subsets in one chart (Inference time per timestamp): the pattern of each subset, in their order
# (drawn in PATTERN_COLOR over the color of the bar: all the bars of a color equally dark), and its geometry in
# pixels
SUBSET_PATTERNS = ['solid', 'hatch', 'dots', 'crosshatch']
PATTERN_COLOR = 'white'
BAR_BORDER = ('#666666', 0.6)   # border of every bar (color, width), drawn over its pattern
PATTERN_SPACING = 4
PATTERN_STROKE = 0.9
PATTERN_DOT_SIZE = 3   # area (px²)
BAR_MIN_WIDTH = 8
BAR_GROUP_PADDING = 0.04   # of the width of the bars of a model, on each side
GRAPH_BAR_GROUP_PADDING = 0.25   # graph construction chart: of the width of the bars of a subset, on each side
LEGEND_SYMBOL_COLOR = '#555555'
LATENCY_LABEL_WINDOW = 30   # its bars show their value on top
MINUTES_FROM_SECONDS = 600   # cost bars of times in s: in minutes when the longest is at least that long
MODEL_LABEL_ANGLE = 35   # the model names of the cost bars, tilted when they would overlap
# y axes of the inference and graph construction charts: little room around their labels, a larger title (not
# bold)
COMPACT_Y_AXIS = dict(tickSize=3, labelPadding=1, titlePadding=2, domain=False, grid=False, titleFontWeight='normal')
Y_TITLE_FONT_INCREASE = 2   # their titles over the other text
BAR_LEGEND_FONT_INCREASE = 2   # their legend (bar_style_legend) over the other text
BAR_ANNOTATION_FONT_INCREASE = 1   # the values and graphs written over their bars, over the tick labels
WINDOW_TINTS = (0.35, 1.0)   # same chart: the family color of the smallest window mixed with white, of the largest
# Lines (Strategies by window tab): strategies by their point shape and dash, models of a family by their shade
STRATEGY_SHAPES = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'triangle-down']
LINE_SHADES = [1.0, 0.55, 0.3]
LINE_DASHES = {False: [1, 0], True: [6, 3]}   # window charts of the results table: solid / dashed
COMBINED_SUBSET_DASH = [2, 2]   # window charts of the results table: lines of a combined subset (e.g. 5xx 4xx count)
# Window charts of the results table, groups in one chart: the dash of each group, solid for a single subset and
# dotted for a combined one (then the next patterns for further groups of the same kind)
GROUP_DASHES = {False: [[1, 0], [6, 3], [8, 3, 2, 3]], True: [COMBINED_SUBSET_DASH, [1, 3], [2, 4, 6, 4]]}
CHART_BORDER = '#9e9e9e'   # border of the plots of the results-table charts
# Window charts of the results table sized as a figure of the paper: chart units are points (1/72 in), so the
# fonts have their size in points (src/figure_style.py) once the PNG is shown FIGURE_WIDTH_INCHES wide
FIGURE_WIDTH_INCHES = 4
POINTS_PER_INCH = 72
FIGURE_WIDTH = FIGURE_WIDTH_INCHES * POINTS_PER_INCH
# Figure of one column of the paper (IEEEtran journal, TNSM: \columnwidth = 3.5 in): the window charts of the results
# table with all the groups in one chart
COLUMN_FIGURE_WIDTH_INCHES = 3.5
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


@st.cache_data(show_spinner='Loading costs file…', max_entries=8)
def load_costs_path(path, mtime):
    """Costs file on disk (grid_results.read_run_costs, e.g. merged_results/hpc_combined_costs.csv); mtime
    refreshes the cache."""
    return read_costs(path)


@st.cache_data(show_spinner='Loading graph construction times…', max_entries=2)
def load_graph_times(path, mtime):
    """Graph construction times (src/time_graph_construction.py); mtime refreshes the cache."""
    return pd.read_csv(path)


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
BUBBLE_MEASURES = {MEASURE_NAMES[column]: (column, fmt) for column, fmt in [
    ('reward_fn_normalized', '.1f'), ('standard_normalized', '.1f'), ('nab_tp', '.0f'), ('f1', '.3f')]}
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
    nab_measures = [m for m in (NAB_LOW_FN, NAB_STANDARD) if MEASURES[m][0] in rows]
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


# ── Deployment cost ───────────────────────────────────────────────────────────

SAMPLING_INTERVAL_S = 300   # data/massaged/timestamps_and_labels.csv: one timestamp every 5 minutes
BYTES_PER_MB = 2 ** 20
# Costs a run can be compared by: label -> (column, axis title, number format)
RUN_COSTS = {
    'Inference per timestamp': ('latency_ms', 'Inference per timestamp (ms)', '.3f'),
    'Training time': ('training_time', 'Training time (s)', '.0f'),
    'Training time per epoch': ('epoch_time', 'Training time per epoch (s)', '.1f'),
    'Model size': ('model_mb', 'Model size (MB)', '.3f'),
    'Estimated memory': ('memory_mb', 'Estimated memory (MB)', '.1f'),
}
# The costs of a run in the tooltips of its bars (run_cost_bars)
COST_TOOLTIPS = [alt.Tooltip('latency_ms:Q', title='inference per timestamp (ms)', format='.3f'),
                 alt.Tooltip('inference_time:Q', title='test inference (s)', format='.1f'),
                 alt.Tooltip('test_timestamps:Q', title='test timestamps', format=',.0f'),
                 alt.Tooltip('training_time:Q', title='training (s)', format=',.0f'),
                 alt.Tooltip('epochs:Q', title='epochs', format='.0f'),
                 alt.Tooltip('epoch_time:Q', title='per epoch (s)', format='.1f')]
# Analytical memory of a deployed run (MB): label -> column; float32 inputs, float64 graph and covariance
MEMORY_PARTS = {
    'Model parameters': 'model_mb',
    'Input window (W×N×C)': 'window_mb',
    'Context graph (3×E)': 'graph_mb',
    'Mahalanobis covariance ((N·F)²)': 'covariance_mb',
}
GRAPH_KIND = {True: 'Graph-based', False: 'Other'}
DEPLOYMENT_FONT_SIZE_INCREASE = FONT_SIZE_INCREASE + 3   # inference and graph construction charts
# Widths of the charts merged in one figure, left to right (the training one when shown), within
# MERGED_FIGURE_WIDTH
MERGED_WIDTHS = {'graph': 4, 'training': 5, 'inference': 5}
MERGED_FIGURE_WIDTH = FIGURE_WIDTH * 10 / 7
HOLLOW = '(hollow)'   # fill value of the markers of the models that are not graph-based
# Build times of the graph of every subset (src/time_graph_construction.py), by implementation: label -> its column
GRAPH_TIMES_FILE = MERGED_DIR / 'graph_construction' / 'graph_construction_times.csv'
GRAPH_BUILDS = {'Naive (utils.py, used for the paper)': 'build_time_naive_mean_s',
                'Grouped by component and endpoint': 'build_time_grouped_mean_s'}
GRAPH_BUILD_COLORS = ['#d9d9d9', '#b3cde3']   # light: the graph of the subset is written in its bars
GRAPH_TEXT_COLOR = '#333333'


def run_cost_table(results, costs):
    """
    One row per run (COST_KEYS) of the results with its COST_COLUMNS (costs: a costs file, grid_results.
    read_run_costs) and the derived costs: training time per epoch, inference time per test timestamp (test
    timestamps: the confusion matrix of the run), model size and the analytical memory of MEMORY_PARTS. Also
    returns the grid-search files of every run.
    """
    first = results.loc[~results['file'].duplicated(),
                        ['file', *COST_KEYS, 'model_family',
                         *(['confusion_matrix'] if 'confusion_matrix' in results else [])]]
    runs = first.drop_duplicates(COST_KEYS).drop(columns='file').reset_index(drop=True)
    if 'confusion_matrix' not in runs:
        runs['confusion_matrix'] = None
    costs = costs.assign(model_folder=[model_variant(m, w) for m, w in zip(costs['model_folder'], costs['window'])])
    runs = runs.merge(costs.drop_duplicates(COST_KEYS)[COST_KEYS + COST_COLUMNS], on=COST_KEYS, how='left')
    runs['test_timestamps'] = runs['confusion_matrix'].map(
        lambda c: sum(ast.literal_eval(c)) if isinstance(c, str) else float('nan'))   # [tn, fp, fn, tp]
    runs['latency_ms'] = runs['inference_time'] / runs['test_timestamps'] * 1000
    runs['epoch_time'] = runs['training_time'] / runs['epochs']
    runs['graph'] = runs['model_family'].isin(GRAPH_MODEL_FAMILIES)
    channels = runs['num_node_features'] * (1 + runs['model_folder'].str.contains('null_padding_feature'))
    runs['model_mb'] = runs['model_size_bytes'] / BYTES_PER_MB
    runs['window_mb'] = runs['window'] * runs['num_nodes'] * channels * 4 / BYTES_PER_MB
    runs['graph_mb'] = (runs['num_edges'] * 3 * 8 / BYTES_PER_MB).where(runs['graph'], 0)
    runs['covariance_mb'] = (runs['num_nodes'] * runs['num_node_features']) ** 2 * 8 / BYTES_PER_MB
    runs['memory_mb'] = runs[list(MEMORY_PARTS.values())].sum(axis=1, min_count=1)
    return runs, first[['file', *COST_KEYS]]


def log_bar_base(values):
    """Where the bars of a log scale start: the power of 10 below the smallest value (Vega-Lite draws no bar
    from 0 on a log scale). Give it to the scale (domainMin) and to the bars (y2=alt.datum(base))."""
    return 10 ** math.floor(math.log10(values[values > 0].min()))


def _cost_axis(cost, log_base=None):
    """y of a cost of RUN_COSTS; log scale from log_base when given (log_bar_base)."""
    column, title, fmt = RUN_COSTS[cost]
    return alt.Y(f'{column}:Q', title=title, axis=alt.Axis(format=fmt, tickSize=3, labelPadding=2, domain=False),
                 scale=alt.Scale(type='log', domainMin=log_base) if log_base else alt.Undefined)


def latency_legend_groups(runs, family_scale):
    """The groups of the legend of latency_chart (bar_style_legend): model families, subsets, windows."""
    subsets = subset_order(runs['subset_name'])
    windows = sorted(runs['window'].unique())
    patterns = subset_patterns(subsets)
    colors = family_colors(family_scale)
    families = sorted(runs['model_family'].dropna().unique())
    groups = [('Model family', [(f, colors.get(f, NO_FAMILY_COLOR), 'solid') for f in families]),
              ('Subset', [(s, LEGEND_SYMBOL_COLOR, patterns[s]) for s in subsets])]
    if len(windows) > 1:
        tints = dict(zip(windows, window_tints(len(windows))))
        groups.append(('Window size', [(str(w), _tint(LEGEND_SYMBOL_COLOR, tints[w]), 'solid') for w in windows]))
    return groups


SUPERSCRIPTS = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
# Vega label of a power of ten on a log axis, short: 0.01, 0.1, 1, 10, then 10², 10³ (the fonts have no superscript
# minus: below 1 as decimals)
POWER_OF_TEN_LABEL = (json.dumps({str(e): (f'{10.0 ** e:.{-e}f}' if e < 0 else '10' + str(e).translate(SUPERSCRIPTS)
                                           if e > 1 else str(10 ** e)) for e in range(-12, 13)})
                      + "['' + round(log(datum.value) / LN10)]")   # '' +: '-1' (format would give a '−')


def subplot_letters(charts, widths, spacing):
    """The charts of a figure side by side with (a), (b), ... below each, as the subfigures of the paper: a grid
    whose second row (the letters, centered under the plots widths wide) starts below the tallest chart."""
    size = AXIS_LABEL_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE + Y_TITLE_FONT_INCREASE
    letters = [alt.Chart(pd.DataFrame({'text': [f'({chr(ord("a") + i)})']})).mark_text(
        fontSize=size, baseline='top').encode(x=alt.value(width / 2), y=alt.value(0), text='text:N')
        .properties(width=width, height=size, view=alt.ViewBackground(stroke=None))
        for i, width in enumerate(widths)]
    return alt.concat(*charts, *letters, columns=len(charts), spacing=alt.RowColnumber(row=2, column=spacing))


def scale_title(note):
    """The unit and scale of the y axis of a plot (e.g. '(ms, log scale)'), above its top-left corner."""
    return alt.TitleParams(note, anchor='start', frame='group', orient='top', offset=4, fontWeight='normal',
                           fontSize=TICK_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE)


def run_cost_bars(runs, family_scale, models, cost='Inference per timestamp', width=FIGURE_WIDTH, legend=True):
    """
    A cost of RUN_COSTS of every run (log scale), in one chart: one bar per model, subset and window side by
    side, in the color of the model family; the subset by the pattern of its bars (SUBSET_PATTERNS), the window by
    the tint of the color, darker for larger windows (WINDOW_TINTS).
    Vega-Lite has no fill patterns: the bars are laid out in pixels along x (an identity scale), and the hatches
    and dots drawn over them as marks of their own. Without paper_style (to concatenate it), width wide at least;
    without its legend (latency_legend_groups) when not legend.
    """
    cost_column, title, fmt = RUN_COSTS[cost]
    runs = runs[runs[cost_column] > 0]   # a log scale
    # Long times in minutes (10, 100 rather than 1,000, 10,000 s on the axis), in a column of their own
    column = 'shown_value'
    if title.endswith('(s)') and runs[cost_column].max() >= MINUTES_FROM_SECONDS:
        runs = runs.assign(shown_value=runs[cost_column] / 60)
        title, fmt = title.replace('(s)', '(min)'), '.0f'
    else:
        runs = runs.assign(shown_value=runs[cost_column])
    unit = title[title.rindex('(') + 1:-1]   # e.g. 'ms'
    title, scale_note = title[:title.rindex('(')].strip(), f'({unit}, log scale)'   # the note above the plot
    subsets = subset_order(runs['subset_name'])
    windows = sorted(runs['window'].unique())
    models = [m for m in models if m in set(runs['model'])]
    patterns = subset_patterns(subsets)
    tints = dict(zip(windows, window_tints(len(windows))))
    colors = family_colors(family_scale)
    slots = {(s, w): i for i, (s, w) in enumerate(itertools.product(subsets, windows))}
    # Pixels along x: a group of bars per model, BAR_GROUP_PADDING of it around its bars, bars never narrower
    # than BAR_MIN_WIDTH (the chart widens instead)
    group = max(width / len(models), len(slots) * BAR_MIN_WIDTH / (1 - 2 * BAR_GROUP_PADDING))
    width, bar = group * len(models), group * (1 - 2 * BAR_GROUP_PADDING) / len(slots)
    height = FIGURE_WIDTH * CHART_ASPECT
    base = log_bar_base(runs[column])
    # The values of the bars of LATENCY_LABEL_WINDOW written on top of them: upright when they fit the width of a
    # bar, else rotated (taller)
    font_size = TICK_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE + BAR_ANNOTATION_FONT_INCREASE
    labelled = runs[runs['window'] == LATENCY_LABEL_WINDOW]
    longest = max((len(f'{v:{fmt}}') for v in labelled[column]), default=0) * 0.55 * font_size
    upright = longest <= bar + 2
    room = (font_size if upright else longest) + 8 if not labelled.empty else 4   # pixels above the tallest bar
    # The top of the axis just above the tallest bar, room pixels left for its label (on a log scale, the share f
    # of the height above the tallest bar m: log top = (log m - f log base) / (1 - f))
    tallest, share = math.log10(runs[column].max()), room / height
    top = max(10 ** ((tallest - share * math.log10(base)) / (1 - share)), base * 10)   # a decade at least
    low, high = math.log10(base), math.log10(top)

    def y_pixel(value):
        return height * (1 - (math.log10(value) - low) / (high - low))

    def y_value(pixel):   # the value at a pixel row of the log scale
        return 10 ** (low + (height - pixel) / height * (high - low))

    bars, boxes = [], []
    for row in runs.itertuples():
        x0 = models.index(row.model) * group + group * BAR_GROUP_PADDING + slots[(row.subset_name, row.window)] * bar
        color = _tint(colors.get(row.model_family, NO_FAMILY_COLOR), tints[row.window])
        pattern = patterns[row.subset_name]
        bars.append({**row._asdict(), 'x0': x0, 'x1': x0 + bar, 'color': _behind_pattern(color, pattern)})
        boxes.append((pattern, x0, x0 + bar, y_pixel(getattr(row, column)), height))
    bars = pd.DataFrame(bars)

    x_scale = alt.Scale(domain=[0, width], nice=False, zero=False)   # identity: one unit a pixel
    y_scale = alt.Scale(type='log', domain=[base, top], nice=False)
    tick_size = TICK_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE
    bold = ['bold' if str(m).startswith(PROPOSED_MODEL_PREFIX) else 'normal' for m in models]   # the proposed models
    # The model names tilted when one is wider than its group of bars (they would overlap)
    tilted = max(len(str(m)) for m in models) * 0.55 * tick_size > 0.95 * group
    # tickCount: else Vega keeps every other of the values on a narrow chart (ceil(width / 40) ticks)
    x_axis = alt.Axis(values=[group * (i + 0.5) for i in range(len(models))], tickCount=len(models), title=None,
                      grid=False, labelAngle=-MODEL_LABEL_ANGLE if tilted else 0,
                      labelAlign='right' if tilted else 'center', labelBaseline='top', labelLimit=0,
                      labelOverlap=False, domain=False,
                      labelFontWeight={'expr': json.dumps(bold) + f'[round((datum.value - {group / 2}) / {group})]'},
                      labelExpr=f'{json.dumps(models)}[round((datum.value - {group / 2}) / {group})]')
    rects = alt.Chart(bars).mark_rect(strokeWidth=1).encode(
        x=alt.X('x0:Q', scale=x_scale, axis=x_axis), x2='x1:Q',
        y=alt.Y(f'{column}:Q', title=title, scale=y_scale,
                # labels at the powers of ten only, as powers (10⁻¹, 10⁰, 10¹ rather than 0.1, 1, 10)
                axis=alt.Axis(values=[10 ** e for e in range(round(low), math.floor(high) + 1)],
                              labelExpr=POWER_OF_TEN_LABEL,
                              titleFontSize=AXIS_LABEL_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE + Y_TITLE_FONT_INCREASE,
                              **COMPACT_Y_AXIS)),
        y2=alt.datum(base),
        fill=alt.Fill('color:N', scale=None), stroke=alt.Stroke('color:N', scale=None),
        tooltip=['model', 'subset_name', 'window', 'fill_nan', *COST_TOOLTIPS])
    layers = [rects, *pattern_layers(boxes, y_value)]
    layers.append(alt.Chart(bars).mark_rect(filled=False, stroke=BAR_BORDER[0], strokeWidth=BAR_BORDER[1]).encode(
        x='x0:Q', x2='x1:Q', y=f'{column}:Q', y2=alt.datum(base)))   # over the patterns
    # The value of the bars of LATENCY_LABEL_WINDOW on top of them (the highest of a bar drawn several times, e.g.
    # for several fill NaN)
    labelled = bars[bars['window'] == LATENCY_LABEL_WINDOW]
    if not labelled.empty:
        labelled = labelled.loc[labelled.groupby(['x0'])[column].idxmax()].copy()
        labelled['label'] = [f'{v:{fmt}}' for v in labelled[column]]   # in the unit of the axis title
        labelled['x_mid'] = labelled['x0'] + bar / 2
        labelled['y_label'] = [y_value(y_pixel(v) - 3) for v in labelled[column]]   # 3 px above the bar
        text = dict(align='center', baseline='bottom') if upright else dict(align='left', baseline='middle', angle=270)
        layers.append(alt.Chart(labelled).mark_text(fontSize=font_size, **text).encode(
            x='x_mid:Q', y='y_label:Q', text='label:N'))
    chart = alt.layer(*layers).properties(width=width, height=height, title=scale_title(scale_note))
    if not legend:
        return chart
    return alt.vconcat(bar_style_legend(latency_legend_groups(runs, family_scale), max_width=width), chart, spacing=2)


def is_combined_subset(name):
    """True for the short name of a subset combining several http codes (e.g. 5xx 4xx count)."""
    return sum(token.endswith('xx') for token in str(name).split()) > 1


def subset_order(subsets):
    """Subsets (short names) in the order of every chart: the single subsets (e.g. 4xx count, 5xx count) by name,
    then the combined ones (e.g. 5xx 4xx count)."""
    return sorted(dict.fromkeys(subsets), key=lambda s: (is_combined_subset(s), str(s)))


def subset_patterns(subsets):
    """{subset: pattern of SUBSET_PATTERNS}, the same in every chart, in the order of subset_order."""
    return {s: SUBSET_PATTERNS[i % len(SUBSET_PATTERNS)] for i, s in enumerate(subset_order(subsets))}


def pattern_layers(boxes, y_value):
    """Layers drawing the patterns of bars over them, boxes [(pattern, x0, x1, y0, y1)] in pixels (y down) on a
    chart whose x is in pixels and whose y_value(pixel) gives its y at a pixel row."""
    lines, dots = [], []
    for pattern, *box in boxes:
        lines += [{'xa': xa, 'xb': xb, 'ya': y_value(ya), 'yb': y_value(yb)}
                  for xa, ya, xb, yb in pattern_lines(pattern, *box)]
        dots += [{'x': xd, 'y': y_value(yd)} for xd, yd in pattern_dots(pattern, *box)]
    layers = []
    if lines:
        layers.append(alt.Chart(pd.DataFrame(lines)).mark_rule(strokeWidth=PATTERN_STROKE).encode(
            x='xa:Q', x2='xb:Q', y='ya:Q', y2='yb:Q', stroke=alt.value(PATTERN_COLOR)))
    if dots:
        layers.append(alt.Chart(pd.DataFrame(dots)).mark_circle(size=PATTERN_DOT_SIZE, opacity=1).encode(
            x='x:Q', y='y:Q', color=alt.value(PATTERN_COLOR)))
    return layers


def pattern_lines(pattern, x0, x1, y0, y1, spacing=PATTERN_SPACING):
    """Segments (xa, ya, xb, yb) of the hatches of a pattern in the box [x0, x1] x [y0, y1] (pixels, y down):
    'hatch' diagonal lines going up to the right, 'crosshatch' both diagonals."""
    directions = {'hatch': [1], 'crosshatch': [1, -1]}.get(pattern, [])
    segments = []
    for d in directions:   # d = 1: x + y = c; d = -1: x - y = c
        low, high = (x0 + y0, x1 + y1) if d == 1 else (x0 - y1, x1 - y0)
        c = low + spacing / 2
        while c < high:
            if d == 1:
                xa, xb = max(x0, c - y1), min(x1, c - y0)
                segments.append((xa, c - xa, xb, c - xb))
            else:
                xa, xb = max(x0, c + y0), min(x1, c + y1)
                segments.append((xa, xa - c, xb, xb - c))
            c += spacing
    return [s for s in segments if s[2] - s[0] > 0.5]


def pattern_dots(pattern, x0, x1, y0, y1, spacing=PATTERN_SPACING):
    """Centers (x, y) of the dots of the 'dots' pattern in the box [x0, x1] x [y0, y1] (pixels): staggered rows."""
    if pattern != 'dots':
        return []
    dots, row, y = [], 0, y0 + spacing / 2
    while y < y1 - spacing / 4:
        x = x0 + spacing / 2 + (row % 2) * spacing / 2
        while x < x1 - spacing / 4:
            dots.append((x, y))
            x += spacing
        row, y = row + 1, y + spacing
    return dots


def pattern_coverage(pattern, spacing=PATTERN_SPACING):
    """Share of the area of a bar under the marks of its pattern (PATTERN_COLOR)."""
    line = PATTERN_STROKE / (spacing / math.sqrt(2))   # diagonal lines, spacing / sqrt(2) apart
    return {'hatch': line, 'crosshatch': 2 * line - line ** 2, 'dots': PATTERN_DOT_SIZE / spacing ** 2}.get(pattern, 0)


def _behind_pattern(color, pattern):
    """The fill of a bar under its white pattern for the bar to look as dark as a solid one of color: darker by
    the share of white of the pattern (as far as black allows)."""
    white = pattern_coverage(pattern)
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{min(255, max(0, round((c - white * 255) / (1 - white)))):02x}' for c in rgb)


def window_tints(n):
    """Tints of n windows, smallest first: from WINDOW_TINTS[0] (light) to WINDOW_TINTS[1] (the full color)."""
    low, high = WINDOW_TINTS
    return [high] if n == 1 else [low + (high - low) * i / (n - 1) for i in range(n)]


def _tint(color, amount):
    """color mixed with white: amount 1 the color itself, 0 white."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(255 - amount * (255 - c)):02x}' for c in rgb)


def bar_style_legend(groups, max_width=None):
    """
    Legend of the styles of the bars of a chart, drawn as a chart of its own, its groups side by side (on rows of
    max_width pixels at most when given): groups [(title, [(label, color, pattern of SUBSET_PATTERNS), ...]), ...].
    Vega-Lite has no fill patterns, nor a legend for colors given without a scale.
    """
    font_size = LEGEND_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE + BAR_LEGEND_FONT_INCREASE
    # symbol side, space after a label, width of a character (px)
    side, gap, char = round(1.1 * font_size), font_size, 0.55 * font_size
    top, row_height = 3, side + 8

    def group_width(title, entries):
        return len(title) * char * 1.1 + gap + sum(side + 4 + len(label) * char + gap for label, _, _ in entries)

    x, y, widest, headings, items, lines, dots = 0, top, 0, [], [], [], []
    for title, entries in groups:
        if max_width and x > 0 and x + group_width(title, entries) > max_width:   # on a row of its own
            x, y = 0, y + row_height
        headings.append({'text': title, 'x': x, 'y': y + side / 2})
        x += len(title) * char * 1.1 + gap
        for label, color, pattern in entries:
            items.append({'text': label, 'x': x, 'x_end': x + side, 'label_x': x + side + 4, 'y': y,
                          'y_end': y + side, 'y_mid': y + side / 2, 'color': _behind_pattern(color, pattern)})
            box = (x, x + side, y, y + side)
            lines += [{'xa': xa, 'ya': ya, 'xb': xb, 'yb': yb}
                      for xa, ya, xb, yb in pattern_lines(pattern, *box)]
            dots += [{'x': xd, 'y': yd} for xd, yd in pattern_dots(pattern, *box)]
            x += side + 4 + len(label) * char + gap
        widest = max(widest, x)
        x += gap
    pixels = dict(scale=None)
    items = pd.DataFrame(items)
    layers = [
        alt.Chart(pd.DataFrame(headings)).mark_text(
            align='left', baseline='middle', fontSize=font_size, fontWeight='bold').encode(
            x=alt.X('x:Q', **pixels), y=alt.Y('y:Q', **pixels), text='text:N'),
        alt.Chart(items).mark_rect(strokeWidth=1).encode(
            x=alt.X('x:Q', **pixels), x2='x_end:Q', y=alt.Y('y:Q', **pixels), y2='y_end:Q',
            fill=alt.Fill('color:N', **pixels), stroke=alt.Stroke('color:N', **pixels)),
        alt.Chart(items).mark_text(align='left', baseline='middle', fontSize=font_size).encode(
            x=alt.X('label_x:Q', **pixels), y=alt.Y('y_mid:Q', **pixels), text='text:N')]
    if lines:
        layers.append(alt.Chart(pd.DataFrame(lines)).mark_rule(strokeWidth=PATTERN_STROKE).encode(
            x=alt.X('xa:Q', **pixels), x2='xb:Q', y=alt.Y('ya:Q', **pixels), y2='yb:Q',
            stroke=alt.value(PATTERN_COLOR)))
    if dots:
        layers.append(alt.Chart(pd.DataFrame(dots)).mark_circle(size=PATTERN_DOT_SIZE, opacity=1).encode(
            x=alt.X('x:Q', **pixels), y=alt.Y('y:Q', **pixels), color=alt.value(PATTERN_COLOR)))
    layers.append(alt.Chart(items).mark_rect(filled=False, stroke=BAR_BORDER[0], strokeWidth=BAR_BORDER[1]).encode(
        x=alt.X('x:Q', **pixels), x2='x_end:Q', y=alt.Y('y:Q', **pixels), y2='y_end:Q'))
    return alt.layer(*layers).properties(   # no frame (paper_style frames every plot)
        width=max(widest, 1), height=y + row_height - 5, view=alt.ViewBackground(stroke=None))


def cost_accuracy_chart(runs, family_scale, cost, measure):
    """The best measure of every run against its cost (log scale): the marker by its subset (e.g. 5xx count,
    5xx 4xx count); graph-based models filled, the others hollow."""
    column = RUN_COSTS[cost][0]
    data = runs.dropna(subset=[column, 'best'])
    data = data[data[column] > 0]
    # Fill by a column of its own, not alt.condition: Vega-Lite would draw the subset legend with the 'white' of
    # the condition and no stroke, i.e. invisible markers
    data = data.assign(fill_family=data['model_family'].where(data['graph'], HOLLOW))
    colors = family_colors(family_scale)
    families = [f for f in colors if f in set(data['model_family'])]
    subsets = list(dict.fromkeys(data['subset_name']))
    points = alt.Chart(data).mark_point(size=60, strokeWidth=1.5).encode(
        x=alt.X(f'{column}:Q', title=RUN_COSTS[cost][1], scale=alt.Scale(type='log'),
                axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),
        y=alt.Y('best:Q', title=f'Best {measure}', scale=alt.Scale(zero=False, nice=False, padding=8),
                axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),
        color=alt.Color('model_family:N', scale=family_scale, legend=None),
        fill=alt.Fill('fill_family:N', title='Model family', scale=alt.Scale(
            domain=[*families, HOLLOW], range=[*(colors[f] for f in families), 'white']),
                      legend=alt.Legend(orient='top', values=families, columns=2, labelLimit=0,
                                        strokeColor='transparent')),
        shape=alt.Shape('subset_name:N', title='Subset', sort=subsets, scale=alt.Scale(
            domain=subsets, range=_cycled(STRATEGY_SHAPES, len(subsets))),
                        legend=alt.Legend(orient='top', columns=2, labelLimit=0, symbolFillColor='black',
                                          symbolStrokeColor='black', strokeColor='transparent')),
        tooltip=['model', 'kind', 'subset_name', 'window', 'fill_nan',
                 alt.Tooltip(f'{column}:Q', title=cost, format=RUN_COSTS[cost][2]),
                 alt.Tooltip('best:Q', title=f'Best {measure}', format=MEASURES[measure][2])])
    return paper_style(points.properties(width=FIGURE_WIDTH, height=FIGURE_WIDTH * CHART_ASPECT))


def ensemble_cost_rows(runs):
    """Runs of the combined subsets (one model on e.g. 5xx 4xx count) and the ensembles of their single subsets
    (the models of 5xx count and 4xx count, same model, window and fill NaN): one row per (comparison, part),
    the cost of an ensemble being the sum of those of its members (the fusion, an OR of their labels, costs
    nothing next to them)."""
    single = runs.set_index(COST_KEYS)
    rows = []
    for _, run in runs.iterrows():
        components = component_subsets(run['subset'])
        if not components:
            continue
        members = [(run['window'], c, run['fill_nan'], run['model_folder']) for c in components]
        if not all(m in single.index for m in members):
            continue
        comparison = f'{run["model"]} · w={run["window"]} · {run["fill_nan"]}'
        rows.append({**run, 'comparison': comparison, 'approach': 'One model (combined subset)',
                     'part': run['subset_name']})
        for member in members:
            rows.append({**single.loc[member].to_dict(), **dict(zip(COST_KEYS, member)), 'comparison': comparison,
                         'approach': f'Ensemble of {len(members)} models', 'part': short_subset_name(member[1]),
                         'model': run['model']})
    return pd.DataFrame(rows)


def ensemble_cost_chart(rows, cost):
    """Stacked bars: the cost of the combined-subset model next to the summed costs of the ensemble members."""
    column, title, fmt = RUN_COSTS[cost]
    comparisons = list(dict.fromkeys(rows['comparison']))
    bars = alt.Chart().mark_bar(stroke='white', strokeWidth=0.5).encode(
        x=alt.X('approach:N', title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
        y=alt.Y(f'sum({column}):Q', title=title, axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),
        color=alt.Color('part:N', title='Subset', legend=alt.Legend(orient='top')),
        order=alt.Order('part:N'),
        tooltip=['comparison', 'approach', 'part', alt.Tooltip(f'{column}:Q', title=cost, format=fmt),
                 alt.Tooltip('best:Q', title='best measure', format='.3f')])
    columns = min(PANEL_COLUMNS, len(comparisons))
    width = panel_width(columns)
    return paper_style(alt.layer(bars, data=rows).properties(width=width, height=width * PANEL_ASPECT).facet(
        facet=alt.Facet('comparison:N', sort=comparisons, title=None), columns=columns, spacing=PANEL_SPACING
    ).resolve_scale(y='independent'))


def _thousands(count):
    """A count in thousands: 2406 -> '2.4k', 170644 -> '171k'; below a thousand as it is."""
    if count < 1000:
        return f'{count:.0f}'
    return f'{count / 1000:.1f}k' if count < 10000 else f'{count / 1000:.0f}k'


def graph_build_chart(times, width=FIGURE_WIDTH, legend=True):
    """Build time of the graph of every subset, one bar per implementation (GRAPH_BUILDS, by its color), the subset
    by the pattern of its bars as in the inference chart (subset_patterns); above the tallest bar of each subset
    (and its label), the nodes, edges and sparsity (the share of the N·(N-1) directed pairs without an edge) of its
    graph, in a smaller font when the subsets are too narrow for it. The bars are laid out in pixels along x (an
    identity scale) for their patterns. Without paper_style (to concatenate it); without its subset legend when
    not legend (e.g. next to the inference chart, which has it)."""
    times = times.assign(sparsity=1 - times['num_edges'] / (times['num_nodes'] * (times['num_nodes'] - 1)))
    long = times.melt(id_vars=['subset_name', 'num_nodes', 'num_edges', 'sparsity', 'repeats'],
                      value_vars=[c for c in GRAPH_BUILDS.values() if c in times], var_name='column',
                      value_name='seconds').dropna(subset=['seconds'])
    long['implementation'] = long['column'].map({c: label for label, c in GRAPH_BUILDS.items()})
    # The axis in ms below a second, else in s: short tick labels (5, 10, 15 ms rather than 0.005, 0.01, 0.015 s);
    # the labels of the bars in its unit too (in its title)
    unit, factor = ('ms', 1000) if long['seconds'].max() < 1 else ('s', 1)
    long['value'] = long['seconds'] * factor
    long['label'] = [f'{v:.3g}' for v in long['value']]
    subsets = subset_order(times['subset_name'])
    implementations = [i for i in GRAPH_BUILDS if i in set(long['implementation'])]
    patterns = subset_patterns(subsets)
    build_colors = dict(zip(GRAPH_BUILDS, GRAPH_BUILD_COLORS))
    graphs = long.loc[long.groupby('subset_name')['value'].idxmax()].copy()   # over the tallest bar of a subset
    # Short lines (counts in thousands, e.g. N=6.5k, E=171k): a larger font within the width of a subset
    graphs['graph'] = [f'N={_thousands(n)}\nE={_thousands(e)}\n{p:.2%}\nsparse'
                       for n, e, p in zip(graphs['num_nodes'], graphs['num_edges'], graphs['sparsity'])]
    lines = 4
    font_size = TICK_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE + BAR_ANNOTATION_FONT_INCREASE
    group = width / len(subsets)
    widest = max(len(line) for text in graphs['graph'] for line in text.split('\n')) * 0.55 * font_size
    graph_font = min(font_size, font_size * 0.92 * group / widest)   # within the width of a subset
    line_height = 1.2 * graph_font
    height = FIGURE_WIDTH * CHART_ASPECT
    # The graph above the label of its bar: label and block heights in pixels, turned into values by the scale
    # (y_max over height), which leaves room for them above the tallest bar
    padding, block, label = 4, lines * line_height, font_size + 7
    y_max = (graphs['value'] / (1 - (label + 2 * padding + block) / height)).max()   # padding under the frame
    pixel = y_max / height
    graphs['y_text'] = graphs['value'] + (label + padding + block / 2) * pixel

    # Pixels along x: a group of bars per subset, GRAPH_BAR_GROUP_PADDING of it around its bars
    bar = group * (1 - 2 * GRAPH_BAR_GROUP_PADDING) / len(implementations)
    long['x0'] = [subsets.index(s) * group + group * GRAPH_BAR_GROUP_PADDING + implementations.index(i) * bar
                  for s, i in zip(long['subset_name'], long['implementation'])]
    long['x1'] = long['x0'] + bar
    long['x_mid'] = long['x0'] + bar / 2
    long['color'] = [_behind_pattern(build_colors[i], patterns[s])
                     for s, i in zip(long['subset_name'], long['implementation'])]
    graphs[['x_mid']] = long.loc[graphs.index, ['x_mid']]
    boxes = [(patterns[s], x0, x1, height * (1 - v / y_max), height)
             for s, x0, x1, v in zip(long['subset_name'], long['x0'], long['x1'], long['value'])]

    x_scale = alt.Scale(domain=[0, width], nice=False, zero=False)   # identity: one unit a pixel
    y_scale = alt.Scale(domain=[0, y_max], nice=False)
    x_axis = alt.Axis(values=[group * (i + 0.5) for i in range(len(subsets))], tickCount=len(subsets), title=None,
                      grid=False,
                      labelAngle=-MODEL_LABEL_ANGLE, labelAlign='right', labelBaseline='top',   # as the cost bars
                      labelLimit=0, labelOverlap=False, domain=False,
                      labelExpr=f'{json.dumps(subsets)}[round((datum.value - {group / 2}) / {group})]')
    bars = alt.Chart(long).mark_rect().encode(
        x=alt.X('x0:Q', scale=x_scale, axis=x_axis), x2='x1:Q',
        y=alt.Y('value:Q', title='Graph construction time', scale=y_scale,
                axis=alt.Axis(format='.4~g', **COMPACT_Y_AXIS,
                              titleFontSize=AXIS_LABEL_FONT_SIZE + DEPLOYMENT_FONT_SIZE_INCREASE + Y_TITLE_FONT_INCREASE)),
        y2=alt.datum(0),
        fill=alt.Fill('color:N', scale=None),   # GRAPH_BUILDS; no legend for now
        tooltip=['subset_name', 'implementation', alt.Tooltip('seconds:Q', title='mean (s)', format='.4f'),
                 alt.Tooltip('repeats:Q', title='builds'), alt.Tooltip('num_nodes:Q', title='nodes N', format=','),
                 alt.Tooltip('num_edges:Q', title='edges E', format=','),
                 alt.Tooltip('sparsity:Q', title='sparsity', format='.2%')])
    borders = alt.Chart(long).mark_rect(filled=False, stroke=BAR_BORDER[0], strokeWidth=BAR_BORDER[1]).encode(
        x='x0:Q', x2='x1:Q', y='value:Q', y2=alt.datum(0))   # over the patterns
    labels = alt.Chart(long).mark_text(dy=-7, fontSize=font_size).encode(
        x='x_mid:Q', y='value:Q', text='label:N')
    graph_labels = alt.Chart(graphs).mark_text(   # Vega places the first line at y: up half the others
        fontSize=graph_font, lineBreak='\n', lineHeight=line_height, baseline='middle',
        dy=-(lines - 1) / 2 * line_height,
        color=GRAPH_TEXT_COLOR).encode(x='x_mid:Q', y='y_text:Q', text='graph:N')
    layers = [bars, *pattern_layers(boxes, lambda px: (height - px) / height * y_max), borders, labels,
              graph_labels]
    chart = alt.layer(*layers).properties(width=width, height=height, title=scale_title(f'({unit}, linear scale)'))
    if not legend:
        return chart
    legend = bar_style_legend([('Subset', [(s, LEGEND_SYMBOL_COLOR, patterns[s]) for s in subsets])], max_width=width)
    return alt.vconcat(legend, chart, spacing=2)


def graph_build_times(graphs):
    """Build times of the graphs of the subsets of graphs (a row per subset) in GRAPH_TIMES_FILE, with their
    checks shown; None (and how to write them) without the file."""
    if not GRAPH_TIMES_FILE.is_file():
        st.info(f'No graph construction times (`{os.path.relpath(GRAPH_TIMES_FILE, PROJECT_ROOT)}`): write them '
                'with `python src/time_graph_construction.py` on the HPC server (`--skip-naive`: the grouped '
                'implementation only, in seconds).')
        return None
    times = load_graph_times(str(GRAPH_TIMES_FILE), os.path.getmtime(GRAPH_TIMES_FILE))
    times = times[times['subset'].isin(graphs['subset'])] if times['subset'].isin(graphs['subset']).any() else times
    times = times.assign(subset_name=times['subset'].map(short_subset_name))
    naive, grouped = GRAPH_BUILDS.values()
    times['speedup'] = times[naive] / times[grouped]
    checks = {'same_as_naive': 'the naive implementation', 'same_as_existing': 'the adjacency_matrix.npy of the subset'}
    for column, what in checks.items():
        checked = times[column].dropna() if column in times else pd.Series(dtype=bool)
        if not checked.empty and not checked.astype(str).eq('True').all():
            st.warning(f'The grouped graph differs from {what} for: '
                       f'{", ".join(times.loc[checked.index[~checked.astype(str).eq("True")], "subset_name"])}.')
    return times


GRAPH_BUILD_CAPTION = ('**Graph construction time**: built from the names of the nodes alone, once per subset. The '
                       'naive implementation compares every pair of nodes through an N×N matrix; the grouped one only '
                       'the nodes of a same component and endpoint (the only ones with an edge), with the same edges.')


def graph_times_caption(times):
    naive = GRAPH_BUILDS[next(iter(GRAPH_BUILDS))]
    skipped = ' No naive build time in the file (written with `--skip-naive`).' if times[naive].isna().all() else ''
    return (f'Mean over the builds of `{os.path.relpath(GRAPH_TIMES_FILE, PROJECT_ROOT)}` (written '
            f'{datetime.fromtimestamp(os.path.getmtime(GRAPH_TIMES_FILE)):%Y-%m-%d %H:%M}).{skipped}')


def memory_chart(runs, models):
    """Analytical memory of the runs of one window: stacked MEMORY_PARTS per model, one panel per subset."""
    long = runs.melt(id_vars=['model', 'subset_name'], value_vars=list(MEMORY_PARTS.values()),
                     var_name='column', value_name='mb')
    long['part'] = long['column'].map({c: p for p, c in MEMORY_PARTS.items()})
    parts = list(MEMORY_PARTS)
    bars = alt.Chart().mark_bar().encode(
        x=alt.X('model:N', sort=models, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=0)),
        y=alt.Y('sum(mb):Q', title='Estimated memory (MB)', axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),
        color=alt.Color('part:N', sort=parts, title=None, legend=alt.Legend(orient='top', columns=2, labelLimit=0)),
        order=alt.Order('part_order:Q'),
        tooltip=['model', 'subset_name', 'part', alt.Tooltip('mb:Q', title='MB', format='.3f')])
    long['part_order'] = long['part'].map(parts.index)
    subsets = list(dict.fromkeys(long['subset_name']))
    columns = min(PANEL_COLUMNS, len(subsets))
    width = panel_width(columns)
    return paper_style(alt.layer(bars, data=long).properties(width=width, height=width * PANEL_ASPECT).facet(
        facet=alt.Facet('subset_name:N', sort=subsets, title=None), columns=columns, spacing=PANEL_SPACING))


def render_deployment_cost(results, family_scale):
    """
    'Deployment cost' tab: the cost of every run (training / inference time, model size, analytical memory, the
    graph of its subset), against its accuracy, and of the ensembles of single subsets against one model on the
    combined subset. Its filters are its own: the sidebar filters do not apply.
    """
    st.caption('This tab has its own filters: the filters of the sidebar do not apply here.')
    costs_files = sorted(MERGED_DIR.glob(f'*{COSTS_SUFFIX}'), key=os.path.getmtime, reverse=True)
    if not costs_files:
        st.info(f'No costs file (`*{COSTS_SUFFIX}`) in `{os.path.relpath(MERGED_DIR, PROJECT_ROOT)}/`: write one on '
                'the HPC server with `grid_results.py <results folder> -o <merged file> --costs-only` (it reads the '
                'training / inference time files and the model file of every run, and the graph of every subset '
                'from `data/massaged/extracted/`), e.g. `merged_results/hpc_combined_costs.csv`.')
        return
    names = [p.name for p in costs_files]
    default = f'hpc_combined{COSTS_SUFFIX}'
    costs_file = costs_files[st.selectbox('Costs file', range(len(names)), format_func=names.__getitem__,
                                          index=names.index(default) if default in names else 0, key='cost_file',
                                          help='Training / inference times, model sizes and graphs of the runs.')]
    try:
        costs = load_costs_path(str(costs_file), os.path.getmtime(costs_file))
    except ValueError as e:
        st.error(f'`{costs_file.name}`: {e}')
        return
    runs, run_files = run_cost_table(results, costs)
    if runs['inference_time'].isna().all():
        st.info('The costs file has no cost for the runs of these results.')
        return
    with st.expander('Filters', expanded=True, icon=':material/filter_alt:'):
        filter_columns = st.columns(len(COST_KEYS))
        chosen = {c: filter_columns[i].multiselect(DIMENSIONS[c], sorted(runs[c].unique(), key=str), default=[],
                                                   placeholder='All', key=f'cost_filter_{c}')
                  for i, c in enumerate(COST_KEYS)}
    for column, values in chosen.items():
        if values:
            runs = runs[runs[column].isin(values)]
    if runs.empty:
        st.info('No run matches the filters.')
        return

    col_measure, col_interval, col_short = st.columns(3)
    available = [m for m, (c, _, _) in MEASURES.items() if c in results and results[c].notna().any()]
    measure = col_measure.selectbox('Accuracy', available, key='cost_measure',
                                    help='Best test value of a run over its settings (strategies, thresholds, ...).')
    interval = col_interval.number_input('Sampling interval (s)', min_value=1, value=SAMPLING_INTERVAL_S,
                                         key='cost_interval', help='Time between two timestamps of the data: the '
                                                                   'budget of the inference of one timestamp.')
    short_names = col_short.toggle('Short names', value=True, key='cost_short_names')

    # Best measure of every run shown, over the rows of its grid-search files only
    column = MEASURES[measure][0]
    files = run_files.merge(runs[COST_KEYS], on=COST_KEYS)['file']
    best = results.loc[results['file'].isin(files)].groupby(COST_KEYS)[column].max().rename('best')
    runs = runs.merge(best.reset_index(), on=COST_KEYS, how='left')
    runs['model'] = runs['model_folder'].map(short_model_name) if short_names else runs['model_folder']
    runs['subset_name'] = runs['subset'].map(short_subset_name) if short_names else runs['subset']
    runs['kind'] = runs['graph'].map(GRAPH_KIND)
    runs['realtime_factor'] = interval * 1000 / runs['latency_ms']
    runs = runs.assign(_order=runs['model_folder'].map(model_order_key)).sort_values(
        ['_order', 'subset', 'window'], kind='stable').drop(columns='_order')
    models = list(dict.fromkeys(runs['model']))

    st.subheader('Inference time per timestamp', divider='gray')
    timed = runs.dropna(subset=['latency_ms'])
    windows = sorted(int(w) for w in timed['window'].unique())
    col_windows, col_merge, col_training = st.columns([2, 1, 1])
    shown_windows = col_windows.multiselect(
        'Window sizes', windows, key='cost_latency_windows', help='Window sizes of the bars of this chart only.',
        default=[LATENCY_LABEL_WINDOW] if LATENCY_LABEL_WINDOW in windows else windows)
    graphs = runs.drop_duplicates('subset')[['subset', 'subset_name', 'num_nodes', 'num_node_features',
                                              'num_edges']].copy()
    times = graph_build_times(graphs)
    merged = col_merge.toggle('With the graph construction time', value=True, key='cost_merge_graph',
                              disabled=times is None, help='One figure: the graph construction chart on the left '
                              'of the inference chart.')
    merged = merged and times is not None
    training_cost = col_training.selectbox(
        'With the training time', ['None', 'Training time', 'Training time per epoch'], index=2,
        key='cost_merge_training', disabled=not merged,
        help='A chart of the training time of the runs between the two (same windows).')
    with_training = merged and training_cost != 'None'
    timed = timed[timed['window'].isin(shown_windows)]
    slowest = timed['latency_ms'].max()
    if not shown_windows and windows:
        st.info('Choose at least one window size.')
    elif pd.notna(slowest):
        st.caption(f'Test-split inference time divided by its number of timestamps (batched on the GPU of the run, '
                   f'post-processing excluded). Slowest run: {slowest:.2f} ms per timestamp, '
                   f'{interval * 1000 / slowest:,.0f}× below the sampling interval of {interval} s.')
        chart = run_cost_bars(timed, family_scale, models)
        name = 'deployment_inference_per_timestamp'
        if merged:   # MERGED_WIDTHS of MERGED_FIGURE_WIDTH
            st.caption(GRAPH_BUILD_CAPTION + ' ' + graph_times_caption(times))
            parts = ['graph', 'training', 'inference'] if with_training else ['graph', 'inference']
            unit = MERGED_FIGURE_WIDTH / sum(MERGED_WIDTHS[p] for p in parts)
            # Graph construction, training, inference, left to right; one legend above them (the plots aligned)
            plots = [graph_build_chart(times, width=MERGED_WIDTHS['graph'] * unit, legend=False)]
            name = 'deployment_graph_construction_time'
            if with_training:
                trained = timed.dropna(subset=[RUN_COSTS[training_cost][0]])
                plots.append(run_cost_bars(trained, family_scale, models, cost=training_cost,
                                           width=MERGED_WIDTHS['training'] * unit, legend=False))
                name += f'_{RUN_COSTS[training_cost][0]}'
            plots.append(run_cost_bars(timed, family_scale, models, width=MERGED_WIDTHS['inference'] * unit,
                                       legend=False))
            legend = bar_style_legend(latency_legend_groups(timed, family_scale), max_width=MERGED_FIGURE_WIDTH)
            widths = [MERGED_WIDTHS[p] * unit for p in parts]
            chart = alt.vconcat(legend, subplot_letters(plots, widths, 3 * PANEL_SPACING), spacing=2)
            name += '_and_inference_per_timestamp'
        chart = paper_style(chart, DEPLOYMENT_FONT_SIZE_INCREASE)
        st.altair_chart(chart, width='content')
        export_buttons(chart, name, 'cost_latency')
    else:
        st.info('No inference time for these runs.')

    st.subheader('Context graph of the subsets', divider='gray')
    graphs['density'] = graphs['num_edges'] / (graphs['num_nodes'] * (graphs['num_nodes'] - 1))
    graphs['graph_mb'] = graphs['num_edges'] * 3 * 8 / BYTES_PER_MB
    graphs['covariance_mb'] = (graphs['num_nodes'] * graphs['num_node_features']) ** 2 * 8 / BYTES_PER_MB
    st.caption('Built once, offline, from the names of the nodes (their service); rebuilt only when the topology '
               'changes.')
    if times is not None:
        if not merged:   # else next to the inference chart
            st.markdown(GRAPH_BUILD_CAPTION)
            chart = paper_style(graph_build_chart(times), DEPLOYMENT_FONT_SIZE_INCREASE)
            st.altair_chart(chart, width='content')
            export_buttons(chart, 'deployment_graph_construction_time', 'cost_graph_build')
            st.caption(graph_times_caption(times))
        naive, grouped = GRAPH_BUILDS.values()
        graphs = graphs.merge(times[['subset', naive, grouped, 'speedup', 'naive_dense_matrix_mb']], on='subset',
                              how='left')
    st.dataframe(graphs.drop(columns='subset'), hide_index=True, column_config={
        'subset_name': st.column_config.Column('Subset'),
        'num_nodes': st.column_config.NumberColumn('Nodes N', format='%d'),
        'num_node_features': st.column_config.NumberColumn('Features F', format='%d'),
        'num_edges': st.column_config.NumberColumn('Edges E', format='%d'),
        'density': st.column_config.NumberColumn('Density', format='%.5f'),
        'graph_mb': st.column_config.NumberColumn('Graph (MB)', format='%.2f'),
        'covariance_mb': st.column_config.NumberColumn('Mahalanobis covariance (MB)', format='%.1f'),
        'build_time_naive_mean_s': st.column_config.NumberColumn('Build naive (s)', format='%.2f'),
        'build_time_grouped_mean_s': st.column_config.NumberColumn('Build grouped (s)', format='%.4f'),
        'speedup': st.column_config.NumberColumn('Speed-up', format='%.0f×'),
        'naive_dense_matrix_mb': st.column_config.NumberColumn('Naive N×N matrix (MB)', format='%.0f')})

    st.subheader('Training time', divider='gray')
    trained = runs.dropna(subset=['training_time'])
    train_windows = sorted(int(w) for w in trained['window'].unique())
    col_cost, col_windows = st.columns([1, 2])
    training_cost = col_cost.segmented_control('Time', ['Training time', 'Training time per epoch'],
                                               default='Training time per epoch', required=True, key='cost_training')
    shown_windows = col_windows.multiselect(
        'Window sizes', train_windows, key='cost_training_windows', help='Window sizes of the bars of this chart only.',
        default=[LATENCY_LABEL_WINDOW] if LATENCY_LABEL_WINDOW in train_windows else train_windows)
    trained = trained[trained['window'].isin(shown_windows)]
    if trained.empty:
        st.info('Choose at least one window size.' if train_windows else 'No training time for these runs.')
    else:
        st.caption('Training time of each run as measured during its training (its <model>_training_time.csv). '
                   'Per epoch: that time over the epochs the run trained (the length of its training history).')
        chart = paper_style(run_cost_bars(trained, family_scale, models, cost=training_cost),
                            DEPLOYMENT_FONT_SIZE_INCREASE)
        st.altair_chart(chart, width='content')
        export_buttons(chart, f'deployment_{RUN_COSTS[training_cost][0]}', 'cost_training')

    st.subheader('Cost against accuracy', divider='gray')
    cost = st.segmented_control('Cost', list(RUN_COSTS), default='Inference per timestamp', required=True,
                                key='cost_axis')
    st.caption('Marker: the subset of the run; filled: graph-based model, hollow: the other models.')
    chart = cost_accuracy_chart(runs, family_scale, cost, measure)
    st.altair_chart(chart, width='content')
    export_buttons(chart, f'deployment_{RUN_COSTS[cost][0]}_vs_{column}', 'cost_pareto')

    st.subheader('Ensemble of single subsets vs. one model on the combined subset', divider='gray')
    rows = ensemble_cost_rows(runs)
    if rows.empty:
        st.caption('No combined subset (e.g. 5xx 4xx count) with the runs of its single subsets (same model, window '
                   'and fill NaN) among the filtered runs.')
    else:
        ensemble_cost = st.segmented_control('Cost', list(RUN_COSTS), default='Inference per timestamp',
                                             required=True, key='ensemble_cost_axis')
        st.caption('The cost of an ensemble is the sum of those of its members: its fusion (an OR of their labels '
                   'per timestamp) costs nothing next to them. The accuracy of the ensembles is not in these '
                   'results (src/run_ensemble_approach_new.py).')
        chart = ensemble_cost_chart(rows, ensemble_cost)
        st.altair_chart(chart, width='content')
        export_buttons(chart, f'deployment_ensemble_{RUN_COSTS[ensemble_cost][0]}', 'cost_ensemble')
        with st.expander('Values'):
            st.dataframe(rows[['comparison', 'approach', 'part', *[c for c, _, _ in RUN_COSTS.values()], 'best']],
                         hide_index=True, column_config={'best': st.column_config.NumberColumn(f'Best {measure}')})

    st.subheader('Estimated memory', divider='gray')
    windows = sorted(runs['window'].unique())
    window = st.selectbox('Window size', windows, index=len(windows) - 1, key='cost_memory_window')
    st.caption('Analytical estimate, not measured: model parameters (size of the saved model), the input window '
               '(W×N×C float32, C = features plus the is_nan channel of the null-padded models), the context graph '
               '(graph-based models: 3×E float64) and the covariance of the Mahalanobis strategies ((N·F)² float64).')
    chart = memory_chart(runs[runs['window'] == window], models)
    st.altair_chart(chart, width='content')
    export_buttons(chart, f'deployment_memory_w{window}', 'cost_memory')

    st.subheader('All runs', divider='gray')
    table = runs[['model', 'subset_name', 'window', 'fill_nan', 'num_nodes', 'num_edges', 'epochs', 'training_time',
                  'epoch_time', 'inference_time', 'test_timestamps', 'latency_ms', 'realtime_factor',
                  *MEMORY_PARTS.values(), 'memory_mb', 'best']]   # MEMORY_PARTS starts with model_mb
    st.dataframe(table, hide_index=True, column_config={
        'subset_name': st.column_config.Column('Subset'),
        'training_time': st.column_config.NumberColumn('Training (s)', format='%.0f'),
        'epoch_time': st.column_config.NumberColumn('Per epoch (s)', format='%.1f'),
        'inference_time': st.column_config.NumberColumn('Test inference (s)', format='%.1f'),
        'latency_ms': st.column_config.NumberColumn('Per timestamp (ms)', format='%.3f'),
        'realtime_factor': st.column_config.NumberColumn('× below interval', format='%.0f'),
        'model_mb': st.column_config.NumberColumn('Model (MB)', format='%.3f'),
        'window_mb': st.column_config.NumberColumn('Input window (MB)', format='%.2f'),
        'graph_mb': st.column_config.NumberColumn('Graph (MB)', format='%.2f'),
        'covariance_mb': st.column_config.NumberColumn('Covariance (MB)', format='%.1f'),
        'memory_mb': st.column_config.NumberColumn('Memory est. (MB)', format='%.1f'),
        'best': st.column_config.NumberColumn(f'Best {measure}', format=f'%{MEASURES[measure][2]}')})
    st.download_button('Download as CSV', table.to_csv(index=False), file_name='deployment_cost.csv',
                       mime='text/csv', icon=':material/download:')


# OR ensembles of the subsets of a model (src/ensemble_subsets.py): the members share the scoring strategy and the
# threshold type; ensemble_results.csv (exact, written on the HPC) or window_union_preview.csv (from the merged
# table alone: detected windows exact, the false positives only bounded)
ENSEMBLE_DIR = MERGED_DIR / 'ensemble_subsets'
ENSEMBLE_SOURCES = {'Exact (ensemble_results.csv)': 'ensemble_results.csv',
                    'Preview (window_union_preview.csv)': 'window_union_preview.csv'}
ENSEMBLE_SELECTIONS = {'train_valid': 'Train + valid NAB Low FN (no test label)', 'test': 'Test NAB Low FN (oracle)'}
ENSEMBLE_VOTES = {'any': 'Any (OR)', 'majority': 'Majority'}   # src/ensemble_subsets.py VOTES
ENSEMBLE_KEY = ['post_processing_strategy', 'threshold_type']
# Measures of each source: name -> (column, axis title, number format)
ENSEMBLE_MEASURES = {
    'ensemble_results.csv': {
        NAB_LOW_FN: ('reward_fn_normalized', f'{NAB_LOW_FN} (normalized)', '.2f'),
        NAB_STANDARD: ('standard_normalized', f'{NAB_STANDARD} (normalized)', '.2f'),
        'F1': ('f1', 'Test F1', '.3f'),
        'Detected anomalies': ('windows_detected', 'Detected anomaly windows', 'd'),
        'NAB false positives': ('nab_fp', 'NAB false positives', 'd')},
    'window_union_preview.csv': {
        'Detected anomalies': ('windows_detected', 'Detected anomaly windows', 'd'),
        'NAB false positives (upper bound)': ('nab_fp_max', 'NAB false positives (≤)', 'd'),
        'NAB false positives (lower bound)': ('nab_fp_min', 'NAB false positives (≥)', 'd')},
}
ENSEMBLE_COUNT_COLUMNS = ('nab_fp', 'nab_fp_max', 'nab_fp_min')   # drawn on a symlog scale (up to tens of thousands)


@st.cache_data(show_spinner='Loading ensemble results…', max_entries=6)
def load_ensemble_file(path, mtime):
    """A CSV of src/ensemble_subsets.py; mtime refreshes the cache."""
    return pd.read_csv(path)


def ensemble_scoring_label(strategy, threshold_type):
    """('likelihood', 'percentile') -> 'LF · percentile'."""
    return f'{STRATEGY_SHORT_NAMES.get(strategy, strategy)} · {threshold_type}'


def ensemble_size_chart(combos, column, title, fmt, family_scale):
    """Every combination (dots) and the best one of each size (tick) against the number of subsets ensembled,
    one panel per scoring (row) and window (column), the models side by side."""
    data = combos.assign(size=combos['size'].astype(str))
    sizes = [str(s) for s in sorted(combos['size'].unique())]
    y_scale = alt.Scale(type='symlog') if column in ENSEMBLE_COUNT_COLUMNS else alt.Scale(zero=False, nice=False, padding=6)
    # x title once, in the column header: repeated under every panel, the titles overlapped
    x = alt.X('size:O', title=None, sort=sizes, axis=alt.Axis(labelAngle=0, tickSize=3))
    models = list(dict.fromkeys(data.sort_values('model_folder', key=lambda s: s.map(model_order_key))['model']))
    family_of = dict(zip(data['model'], data['model_family']))
    colors = family_colors(family_scale)
    color = alt.Color('model:N', title='Model', sort=models, scale=alt.Scale(
        domain=models, range=[colors.get(family_of[m], NO_FAMILY_COLOR) for m in models]),
                      legend=alt.Legend(orient='top', labelLimit=0, strokeColor='transparent', symbolOpacity=1))
    base = alt.Chart().encode(x=x, xOffset=alt.XOffset('model:N', sort=models), color=color)
    dots = base.mark_circle(size=14, opacity=0.35).encode(
        y=alt.Y(f'{column}:Q', title=title, scale=y_scale, axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),
        tooltip=['model', 'scoring', 'window', 'size', 'combination',
                 alt.Tooltip(f'{column}:Q', title=title, format=fmt)])
    best = base.transform_aggregate(best=f'max({column})', groupby=['model', 'model_family', 'scoring', 'window', 'size']
                                    ).mark_tick(thickness=2, size=10).encode(
        y=alt.Y('best:Q', scale=y_scale), tooltip=['model', 'scoring', 'window', 'size',
                                                   alt.Tooltip('best:Q', title=f'Best {title}', format=fmt)])
    panel_width = max(70, 13 * len(sizes) * len(models))
    chart = alt.layer(dots, best, data=data).properties(width=panel_width, height=150).facet(
        row=alt.Row('scoring:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left')),
        column=alt.Column('window:O', title='Sliding window (x: subsets in the ensemble)'), spacing=PANEL_SPACING)
    return paper_style(chart)


def render_subset_ensembles(family_scale):
    """
    'Subset ensembles' tab: OR ensembles of 1 (the subsets alone), 2, 3 ... subsets of a model, sharing the scoring
    strategy and threshold type, from the files of src/ensemble_subsets.py. Its filters are its own.
    """
    st.caption('This tab has its own filters: the filters of the sidebar do not apply here. Members of an ensemble '
               'share the scoring strategy and the threshold type; each subset keeps its own threshold, top-k, '
               'likelihood windows and imputation. A timestamp is flagged when any member flags it (OR), or more '
               'than half of them (majority vote).')
    available = {name: ENSEMBLE_DIR / file for name, file in ENSEMBLE_SOURCES.items() if (ENSEMBLE_DIR / file).exists()}
    if not available:
        st.info(f'No ensemble file in `{os.path.relpath(ENSEMBLE_DIR, PROJECT_ROOT)}/`: write the preview locally with '
                '`python src/ensemble_subsets.py --select-only`, the exact results on the HPC server with '
                '`sbatch slurm_script_ensemble_subsets.sh`.')
        return
    source = st.segmented_control('Results', list(available), default=next(iter(available)), required=True,
                                  key='ensemble_source',
                                  help='Exact: NAB / F1 of the OR of the saved test predictions (HPC). Preview: from '
                                       'the merged table alone, detected windows exact (union of those of the '
                                       'members), false positives only bounded (max .. sum of those of the members).')
    path = available[source]
    combos = load_ensemble_file(str(path), os.path.getmtime(path))
    if 'vote' not in combos:   # written before the votes: OR ensembles only
        combos = combos.assign(vote='any', min_votes=1)
    combos = combos.assign(model=combos['model_folder'].map(short_model_name),
                           model_family=combos['model_folder'].map(model_family),
                           scoring=[ensemble_scoring_label(s, t) for s, t in
                                    zip(combos['post_processing_strategy'], combos['threshold_type'])])
    st.caption(f':material/description: `{os.path.relpath(path, PROJECT_ROOT)}`: {len(combos):,} combinations, '
               f'modified {datetime.fromtimestamp(os.path.getmtime(path)):%Y-%m-%d %H:%M}')

    col_model, col_select, col_norm, col_vote = st.columns(4)
    folders = sorted(combos['model_folder'].unique(), key=model_order_key)
    chosen_folders = col_model.multiselect('Models', folders, default=folders, format_func=short_model_name,
                                           key='ensemble_models')
    selections = [s for s in ENSEMBLE_SELECTIONS if s in set(combos['selection'])]
    selection = col_select.selectbox('Setting of each subset chosen on', selections, format_func=ENSEMBLE_SELECTIONS.get,
                                     key='ensemble_selection',
                                     help='Best grid-search setting of each subset within the scoring strategy and '
                                          'threshold type. On the test split, the results are an oracle.')
    normalizations = sorted(combos['score_normalization'].unique())
    normalization = col_norm.segmented_control('Score normalization', normalizations, default=normalizations[0],
                                               required=True, key='ensemble_normalization')
    votes = [v for v in ENSEMBLE_VOTES if v in set(combos['vote'])]
    vote = col_vote.segmented_control('Vote', votes, default=votes[0], required=True, format_func=ENSEMBLE_VOTES.get,
                                      key='ensemble_vote',
                                      help='Members flagging a timestamp for the ensemble to flag it: at least one '
                                           '(OR), or more than half (2 of 2, 2 of 3, 3 of 4, 5 of 8). One subset '
                                           'alone is the same for both.')
    col_scoring, col_window, col_size, col_measure = st.columns(4)
    scorings = sorted(combos['scoring'].unique())
    default_scorings = [s for s in scorings if s.startswith(f'{STRATEGY_SHORT_NAMES["likelihood"]} ·')] or scorings[:1]
    chosen_scorings = col_scoring.multiselect('Scoring · threshold type', scorings, default=default_scorings,
                                              key='ensemble_scorings')
    windows = sorted(combos['window'].unique())
    chosen_windows = col_window.multiselect('Windows', windows, default=windows, key='ensemble_windows')
    sizes = sorted(combos['size'].unique())
    chosen_sizes = col_size.multiselect('Subsets per ensemble', sizes, default=sizes, key='ensemble_sizes')
    measures = ENSEMBLE_MEASURES[path.name]
    measure = col_measure.selectbox('Measure', list(measures), key=f'ensemble_measure_{path.name}')
    column, title, fmt = measures[measure]

    shown = combos[combos['model_folder'].isin(chosen_folders) & (combos['selection'] == selection)
                   & (combos['score_normalization'] == normalization) & combos['scoring'].isin(chosen_scorings)
                   & combos['window'].isin(chosen_windows) & combos['size'].isin(chosen_sizes)
                   & ((combos['vote'] == vote) | (combos['size'] == 1))]   # the subsets alone: the reference
    if shown.empty:
        st.info('No combination matches these filters.')
        return

    st.subheader(f'{measure} against the number of subsets', divider='gray')
    st.caption('Dots: every combination of subsets; tick: the best one of each size. One subset: the subsets alone, '
               'their best tick is the best single subset.' +
               (' False positives on a symlog scale.' if column in ENSEMBLE_COUNT_COLUMNS else ''))
    chart = ensemble_size_chart(shown, column, title, fmt, family_scale)
    st.altair_chart(chart, width='content')
    export_buttons(chart, f'subset_ensembles_{column}_{selection}_{normalization}_{vote}', 'ensemble_sizes')

    # Best combination of each size against the best subset alone (same model, scoring, window)
    keys = ['model', 'scoring', 'window']
    best = shown.loc[shown.groupby(keys + ['size'])[column].idxmax()]
    singles = combos[(combos['size'] == 1) & (combos['selection'] == selection)
                     & (combos['score_normalization'] == normalization)]
    best_single = singles.groupby(['model_folder', 'scoring', 'window'])[column].max().rename('best_single')
    best = best.join(best_single, on=['model_folder', 'scoring', 'window'])
    best['gain'] = best[column] - best['best_single']
    st.subheader('Best combination of each size', divider='gray')
    st.caption(f'Gain: {measure} of the best combination minus that of the best subset alone (same model, scoring '
               'and window).' + (' Higher false positives are worse.' if column in ENSEMBLE_COUNT_COLUMNS else ''))
    # The gain of one subset is 0 by definition: its value is shown instead, the reference of the gains
    gains = best[best['size'] > 1].pivot_table(index=keys, columns='size', values='gain')
    gains.columns = [f'+{s} subsets' for s in gains.columns]
    gains.insert(0, 'Best subset alone', best.groupby(keys)['best_single'].first())
    st.dataframe(gains.reset_index(), hide_index=True,
                 column_config={'Best subset alone': st.column_config.NumberColumn(format=f'%{fmt}'),
                                **{c: st.column_config.NumberColumn(format=f'%+{fmt}' if fmt != 'd' else '%+d')
                                   for c in gains.columns if c != 'Best subset alone'}})
    table_columns = [*keys, 'size', 'min_votes', 'combination', column, 'best_single', 'gain',
                     *[c for c, _, _ in measures.values() if c != column],
                     *[c for c in ('precision', 'recall', 'detected_window_ids') if c in best.columns]]
    with st.expander('Best combinations'):
        st.dataframe(best[table_columns].sort_values(keys + ['size']), hide_index=True,
                     column_config={column: st.column_config.NumberColumn(measure, format=f'%{fmt}'),
                                    'best_single': st.column_config.NumberColumn('Best subset alone', format=f'%{fmt}'),
                                    'gain': st.column_config.NumberColumn('Gain', format=f'%+{fmt}')})

    st.subheader('All combinations', divider='gray')
    ordered = shown.sort_values(column, ascending=column in ENSEMBLE_COUNT_COLUMNS)
    all_columns = ['model', 'scoring', 'window', 'size', 'min_votes', 'combination',
                   *[c for c, _, _ in measures.values()],
                   *[c for c in ('precision', 'recall', 'num_windows', 'detected_window_ids', 'check_diff')
                     if c in shown.columns]]
    st.dataframe(ordered[all_columns], hide_index=True, height=400)
    st.download_button('Download as CSV', ordered[all_columns].to_csv(index=False), icon=':material/download:',
                       file_name=f'subset_ensembles_{selection}_{normalization}_{vote}.csv', mime='text/csv')

    selected_path = ENSEMBLE_DIR / 'selected_configs.csv'
    if selected_path.exists():
        with st.expander('Setting chosen for each subset'):
            configs = load_ensemble_file(str(selected_path), os.path.getmtime(selected_path))
            configs = configs.assign(model=configs['model_folder'].map(short_model_name),
                                     scoring=[ensemble_scoring_label(s, t) for s, t in
                                              zip(configs['post_processing_strategy'], configs['threshold_type'])])
            configs = configs[configs['model_folder'].isin(chosen_folders) & (configs['selection'] == selection)
                              & (configs['score_normalization'] == normalization)
                              & configs['scoring'].isin(chosen_scorings) & configs['window'].isin(chosen_windows)]
            st.dataframe(configs[['model', 'scoring', 'window', 'subset', 'fill_nan', 'topk', 'long_window',
                                  'short_window', 'anomaly_threshold', 'train_valid_reward_fn_normalized',
                                  'reward_fn_normalized', 'standard_normalized', 'f1', 'detected_window_ids', 'nab_fp']]
                         .sort_values(['model', 'scoring', 'window', 'subset']), hide_index=True,
                         column_config={'train_valid_reward_fn_normalized': st.column_config.NumberColumn(
                             f'Train + valid {NAB_LOW_FN}', format='%.2f'),
                             'reward_fn_normalized': st.column_config.NumberColumn(f'Test {NAB_LOW_FN}', format='%.2f'),
                             'standard_normalized': st.column_config.NumberColumn(f'Test {NAB_STANDARD}', format='%.2f')})


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
            default=[m for m in (NAB_LOW_FN, 'AUC-PR', 'VUS-PR') if m in available_measures],
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

    (tab_overview, tab_compare, tab_target, tab_windows, tab_rows, tab_table, tab_bubbles, tab_cost,
     tab_ensembles) = st.tabs(
        ['Overview', 'Compare', 'Target model', 'Strategies by window', 'All settings', 'Results table',
         'Likelihood windows', 'Deployment cost', 'Subset ensembles'])

    # ── Results table and likelihood windows: their own filters, shown whatever the sidebar filters ──
    with tab_table:
        render_results_table(results, family_scale, options)
    with tab_bubbles:
        render_likelihood_windows(results, options, family_scale)
    with tab_cost:
        render_deployment_cost(results, family_scale)
    with tab_ensembles:
        render_subset_ensembles(family_scale)

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
# Plots of a one-column figure: the y axis (labels and the measure as title, estimated) on their left
COLUMN_PLOT_HEIGHT = 110   # the plots of a one-column figure side by side, one per measure


def column_plot_width(columns):
    """Width of the plots of a one-column figure with `columns` plots side by side, each its y axis (estimated)
    on its left and its measure as title above."""
    return ((COLUMN_FIGURE_WIDTH_INCHES * POINTS_PER_INCH - PANEL_SPACING * (columns - 1)) / columns
            - PANEL_AXIS_SPACE)


def panel_width(columns, axis_space=PANEL_AXIS_SPACE):
    """Width of the panels of a figure FIGURE_WIDTH wide with `columns` panels per row, the measure on their left
    and a y axis axis_space wide (estimated) on the left of each."""
    return ((FIGURE_WIDTH - PANEL_TITLE_SIZE - PANEL_TITLE_OFFSET - PANEL_SPACING * (columns - 1)) / columns
            - axis_space)


def paper_style(chart, increase=FONT_SIZE_INCREASE):
    """The text sizes (src/figure_style.py, plus increase points) and frames of a figure of the paper."""
    return (chart.configure_view(stroke=CHART_BORDER, strokeWidth=1)   # a border around each plot
            .configure_axis(labelFontSize=TICK_FONT_SIZE + increase,
                            titleFontSize=AXIS_LABEL_FONT_SIZE + increase)
            .configure_axisY(labelFontSize=TICK_FONT_SIZE + increase - Y_TICK_FONT_SIZE_DECREASE)
            .configure_legend(labelFontSize=LEGEND_FONT_SIZE + increase,
                              titleFontSize=LEGEND_FONT_SIZE + increase,
                              strokeColor=CHART_BORDER, padding=LEGEND_PADDING)   # a frame around each legend
            .configure_header(labelFontSize=AXIS_LABEL_FONT_SIZE + increase,   # the panel titles
                              labelPadding=2))


def export_buttons(chart, name, key, width_inches=FIGURE_WIDTH_INCHES):
    """PNG and PDF downloads of a chart (a figure width_inches wide), made only when clicked."""
    col_png, col_pdf, _ = st.columns([1, 1, 3])
    col_png.download_button('Export as PNG', lambda: chart_png(chart, width_inches), file_name=f'{name}.png', mime='image/png',
                            key=f'png_{key}', icon=':material/image:', on_click='ignore')
    col_pdf.download_button('Export as PDF', lambda: chart_pdf(chart), file_name=f'{name}.pdf',
                            mime='application/pdf', key=f'pdf_{key}', icon=':material/picture_as_pdf:',
                            on_click='ignore', help='Vector figure (sharp at any zoom), for LaTeX')


def chart_pdf(chart):
    """PDF bytes of an Altair chart: vector, its units in points, so FIGURE_WIDTH_INCHES wide as it is."""
    buffer = io.BytesIO()
    chart.save(buffer, format='pdf')
    return buffer.getvalue()


def chart_png(chart, width_inches=FIGURE_WIDTH_INCHES):
    """PNG bytes of an Altair chart at PNG_DPI, its resolution set so that it is width_inches wide."""
    buffer = io.BytesIO()
    chart.save(buffer, format='png', scale_factor=PNG_DPI / POINTS_PER_INCH)
    image = Image.open(io.BytesIO(buffer.getvalue()))
    dpi = image.width / width_inches   # PNG_DPI, up to the width of the panel axes (estimated)
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
    solid with the sparsity indicator (*) and dashed without it, the other models solid; the lines of a combined
    subset (grid_results.COMBINED_SUBSETS) dotted (COMBINED_SUBSET_DASH). One panel per group of rows of
    facet_dims (Group rows by, e.g. the subset); or the groups of overlay_dims all in the same chart, told apart by
    their dash (GROUP_DASHES: solid a single subset, dotted a combined one), the charts of the measures side by side
    in a one-column figure of the paper with their legends on top. The series that still share a color and a dash
    are told apart by their point shape.
    values: the cells of the table, one row per (row, column) key, one column per measure.
    """
    overlay_dims = list(overlay_dims)
    data = values.reset_index()
    order_key = data['model_folder'].map(model_order_key) if 'model_folder' in data else pd.Series(0, index=data.index)
    family = data['model_folder'].map(model_family) if 'model_folder' in data else pd.Series('', index=data.index)
    dashed = (data['model_folder'].map(lambda f: short_model_name(f).startswith(PROPOSED_MODEL_PREFIX)
                                       and not NULL_PADDING_SUFFIX.search(str(f)))
              if 'model_folder' in data else pd.Series(False, index=data.index))
    combined = (data['subset'].map(lambda s: bool(component_subsets(str(s))))
                if 'subset' in data else pd.Series(False, index=data.index))
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
    data['_dash'] = [COMBINED_SUBSET_DASH if c else LINE_DASHES[d] for d, c in zip(dashed, combined)]
    first = data.assign(_order=order_key, _family=family, _dashed=dashed).sort_values(['_order', 'series'],
                                                                                      kind='stable')
    first = first.drop_duplicates('series')
    colors = family_colors(family_scale)
    series = first['series'].tolist()
    rank = first.groupby(['_family', '_dashed']).cumcount()
    legend_domain, legend_columns = family_legend_layout(first['series'], first['_family'])
    color_of = dict(zip(series, [colors.get(f, NO_FAMILY_COLOR) for f in first['_family']]))
    dash_of = dict(zip(series, first['_dash']))
    # The subset among the series: the dash of each series (in its legend); otherwise (the subset in the panels /
    # groups) a series has lines of combined and of single subsets: the dash of each line, no dash in the legend
    dash_by_line = combined.any() and 'subset' not in series_dims
    data['panel_line'] = data['panel'] + ' · ' + data['line']   # a line is drawn in every panel
    line_dash_of = dict(zip(data['panel_line'], data['_dash']))
    shape_of = dict(zip(series, [STRATEGY_SHAPES[r % len(STRATEGY_SHAPES)] for r in rank]))
    color_scale = alt.Scale(domain=legend_domain, range=[color_of.get(s, 'transparent') for s in legend_domain])
    dash_scale = alt.Scale(domain=legend_domain, range=[dash_of.get(s, LINE_DASHES[False]) for s in legend_domain])
    panels = list(dict.fromkeys(data['panel']))
    groups = list(dict.fromkeys(data['group']))
    long = data.melt(id_vars=['series', 'panel', 'group', 'line', 'panel_line', 'window'], value_vars=measures,
                     var_name='measure')
    # Legends compact (e.g. for a figure of a paper) above the chart: the series, then the groups drawn in one chart
    legend = alt.Legend(orient='top', direction='horizontal', columns=legend_columns, labelLimit=0,
                        symbolStrokeWidth=1.5, symbolSize=40, symbolOpacity=1, columnPadding=8, rowPadding=1,
                        titleOrient='left', titleAnchor='start', titleBaseline='top',   # title in the top left corner
                        offset=LEGEND_OFFSET)
    if overlay_dims:   # one-column figure: the legend titles (Model, Subset) on a row of their own above the entries
        legend = legend.copy()
        legend.titleOrient = 'top'
        legend.strokeColor = 'transparent'   # no frame (paper_style frames the legends)
    title = ' · '.join(DIMENSIONS[d] for d in series_dims) or None
    shape = alt.Shape('series:N', scale=alt.Scale(domain=legend_domain,
                                                  range=[shape_of.get(s, 'circle') for s in legend_domain]),
                      title=title, legend=legend)
    tooltip = ['series', 'group' if overlay_dims else 'panel', 'window']
    if overlay_dims:
        # The dash tells apart the groups, the same for every series (the series by their color and point shape):
        # solid a single subset, dotted a combined one
        group_combined = data.assign(_combined=combined).groupby('group', sort=False)['_combined'].any()
        group_rank = group_combined.groupby(group_combined).cumcount()
        group_dash = alt.StrokeDash('group:N', title=' · '.join(DIMENSIONS[d] for d in overlay_dims),
                                    scale=alt.Scale(domain=groups, range=[
                                        GROUP_DASHES[group_combined[g]][group_rank[g] % len(GROUP_DASHES[False])]
                                        for g in groups]),
                                    legend=alt.Legend(orient='top', direction='vertical', labelLimit=0,
                                                      symbolType='stroke', symbolStrokeWidth=1.5, symbolSize=300,
                                                      symbolStrokeColor='black', columnPadding=8, rowPadding=1,
                                                      titleOrient='top', titleAnchor='start', strokeColor='transparent',
                                                      offset=LEGEND_OFFSET))
    charts = []
    for m in measures:
        fmt = MEASURES[m][2]
        encodings = dict(
            x=alt.X('window:O', sort=windows, title='Window size', axis=alt.Axis(labelAngle=0, domain=False, tickSize=3, labelPadding=2)),
            # padding: the top / bottom grid lines inside the border instead of over it
            # panels: the measure once on the left of all of them (the chart title) instead of on every y axis;
            # groups in one chart: the measure as title above each plot (side by side in one column)
            y=alt.Y('value:Q', title=None if facet_dims or overlay_dims else m,
                    scale=alt.Scale(zero=False, nice=False, padding=8),
                    axis=alt.Axis(tickSize=3, labelPadding=2, domain=False)),   # the border as axis line
            color=alt.Color('series:N', scale=color_scale, title=title, legend=legend),
            tooltip=[*tooltip, alt.Tooltip('value:Q', title=m, format=fmt)])
        if overlay_dims:
            stroke_dash = group_dash
        elif dash_by_line:
            stroke_dash = alt.StrokeDash('panel_line:N', legend=None, scale=alt.Scale(
                domain=list(line_dash_of), range=list(line_dash_of.values())))
        else:
            stroke_dash = alt.StrokeDash('series:N', scale=dash_scale, title=title, legend=legend)
        layers = alt.layer(
            alt.Chart().mark_line(strokeWidth=1.5).encode(strokeDash=stroke_dash, detail='line:N', **encodings),
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
        elif overlay_dims:   # side by side in one figure: concatenated charts cannot fit a width (autosize)
            chart = layers.properties(width=column_plot_width(len(measures)), height=COLUMN_PLOT_HEIGHT,
                                      title=alt.TitleParams(m, fontSize=PANEL_TITLE_SIZE, fontWeight='bold',
                                                            offset=PANEL_TITLE_OFFSET))
        else:
            chart = layers.properties(width=FIGURE_WIDTH, height=FIGURE_WIDTH * CHART_ASPECT,
                                      autosize=alt.AutoSizeParams(type='fit-x', contains='padding'))
        charts.append(chart)
    if overlay_dims:   # one figure, one chart per measure side by side, their legends shared (same scales)
        figure = alt.hconcat(*charts, spacing=PANEL_SPACING).resolve_scale(
            y='independent', color='shared', strokeDash='shared', shape='shared'
        ).resolve_legend(color='shared', strokeDash='shared', shape='shared')
        return [paper_style(figure)]
    return [paper_style(chart) for chart in charts]


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
                                        default=[m for m in ('F1', NAB_STANDARD, NAB_LOW_FN)
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
                help='One panel per group of rows, or the lines of every group in the same chart (one dash per '
                     'group: solid a single subset, dotted a combined one), the measures side by side in a figure '
                     'one column of the paper wide.')
            if grouping == CHART_GROUPINGS[1]:
                panel_dims, overlay_dims = [], group_dims
        charts = window_line_charts(values, measures, series_dims, panel_dims, family_scale, short_names,
                                    overlay_dims)
        if overlay_dims:   # one figure of all the measures, one column of the paper wide
            st.altair_chart(charts[0], width='content')
            export_buttons(charts[0], 'results_by_window', 'all_measures', COLUMN_FIGURE_WIDTH_INCHES)
        for measure, chart in zip(measures, [] if overlay_dims else charts):
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
