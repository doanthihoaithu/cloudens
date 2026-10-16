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
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from grid_results import (PROJECT_ROOT, TAG_COLUMNS, MERGED_DIR, MERGED_SUFFIXES, find_grid_files, grid_signature,
                          read_grid_files, read_merged, write_merged, merged_bytes, find_merged_files,
                          default_merged_path)

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
# Whole folder names shown as given: Anomaly Transformer with the association-based score of its paper as the
# model itself, the forecast-error variant (folder without suffix) tagged
FOLDER_SHORT_NAMES = {'AnomalyTransformer_association': 'Anom.Trans.', 'AnomalyTransformer': 'Anom.Trans.-fc'}
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


def dimension_options(results):
    """Filter options of every dimension (its values, sorted) and whether it has missing values."""
    return {column: (sorted(results[column].dropna().unique().tolist(), key=str), bool(results[column].isna().any()))
            for column in DIMENSIONS}


def filter_rows(results, options, chosen):
    """Rows whose dimensions take chosen values. A dimension with all its values chosen and no missing value
    restricts nothing: its (costly, on millions of rows) isin test is skipped, and results is returned as is
    when no dimension restricts it."""
    mask = None
    for column, values in chosen.items():
        all_values, has_missing = options[column]
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
    the F1 rank of the own absolute thresholds, the NAB detections, the top-k labels and model families."""
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
    return results


# Source of the code reading and shaping the rows, passed to every cached loader (its cache key holds the
# arguments given, not the defaults): st.cache_data only hashes the source of the cached function itself, so without it a change to prepare_results (picked up by
# runOnSave) would keep serving the rows prepared by the previous version until the cache is cleared
LOADING_CODE = hashlib.md5(''.join(inspect.getsource(f) for f in (
    prepare_results, model_variant, model_family, _topk_label, _parse_counters, _detection_summary,
    read_grid_files, read_merged)).encode()).hexdigest()


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
def load_merged_upload(data, name, loading_code):
    """Rows of an uploaded merged file (its bytes)."""
    return prepare_results(read_merged(io.BytesIO(data), name))


# ── Views ─────────────────────────────────────────────────────────────────────

def render_explorer(results):
    """Filters (sidebar) and tabs of the explorer over the rows of a merged table, whatever its source."""
    family_scale = family_color_scale(results['model_family'].unique())

    # ── Filters ──
    with st.sidebar:
        st.header('Filters')
        options = dimension_options(results)
        selected = {column: st.multiselect(label, options[column][0], default=options[column][0])
                    for column, label in DIMENSIONS.items()}

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

    filtered = filter_rows(results, options, selected)

    tab_overview, tab_compare, tab_target, tab_windows, tab_rows, tab_table = st.tabs(
        ['Overview', 'Compare', 'Target model', 'Strategies by window', 'All settings', 'Results table'])

    # ── Results table: its own filters, shown whatever the sidebar filters ──
    with tab_table:
        render_results_table(results, family_scale, options)

    if filtered.empty:
        for tab in (tab_overview, tab_compare, tab_target, tab_windows, tab_rows):
            tab.info('No grid-search row matches the filters of the sidebar.')
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
        st.caption(f'{len(filtered):,} settings match the filters (sorted by {measure_title}).')
        leading = list(DIMENSIONS) + ['topk', 'long_window', 'short_window', 'anomaly_threshold',
                                      'reward_fn_normalized', 'standard_normalized', *DETECTION_COLUMNS,
                                      'nab_tp', 'nab_fp', 'nab_fn', 'precision', 'recall', 'f1']
        leading = [c for c in leading if c in filtered.columns]
        others = [c for c in filtered.columns if c not in leading and c not in ('detection_counters',)]
        table = filtered[leading + others].sort_values(measure_column, ascending=False)
        st.dataframe(table, hide_index=True, height=600, column_config=DETECTION_COLUMN_CONFIG)
        # The CSV of up to millions of rows is written on click only, not on every rerun
        st.download_button('Download as CSV', lambda: table.to_csv(index=False).encode(),
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
        chosen = {column: filter_columns[i % 4].multiselect(label, options[column][0], default=options[column][0],
                                                            key=f'table_filter_{column}')
                  for i, (column, label) in enumerate(DIMENSIONS.items())}
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
            load, args, label = load_merged_upload, (uploaded.getvalue(), uploaded.name, LOADING_CODE), uploaded.name
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
