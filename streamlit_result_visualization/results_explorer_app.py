"""
Streamlit explorer of the grid-search results under a results folder (default: trained_models_log1p).

Layout scanned:
    <root>/window_<w>/<subset>/fill_nan_with_<fill>/<model folder>/<model>_grid_search_<normalization>.csv

Run from the project root:
    streamlit run streamlit_result_visualization/results_explorer_app.py
The app reruns when this file is saved (.streamlit/config.toml next to it: runOnSave) and, with
'Auto-refresh' on, when grid-search files change on disk.
"""
import ast
import os
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # streamlit_result_visualization/ is at the project root
GRID_FILE_PATTERN = re.compile(r'^(?P<model>.+)_grid_search_(?P<normalization>global|per_node)\.csv$')

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
# Groups are colored by model family; Vega-Lite has no hatch patterns, so up to two other group
# dimensions are told apart by the fill shade and by the border style of the bars / boxes
FILL_SHADES = [1.0, 0.55, 0.25, 0.8, 0.4]
BORDER_DASHES = [[1, 0], [6, 3], [2, 2], [8, 3, 2, 3]]


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
            family_colors = dict(zip(family_scale.domain, family_scale.range))
            legend_channels['fill'] = alt.Fill(f'{dim}:N', title=DIMENSIONS[dim], legend=legend, scale=alt.Scale(
                domain=domain, range=[family_colors.get(model_family(m), NO_FAMILY_COLOR) for m in domain]))

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


# ── Loading ───────────────────────────────────────────────────────────────────

def parse_grid_path(path):
    """Experiment of a grid-search CSV, from its location; None when the name does not match."""
    path = Path(path)
    match = GRID_FILE_PATTERN.match(path.name)
    if not match:
        return None
    model_dir = path.parent
    return {
        'path': str(path),
        'window': int(model_dir.parents[2].name.removeprefix('window_')),
        'subset': model_dir.parents[1].name,
        'fill_nan': model_dir.parent.name.removeprefix('fill_nan_with_'),
        'model_folder': model_dir.name,
        'score_normalization': match['normalization'],
        'modified': pd.Timestamp(os.path.getmtime(path), unit='s'),
    }


def find_grid_files(root):
    """Experiment info of every grid-search CSV under root."""
    found = (parse_grid_path(p) for p in sorted(Path(root).glob('window_*/*/fill_nan_with_*/*/*_grid_search_*.csv')))
    return [info for info in found if info is not None]


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


def grid_signature(files):
    """(path, modification time) of every grid-search file: changes when a file is added, removed or rewritten."""
    return tuple((f['path'], f['modified'].value) for f in files)


@st.cache_data(show_spinner='Loading grid-search results…', max_entries=8)
def load_results(files):
    """All grid-search rows, tagged with where they come from. `files`: (path, mtime) pairs, so that the
    cache is refreshed when a file changes."""
    frames = []
    for path, _ in files:
        info = parse_grid_path(path)
        frame = pd.read_csv(path)
        for key in ('window', 'subset', 'fill_nan', 'model_folder', 'score_normalization'):
            frame[key] = info[key]
        frame['file'] = os.path.relpath(path, PROJECT_ROOT)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    # Older grid searches have no threshold_type: their likelihood thresholds were absolute values
    if 'threshold_type' not in results:
        results['threshold_type'] = None
    is_likelihood = results['post_processing_strategy'].str.startswith('likelihood')
    results['threshold_type'] = results['threshold_type'].fillna(
        is_likelihood.map({True: 'absolute', False: 'percentile'}))
    counters = results['detection_counters'].map(_parse_counters)
    for key in ('tp', 'fp', 'fn'):
        results[f'nab_{key}'] = counters.map(lambda c, k=key: c.get(k))
    detections = pd.DataFrame(counters.map(_detection_summary).tolist(), index=results.index)
    results = pd.concat([results, detections[DETECTION_COLUMNS]], axis=1)
    results['topk'] = [_topk_label(k, s) for k, s in zip(results['topk'], results['post_processing_strategy'])]
    results['model_family'] = results['model_folder'].map(model_family)
    return results


# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Grid-search explorer', page_icon=':material/monitoring:', layout='wide')
st.title('Grid-search results explorer')

with st.sidebar:
    st.header('Data')
    candidate_roots = sorted(p.name for p in PROJECT_ROOT.glob('trained_models*') if p.is_dir())
    default_root = 'trained_models_log1p' if 'trained_models_log1p' in candidate_roots else (candidate_roots or [''])[0]
    root_name = st.selectbox('Results folder', candidate_roots or [default_root],
                             index=(candidate_roots or [default_root]).index(default_root))
    if st.button('Rescan folder', icon=':material/refresh:', width='stretch'):
        load_results.clear()
    auto_refresh = st.toggle('Auto-refresh', value=True,
                             help='Reload the app when a grid-search file is added, removed or rewritten.')
    refresh_interval = st.segmented_control('Check every', AUTO_REFRESH_INTERVALS, default='30s', required=True,
                                            disabled=not auto_refresh)

grid_files = find_grid_files(PROJECT_ROOT / root_name)
# Files this run shows; the watcher reruns the app when the folder no longer matches
st.session_state['grid_signature'] = grid_signature(grid_files)


def watch_results_folder():
    """Fragment rerun on a timer: cheap check of the folder, full rerun only when it changed."""
    if grid_signature(find_grid_files(PROJECT_ROOT / root_name)) != st.session_state.get('grid_signature'):
        st.rerun()   # whole app: reloads the changed files (the cache is keyed by their modification time)
    st.caption(f':material/schedule: Last checked {datetime.now():%H:%M:%S}')


with st.sidebar:
    st.fragment(watch_results_folder, run_every=refresh_interval if auto_refresh else None)()

if not grid_files:
    st.warning(f'No `*_grid_search_*.csv` found under `{root_name}`.')
    st.stop()
results = load_results(grid_signature(grid_files))
family_scale = family_color_scale(results['model_family'].unique())

# ── Filters ──
with st.sidebar:
    st.header('Filters')
    selected = {}
    for column, label in DIMENSIONS.items():
        options = sorted(results[column].dropna().unique().tolist(), key=str)
        selected[column] = st.multiselect(label, options, default=options)

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

mask = pd.Series(True, index=results.index)
for column, values in selected.items():
    mask &= results[column].isin(values)
filtered = results[mask]

if filtered.empty:
    st.info('No grid-search row matches the filters.')
    st.stop()

tab_overview, tab_compare, tab_target, tab_rows = st.tabs(['Overview', 'Compare', 'Target model', 'All settings'])

# ── Overview: what has been run ──
with tab_overview:
    cols = st.columns(5)
    for col, key in zip(cols, ('window', 'subset', 'fill_nan', 'model_folder', 'score_normalization')):
        values = sorted(results[key].unique().tolist(), key=str)
        col.metric(DIMENSIONS[key], len(values))
        col.caption(', '.join(map(str, values)))

    st.subheader('Grid-search files')
    inventory = pd.DataFrame(grid_files)
    rows = results.groupby('file').agg(
        settings=('post_processing_strategy', 'size'),
        strategies=('post_processing_strategy', lambda s: ', '.join(sorted(s.unique()))),
        **{f'best_{c}': (c, 'max') for c in measure_columns})
    inventory['file'] = inventory['path'].map(lambda p: os.path.relpath(p, PROJECT_ROOT))
    inventory = inventory.merge(rows, left_on='file', right_index=True)
    inventory['has_split_metrics'] = inventory['file'].map(
        lambda f: bool(results.loc[results['file'] == f].filter(like='train_valid_').notna().any().any()))
    st.dataframe(
        inventory[['window', 'subset', 'fill_nan', 'model_folder', 'score_normalization', 'settings', 'strategies',
                   *[f'best_{c}' for c in measure_columns], 'has_split_metrics', 'modified', 'file']],
        hide_index=True,
        column_config={
            **{f'best_{MEASURES[m][0]}': st.column_config.NumberColumn(f'Best {m}', format=f'%{MEASURES[m][2]}')
               for m in measures},
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
            best = ranked.loc[ranked.groupby(group_by)[select_by].idxmax()].copy()
            best['group'] = best[group_by].astype(str).agg(' · '.join, axis=1)
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
        spread['group'] = spread[group_by].astype(str).agg(' · '.join, axis=1)
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
        # Compared as text: NaN windows (strategies without them) match each other
        keys = ranked[match_on].astype(str)
        same = (keys == keys.loc[target_index]).all(axis=1)
        matched = ranked[same]
        best_matched = matched.loc[matched.groupby('model_folder')[select_by].idxmax()].copy()
        best_matched['role'] = ['target' if i == target_index else 'same configuration' for i in best_matched.index]
        # Target first, so that every other row can be checked against it
        best_matched = pd.concat([best_matched.loc[[target_index]],
                                  best_matched.drop(index=target_index).sort_values(select_by, ascending=False)])

        st.caption(f'Best `{target}` configuration by **{select_by}** (first row), and the setting each other '
                   'model is compared with:')
        table_columns = ['model_family', 'model_folder', 'role', *config_columns, select_by, *measure_columns,
                         *DETECTION_COLUMNS, 'nab_tp', 'nab_fp', 'nab_fn', 'file']
        st.dataframe(best_matched[list(dict.fromkeys(table_columns))], hide_index=True,
                     column_config={**{MEASURES[m][0]: st.column_config.NumberColumn(m, format=f'%{MEASURES[m][2]}')
                                       for m in measures},
                                    **DETECTION_COLUMN_CONFIG})
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
    st.download_button('Download as CSV', table.to_csv(index=False).encode(), file_name='grid_search_filtered.csv',
                       mime='text/csv')
