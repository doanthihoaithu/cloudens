"""
Streamlit explorer of the grid-search results under a results folder (default: trained_models_log1p).

Layout scanned:
    <root>/window_<w>/<subset>/fill_nan_with_<fill>/<model folder>/<model>_grid_search_<normalization>.csv

Run from the project root:
    streamlit run streamlit_result_visualization/results_explorer_app.py
"""
import ast
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # streamlit_result_visualization/ is at the project root
GRID_FILE_PATTERN = re.compile(r'^(?P<model>.+)_grid_search_(?P<normalization>global|per_node)\.csv$')

NAB_COLUMNS = {'reward_fn': 'reward_fn_normalized', 'standard': 'standard_normalized'}
# Metrics a setting can be selected by: NAB on test, or metrics on train / valid (no test labels)
SELECTION_METRICS = [
    'reward_fn_normalized', 'standard_normalized', 'f1',
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
PARAMETER_COLUMNS = ['post_processing_strategy', 'threshold_type', 'topk', 'long_window', 'short_window', 'anomaly_threshold']


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


def _detection_count(counters, key):
    try:
        return ast.literal_eval(counters).get(key)
    except (ValueError, SyntaxError, AttributeError):
        return None


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
    for key in ('tp', 'fp', 'fn'):
        results[f'nab_{key}'] = results['detection_counters'].map(lambda c, k=key: _detection_count(c, k))
    results['topk'] = [_topk_label(k, s) for k, s in zip(results['topk'], results['post_processing_strategy'])]
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

grid_files = find_grid_files(PROJECT_ROOT / root_name)
if not grid_files:
    st.warning(f'No `*_grid_search_*.csv` found under `{root_name}`.')
    st.stop()
results = load_results(tuple((f['path'], f['modified'].value) for f in grid_files))

# ── Filters ──
with st.sidebar:
    st.header('Filters')
    selected = {}
    for column, label in DIMENSIONS.items():
        options = sorted(results[column].dropna().unique().tolist(), key=str)
        selected[column] = st.multiselect(label, options, default=options)

    st.header('Comparison')
    nab_profile = st.segmented_control('NAB profile', list(NAB_COLUMNS), default='reward_fn', required=True)
    available_metrics = [m for m in SELECTION_METRICS if m in results.columns]
    select_by = st.selectbox(
        'Pick the best setting of each group by', available_metrics,
        index=available_metrics.index(NAB_COLUMNS[nab_profile]),
        help='NAB / F1 select on the test split itself; train_valid_* / valid_* select without the test labels.')
    group_by = st.multiselect('Compare groups of', list(DIMENSIONS), default=['model_folder', 'post_processing_strategy'],
                              format_func=DIMENSIONS.get)

mask = pd.Series(True, index=results.index)
for column, values in selected.items():
    mask &= results[column].isin(values)
filtered = results[mask]
nab_column = NAB_COLUMNS[nab_profile]

if filtered.empty:
    st.info('No grid-search row matches the filters.')
    st.stop()

tab_overview, tab_compare, tab_rows = st.tabs(['Overview', 'Compare NAB', 'All settings'])

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
        best_nab=(nab_column, 'max'))
    inventory['file'] = inventory['path'].map(lambda p: os.path.relpath(p, PROJECT_ROOT))
    inventory = inventory.merge(rows, left_on='file', right_index=True)
    inventory['has_split_metrics'] = inventory['file'].map(
        lambda f: bool(results.loc[results['file'] == f].filter(like='train_valid_').notna().any().any()))
    st.dataframe(
        inventory[['window', 'subset', 'fill_nan', 'model_folder', 'score_normalization', 'settings', 'strategies',
                   'best_nab', 'has_split_metrics', 'modified', 'file']],
        hide_index=True,
        column_config={
            'best_nab': st.column_config.NumberColumn(f'Best NAB ({nab_profile})', format='%.2f'),
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
            best = best.sort_values(nab_column, ascending=False)

            st.caption(f'Best setting of each group by **{select_by}**; bars show its test NAB ({nab_profile}).')
            color_dim = 'post_processing_strategy' if 'post_processing_strategy' in group_by else group_by[-1]
            chart = alt.Chart(best).mark_bar().encode(
                x=alt.X(f'{nab_column}:Q', title=f'NAB {nab_profile} (normalized)'),
                y=alt.Y('group:N', sort='-x', title=None),
                color=alt.Color(f'{color_dim}:N', title=DIMENSIONS[color_dim]),
                tooltip=['group', alt.Tooltip(f'{nab_column}:Q', format='.2f'), alt.Tooltip(f'{select_by}:Q', format='.4f'),
                         *PARAMETER_COLUMNS, 'nab_tp', 'nab_fp', 'nab_fn'],
            ).properties(height=max(200, 28 * len(best)))
            st.altair_chart(chart)

            shown = group_by + [c for c in PARAMETER_COLUMNS if c not in group_by] + \
                [select_by] * (select_by not in NAB_COLUMNS.values()) + \
                ['reward_fn_normalized', 'standard_normalized', 'nab_tp', 'nab_fp', 'nab_fn', 'f1', 'test_auc_pr', 'test_vus_pr']
            shown = [c for c in dict.fromkeys(shown) if c in best.columns]
            st.dataframe(best[shown], hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format='%.4f')
                                        for c in shown if c.endswith(('_pr', 'f1'))})

        st.subheader('Spread of NAB over all settings')
        spread = filtered.copy()
        spread['group'] = spread[group_by].astype(str).agg(' · '.join, axis=1)
        box = alt.Chart(spread).mark_boxplot(extent='min-max').encode(
            x=alt.X(f'{nab_column}:Q', title=f'NAB {nab_profile} (normalized)'),
            y=alt.Y('group:N', title=None),
            color=alt.Color('group:N', legend=None),
        ).properties(height=max(200, 28 * spread['group'].nunique()))
        st.altair_chart(box)

# ── All settings ──
with tab_rows:
    st.caption(f'{len(filtered):,} settings match the filters (sorted by NAB {nab_profile}).')
    leading = list(DIMENSIONS) + ['topk', 'long_window', 'short_window', 'anomaly_threshold',
                                  'reward_fn_normalized', 'standard_normalized', 'nab_tp', 'nab_fp', 'nab_fn',
                                  'precision', 'recall', 'f1']
    leading = [c for c in leading if c in filtered.columns]
    others = [c for c in filtered.columns if c not in leading and c not in ('detection_counters',)]
    table = filtered[leading + others].sort_values(nab_column, ascending=False)
    st.dataframe(table, hide_index=True, height=600)
    st.download_button('Download as CSV', table.to_csv(index=False).encode(), file_name='grid_search_filtered.csv',
                       mime='text/csv')
