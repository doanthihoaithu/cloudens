"""
Evidence for the contributions of the paper, from a merged grid-search file (grid_results.py).

Every model is evaluated on the experiment cells (window, subset). In each cell, the post-processing
configuration (strategy, threshold, score normalization, fill_nan, ...) is the one with the best NAB score
on train_valid, and the reported score is the test NAB score of that configuration: the test split is never
used for the choice. The test-best ("oracle") score is reported next to it only to show the gap.

Outputs (CSV + LaTeX tables, figures, report.md) for each NAB profile:
    main            per model: test NAB (median [IQR], mean), oracle median, mean rank, detections, F1
    vs_proposed     the proposed model against every other one, paired on the cells (Wilcoxon, Holm)
    cd_diagram      Friedman test + Nemenyi critical-difference diagram of the mean ranks
    ablations       null-padding feature, post-processing strategy, score normalization (paired Wilcoxon)
    window          median test NAB per model and window
    boxplot         distribution of the test NAB over the cells, per model

Run from the project root (needs pandas, scipy, matplotlib):
    python src/contribution_analysis.py merged_results/hpc.csv.gz
    python src/contribution_analysis.py merged_results/hpc.csv.gz --proposed A3TGCN_null_padding_feature -o out/
"""
import argparse
import ast
import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / 'merged_results'
MERGED_SUFFIXES = ('.csv.gz', '.csv', '.parquet')

PROFILES = {'standard': 'Standard', 'reward_fn': 'Low FN'}
PROPOSED_DISPLAY_NAME = 'ClouDens'
CELL_KEYS = ['window', 'subset']
NULL_PADDING = '_null_padding_feature'
ALPHA = 0.05

COLUMNS = ['model_folder', 'window', 'subset', 'fill_nan', 'score_normalization', 'post_processing_strategy',
           'threshold_type', 'topk', 'anomaly_threshold', 'long_window', 'short_window',
           'standard_normalized', 'reward_fn_normalized', 'train_valid_standard_normalized',
           'train_valid_reward_fn_normalized', 'train_valid_f1', 'precision', 'recall', 'f1',
           'detection_counters', 'file']

# STGformer pooled over the whole window is saved as <model>_pool<window>: one model, <model>_poolwin
# (as in the results explorer)
POOL_SUFFIX = re.compile(r'^(?P<model>.+)_pool(?P<size>\d+)$')


def model_variant(model_folder, window):
    match = POOL_SUFFIX.match(str(model_folder))
    if match and int(match['size']) == int(window):
        return match['model'] + '_poolwin'
    return model_folder


# ── Loading and selection ─────────────────────────────────────────────────────

def latest_merged_file():
    files = [p for p in MERGED_DIR.glob('*') if p.is_file() and p.name.endswith(MERGED_SUFFIXES)]
    if not files:
        raise SystemExit(f'No merged file in {MERGED_DIR}: run grid_results.py first')
    return max(files, key=lambda p: p.stat().st_mtime)


def read_results(path):
    """Rows of a merged file (only the columns used here), one 'model' per model variant."""
    path = Path(path)
    if path.name.endswith('.parquet'):
        results = pd.read_parquet(path)
        results = results[[c for c in COLUMNS if c in results]]
    else:
        header = pd.read_csv(path, nrows=0).columns
        results = pd.read_csv(path, usecols=[c for c in COLUMNS if c in header])
    missing = [c for c in COLUMNS if c not in results]
    if missing:
        raise SystemExit(f'{path}: missing columns {", ".join(missing)} (not a merged grid-search file?)')
    results['model'] = [model_variant(m, w) for m, w in zip(results['model_folder'], results['window'])]
    return results


def select_configs(results, profile, by):
    """
    One row per group of `by`: the configuration with the best NAB score on train_valid (ties broken by
    the train_valid F1, then the train_valid NAB score of the other profile, then the file order).
    """
    other = next(p for p in PROFILES if p != profile)
    order = [f'train_valid_{profile}_normalized', 'train_valid_f1', f'train_valid_{other}_normalized']
    ranked = results.dropna(subset=[order[0]]).sort_values(order, ascending=False, na_position='last',
                                                            kind='stable')
    return ranked.drop_duplicates(by).reset_index(drop=True)


def add_detections(selected):
    """NAB TP / FP / FN of the selected rows (parsed only for them: the column is large)."""
    counters = selected['detection_counters'].map(_parse_counters)
    for key in ('tp', 'fp', 'fn'):
        selected[f'nab_{key}'] = counters.map(lambda c, k=key: c.get(k))
    return selected


def _parse_counters(counters):
    try:
        parsed = ast.literal_eval(counters)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def cell_table(selected, profile, column='model'):
    """Test NAB score: one row per cell, one column per value of `column`."""
    return selected.pivot_table(index=CELL_KEYS, columns=column, values=f'{profile}_normalized')


# ── Statistics ────────────────────────────────────────────────────────────────

def holm(p_values):
    """Holm-adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[idx]))
        adjusted[idx] = running
    return adjusted


def paired_test(a, b):
    """Wilcoxon signed-rank test of a vs b over the cells where both exist, with wins/ties/losses of a and
    the matched-pairs rank-biserial correlation (> 0: a better)."""
    pair = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
    d = pair['a'] - pair['b']
    row = {'n': len(pair), 'median_a': pair['a'].median(), 'median_b': pair['b'].median(),
           'median_diff': d.median(), 'wins': int((d > 0).sum()), 'ties': int((d == 0).sum()),
           'losses': int((d < 0).sum()), 'W': np.nan, 'p': np.nan, 'r_rb': np.nan}
    nonzero = d[d != 0]
    if len(nonzero) >= 2:
        ranks = stats.rankdata(nonzero.abs())
        row['r_rb'] = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
        try:
            with warnings.catch_warnings():   # ties / small samples: scipy switches method and says so
                warnings.simplefilter('ignore')
                test = stats.wilcoxon(pair['a'], pair['b'], zero_method='wilcox')
            row['W'], row['p'] = test.statistic, test.pvalue
        except ValueError:
            pass
    return row


def paired_tests(pairs):
    """pairs: [(labels dict, a, b)] -> table with Holm-adjusted p-values over all the pairs."""
    rows = [{**labels, **paired_test(a, b)} for labels, a, b in pairs]
    table = pd.DataFrame(rows)
    if not table.empty:
        table['p_holm'] = holm(table['p'])
        table['significant'] = table['p_holm'] < ALPHA
    return table


def friedman_nemenyi(table):
    """
    Friedman test and Nemenyi critical difference over the cells where every model of `table` has a score.
    Returns (mean ranks, best first; number of cells; Friedman statistic; p-value; critical difference).
    """
    complete = table.dropna()
    k, n = complete.shape[1], len(complete)
    ranks = complete.rank(axis=1, ascending=False).mean().sort_values()
    if k < 3 or n < 2:
        return ranks, n, np.nan, np.nan, np.nan
    statistic, p = stats.friedmanchisquare(*[complete[c] for c in complete])
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    return ranks, n, statistic, p, q * np.sqrt(k * (k + 1) / (6 * n))


# ── Figures ───────────────────────────────────────────────────────────────────

def cd_diagram(ranks, cd, title, path):
    """Critical-difference diagram (Demšar, 2006): models joined by a bar are not significantly different."""
    k = len(ranks)
    names, values = list(ranks.index), list(ranks.values)
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.28 * k))
    low, high = 1, max(k, int(np.ceil(max(values))))
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(0, 1.15)
    ax.axis('off')
    axis_y = 0.85
    ax.hlines(axis_y, low, high, color='black', linewidth=1)
    for r in range(low, high + 1):
        ax.vlines(r, axis_y, axis_y + 0.03, color='black', linewidth=1)
        ax.text(r, axis_y + 0.05, str(r), ha='center', va='bottom', fontsize=8)
    step = 0.7 / max(1, (k + 1) // 2)
    for i, (name, rank) in enumerate(zip(names, values)):
        left = i < (k + 1) // 2
        level = (i if left else k - 1 - i) + 1
        y = axis_y - 0.05 - level * step
        x_end = low - 0.4 if left else high + 0.4
        ax.plot([rank, rank, x_end], [axis_y, y, y], color='black', linewidth=0.8)
        ax.text(x_end, y, f'{name} ({rank:.2f})', ha='right' if left else 'left', va='center', fontsize=8)
    if not np.isnan(cd):
        ax.plot([low, low + cd], [axis_y + 0.12] * 2, color='black', linewidth=1.5)
        ax.text(low + cd / 2, axis_y + 0.14, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=8)
        # Maximal groups of models whose mean ranks differ by less than the critical difference
        cliques = []
        for i in range(k):
            j = max(j for j in range(i, k) if values[j] - values[i] < cd)
            if j > i and not any(a <= i and j <= b for a, b in cliques):
                cliques.append((i, j))
        for level, (i, j) in enumerate(cliques):
            y = axis_y - 0.04 - (level + 0.5) * 0.04
            ax.plot([values[i] - 0.03, values[j] + 0.03], [y, y], color='black', linewidth=3, solid_capstyle='round')
    ax.set_title(title, fontsize=9, pad=12)
    fig.tight_layout()
    _save_figure(fig, path)


def boxplot(table, title, path):
    order = table.median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.35 * len(order)))
    ax.boxplot([table[m].dropna() for m in order], vert=False, labels=list(order), showmeans=True)
    ax.invert_yaxis()
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Test NAB score (configuration chosen on train_valid)')
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save_figure(fig, path)


def _save_figure(fig, path):
    for suffix in ('.pdf', '.png'):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── Tables ────────────────────────────────────────────────────────────────────

def _format(value):
    if isinstance(value, (bool, np.bool_)):
        return 'yes' if value else 'no'
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return '–'
        return f'{value:.3g}' if abs(value) < 0.01 and value != 0 else f'{value:.2f}'
    return str(value)


def markdown_table(table):
    header = '| ' + ' | '.join(map(str, table.columns)) + ' |'
    rule = '|' + '|'.join('---' for _ in table.columns) + '|'
    rows = ['| ' + ' | '.join(_format(v) for v in row) + ' |' for row in table.itertuples(index=False)]
    return '\n'.join([header, rule, *rows])


def latex_table(table, caption, label):
    escape = lambda v: _format(v).replace('_', r'\_').replace('%', r'\%').replace('–', '--')
    lines = [r'\begin{table}[t]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\resizebox{\columnwidth}{!}{%', rf'\begin{{tabular}}{{{"l" * 1 + "r" * (len(table.columns) - 1)}}}',
             r'\toprule', ' & '.join(escape(c) for c in table.columns) + r' \\', r'\midrule']
    lines += [' & '.join(escape(v) for v in row) + r' \\' for row in table.itertuples(index=False)]
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(lines) + '\n'


class Report:
    """Writes every table as CSV + LaTeX and collects them in report.md."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sections = []

    def text(self, text):
        self.sections.append(text)

    def table(self, table, name, title, note=''):
        table.to_csv(self.directory / f'{name}.csv', index=False)
        (self.directory / f'{name}.tex').write_text(latex_table(table, title, f'tab:{name}'))
        self.sections.append(f'### {title}\n\n{note + chr(10) + chr(10) if note else ""}{markdown_table(table)}\n'
                             f'\n`{name}.csv` / `{name}.tex`')

    def write(self):
        path = self.directory / 'report.md'
        path.write_text('\n\n'.join(self.sections) + '\n')
        return path


# ── Analyses ──────────────────────────────────────────────────────────────────

def display(model, proposed):
    return f'{PROPOSED_DISPLAY_NAME} ({model})' if model == proposed else model


def main_comparison(results, profile, proposed, min_coverage, report):
    label = PROFILES[profile]
    score = f'{profile}_normalized'
    selected = add_detections(select_configs(results, profile, ['model', *CELL_KEYS]))
    selected.to_csv(report.directory / f'selected_configs_{profile}.csv', index=False)
    table = cell_table(selected, profile)
    oracle = results.groupby(['model', *CELL_KEYS])[score].max().groupby('model').median()

    coverage = table.notna().sum()
    ranked_models = coverage[coverage >= min_coverage * coverage.max()].index
    ranks, n_complete, statistic, p, cd = friedman_nemenyi(table[ranked_models])

    by_model = selected.groupby('model')
    main = pd.DataFrame({
        'cells': coverage,
        'median': table.median(),
        'q1': table.quantile(0.25),
        'q3': table.quantile(0.75),
        'mean': table.mean(),
        'oracle median': oracle,
        'mean rank': ranks,
        'detected (median)': by_model['nab_tp'].median(),
        'FP (median)': by_model['nab_fp'].median(),
        'test F1 (median)': by_model['f1'].median(),
    }).sort_values('median', ascending=False)
    main.insert(0, 'model', [display(m, proposed) for m in main.index])
    report.text(f'## NAB profile: {label}')
    report.table(main, f'main_{profile}', f'Test NAB ({label}), configuration chosen on train_valid',
                 f'Over the cells {" × ".join(CELL_KEYS)}; oracle = test-best configuration (optimistic, for '
                 f'reference only). Mean rank over the {n_complete} cells shared by the models covering at least '
                 f'{min_coverage:.0%} of the cells.')

    excluded = [m for m in table if m not in ranked_models]
    title = f'Mean rank, test NAB ({label}), {n_complete} cells'
    cd_diagram(ranks.rename(lambda m: display(m, proposed)), cd, title, report.directory / f'cd_diagram_{profile}')
    report.text(f'Friedman test over {len(ranked_models)} models and {n_complete} cells: '
                f'χ² = {_format(statistic)}, p = {_format(p)}; Nemenyi CD (α = {ALPHA}) = {_format(cd)}.'
                + (f' Not ranked (too few cells): {", ".join(excluded)}.' if excluded else '')
                + f'\n\n![CD diagram](cd_diagram_{profile}.png)')
    boxplot(table.rename(columns=lambda m: display(m, proposed)), f'Test NAB ({label}) over the cells',
            report.directory / f'boxplot_{profile}')
    report.text(f'![Box plot](boxplot_{profile}.png)')

    if proposed in table:
        tests = paired_tests([({'vs': other}, table[proposed], table[other])
                              for other in table if other != proposed])
        report.table(tests, f'vs_proposed_{profile}', f'{PROPOSED_DISPLAY_NAME} against each model, test NAB ({label})',
                     'Paired on the shared cells; wins/ties/losses of the proposed model; Wilcoxon signed-rank '
                     'test, Holm-adjusted over the comparisons; r_rb = matched-pairs rank-biserial correlation '
                     '(> 0: proposed better).')

    window = selected.pivot_table(index='model', columns='window', values=score, aggfunc='median')
    window.columns = [f'w={w}' for w in window.columns]
    window = window.loc[main.index].reset_index()
    window['model'] = [display(m, proposed) for m in window['model']]
    report.table(window, f'window_{profile}', f'Median test NAB ({label}) per sliding window')
    return table


def ablations(results, profile, table, proposed, report):
    label = PROFILES[profile]
    # Null-padding feature: each model against its variant with the feature, both chosen as in the main table
    pairs = [({'model': m.replace(NULL_PADDING, '')}, table[m], table[m.replace(NULL_PADDING, '')])
             for m in table if NULL_PADDING in m and m.replace(NULL_PADDING, '') in table]
    if pairs:
        report.table(paired_tests(pairs), f'ablation_null_padding_{profile}',
                     f'Ablation: null-padding feature, test NAB ({label})',
                     'a = with the null-padding feature, b = without; wins of a.')

    # Post-processing strategy and score normalization: the best configuration within each value
    for column, name in (('post_processing_strategy', 'strategy'), ('score_normalization', 'normalization')):
        selected = select_configs(results, profile, ['model', *CELL_KEYS, column])
        pairs = []
        for model, rows in selected.groupby('model'):
            per_value = cell_table(rows, profile, column)
            for a, b in combinations(per_value.columns, 2):
                pairs.append(({'model': display(model, proposed), 'a': a, 'b': b}, per_value[a], per_value[b]))
        if pairs:
            # The more configurations a value has, the more its choice on train_valid overfits: shown with it
            grid = (results.groupby(column).size() / results['file'].nunique()).round().astype(int)
            report.table(paired_tests(pairs), f'ablation_{name}_{profile}',
                         f'Ablation: {column.replace("_", " ")}, test NAB ({label})',
                         f'Within each model, the best configuration for each {column} (chosen on train_valid); '
                         f'wins of a. Configurations per grid-search file: '
                         f'{", ".join(f"{v} {n}" for v, n in grid.items())}.')


def main():
    parser = argparse.ArgumentParser(description='Contribution analysis of a merged grid-search file.')
    parser.add_argument('merged', nargs='?', help='Merged file (default: the most recent in merged_results/)')
    parser.add_argument('-o', '--output', help='Output folder (default: merged_results/contribution_analysis/<file>)')
    parser.add_argument('--proposed', default='A3TGCN_null_padding_feature',
                        help=f'Model shown as {PROPOSED_DISPLAY_NAME} (default: %(default)s)')
    parser.add_argument('--min-coverage', type=float, default=0.9,
                        help='Models in the Friedman test / CD diagram: those covering at least this fraction of '
                             'the cells of the best-covered model (default: %(default)s)')
    parser.add_argument('--exclude', nargs='*', default=[], help='Models (model folders or variants) to leave out')
    parser.add_argument('--threshold-type', choices=['percentile', 'absolute'],
                        help='Only the thresholds of this type: e.g. percentile, the grid every grid-search file has, '
                             'so that every model is chosen among the same configurations')
    args = parser.parse_args()

    path = Path(args.merged) if args.merged else latest_merged_file()
    stem = path.name.removesuffix(next(s for s in MERGED_SUFFIXES if path.name.endswith(s)))
    output = Path(args.output) if args.output else MERGED_DIR / 'contribution_analysis' / stem
    results = read_results(path)
    results = results[~results['model'].isin(args.exclude) & ~results['model_folder'].isin(args.exclude)]
    if args.threshold_type:
        results = results[results['threshold_type'] == args.threshold_type]
    if args.proposed not in set(results['model']):
        print(f'Warning: proposed model {args.proposed} not in {path.name}')

    report = Report(output)
    report.text(f'# Contribution analysis: `{path.name}`\n\n{results["file"].nunique()} grid-search files, '
                f'{len(results):,} rows, models: {", ".join(sorted(results["model"].unique()))}.'
                + (f' Thresholds: {args.threshold_type} only.' if args.threshold_type else '') + '\n\n'
                'Each model is scored on every cell (window, subset) with the configuration that has the best NAB '
                'score on train_valid; the score reported is the test NAB score of that configuration.')
    for profile in PROFILES:
        table = main_comparison(results, profile, args.proposed, args.min_coverage, report)
        report.text(f'## Ablations: {PROFILES[profile]}')
        ablations(results, profile, table, args.proposed, report)
    print(f'Report written to {report.write()}')


if __name__ == '__main__':
    main()
