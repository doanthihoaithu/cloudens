"""
Grid-search results of a results folder as one table: scanned from the folder, or read back from a merged
file (e.g. merged on the HPC server, then copied to a PC and opened in the explorer's 'Merged file' page).

Layout scanned:
    <root>/window_<w>/<subset>/fill_nan_with_<fill>/<model folder>/<model>_grid_search_<normalization>.csv
Every row is tagged with its experiment (TAG_COLUMNS, from the location of its file), so that a merged
file holds everything the explorer needs without the folder.

Merge a results folder (no Streamlit needed), from the project root:
    python streamlit_result_visualization/grid_results.py trained_models_log1p
    python streamlit_result_visualization/grid_results.py trained_models_log1p -o merged_results/hpc.csv.gz
"""
import argparse
import fnmatch
import gzip
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # streamlit_result_visualization/ is at the project root
GRID_FILE_PATTERN = re.compile(r'^(?P<model>.+)_grid_search_(?P<normalization>global|per_node)\.csv$')
# Experiment of every row, from the location of its grid-search file, and that file (relative to the
# project root it was merged on) with its modification time
TAG_COLUMNS = ['window', 'subset', 'fill_nan', 'model_folder', 'score_normalization', 'file', 'modified']
MERGED_DIR = PROJECT_ROOT / 'merged_results'
MERGED_SUFFIXES = ('.csv.gz', '.csv', '.parquet')
# Models left out of the merged files and of the explorer: whole model families (the folder name before its
# first '_', e.g. 'STGformer' for STGformer_null_padding_feature_pool6) and single model folders
IGNORED_MODEL_FAMILIES = {'MAGE', 'STGformer'}
IGNORED_MODEL_FOLDERS = {'AnomalyTransformer_association'}


def is_ignored_model(model_folder):
    """True for a model folder of IGNORED_MODEL_FOLDERS or of a family of IGNORED_MODEL_FAMILIES."""
    model_folder = str(model_folder)
    return model_folder in IGNORED_MODEL_FOLDERS or model_folder.split('_')[0] in IGNORED_MODEL_FAMILIES


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


def find_grid_files(root, include_ignored=False, subsets=None):
    """Experiment info of every grid-search CSV under root, but those of the ignored models (is_ignored_model)
    unless include_ignored; only the subsets matching one of the glob patterns of subsets when given (e.g.
    'no_group_*xx_*xx_*': the subsets combining two http codes)."""
    found = (parse_grid_path(p) for p in sorted(Path(root).glob('window_*/*/fill_nan_with_*/*/*_grid_search_*.csv')))
    return [info for info in found
            if info is not None and (include_ignored or not is_ignored_model(info['model_folder']))
            and (not subsets or any(fnmatch.fnmatch(info['subset'], pattern) for pattern in subsets))]


def grid_signature(files):
    """(path, modification time) of every grid-search file: changes when a file is added, removed or rewritten."""
    return tuple((f['path'], f['modified'].value) for f in files)


def read_grid_files(paths):
    """Rows of the grid-search CSVs at paths, tagged with TAG_COLUMNS: the table a merged file holds."""
    frames = []
    for path in paths:
        info = parse_grid_path(path)
        frame = pd.read_csv(path)
        for key in TAG_COLUMNS:
            if key != 'file':
                frame[key] = info[key]
        frame['file'] = os.path.relpath(path, PROJECT_ROOT)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _suffix(name):
    name = str(name).lower()
    return next((s for s in MERGED_SUFFIXES if name.endswith(s)), None)


def write_merged(results, path):
    """Write a merged table to path: .csv.gz (default, compressed), .csv or .parquet."""
    path = Path(path)
    suffix = _suffix(path.name)
    if suffix is None:
        raise ValueError(f'Merged file must end with one of {MERGED_SUFFIXES}: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.parquet':
        results.to_parquet(path, index=False)
    else:
        results.to_csv(path, index=False)   # gzip inferred from .gz
    return path


def merged_bytes(results):
    """A merged table as .csv.gz bytes, for a download."""
    return gzip.compress(results.to_csv(index=False).encode())


def read_merged(source, name=None):
    """
    Merged table from a path or a file-like object (an upload; name gives its format). Raises ValueError
    when it lacks TAG_COLUMNS, i.e. it is not a merged file (e.g. a single grid-search CSV).
    """
    suffix = _suffix(name if name is not None else source)
    if suffix is None:
        raise ValueError(f'Merged file must end with one of {MERGED_SUFFIXES}: {name or source}')
    if suffix == '.parquet':
        results = pd.read_parquet(source)
    else:
        results = pd.read_csv(source, compression='gzip' if suffix == '.csv.gz' else None)
    missing = [c for c in TAG_COLUMNS if c not in results]
    if missing:
        raise ValueError(f'Not a merged grid-search file, missing columns: {", ".join(missing)}')
    results['modified'] = pd.to_datetime(results['modified'])
    return results


def find_merged_files(directory=MERGED_DIR):
    """Merged files in directory, most recent first."""
    files = [p for p in Path(directory).glob('*') if p.is_file() and _suffix(p.name)]
    return sorted(files, key=os.path.getmtime, reverse=True)


def default_merged_path(root):
    """merged_results/<results folder>_<timestamp>.csv.gz"""
    return MERGED_DIR / f'{Path(root).resolve().name}_{datetime.now():%Y%m%d_%H%M%S}.csv.gz'


def write_merged_streaming(paths, path):
    """
    Merge the grid-search CSVs at paths into path (.csv.gz or .csv) one file at a time, so that the merged
    table never sits in memory (millions of rows). Written to a temporary file renamed at the end: a merge
    killed midway (e.g. out of memory) leaves no truncated file behind. Returns (rows, model folders).
    """
    path = Path(path)
    if _suffix(path.name) not in ('.csv.gz', '.csv'):
        raise ValueError(f'Streamed merge writes .csv.gz or .csv only: {path}')
    # Columns of every file (older grids lack some): one header for all, in order of appearance
    columns = []
    for p in paths:
        columns += [c for c in pd.read_csv(p, nrows=0).columns if c not in columns]
    columns += [c for c in TAG_COLUMNS if c not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    rows, model_folders = 0, set()
    try:
        opener = gzip.open if path.name.lower().endswith('.gz') else open
        with opener(tmp_path, 'wt', newline='') as out:
            for i, p in enumerate(paths):
                frame = read_grid_files([p]).reindex(columns=columns)
                frame.to_csv(out, index=False, header=i == 0)
                rows += len(frame)
                model_folders.add(frame['model_folder'].iat[0] if len(frame) else None)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return rows, len(model_folders - {None})


def main():
    parser = argparse.ArgumentParser(description='Merge the grid-search CSVs of a results folder into one file.')
    parser.add_argument('root', help='Results folder, e.g. trained_models_log1p')
    parser.add_argument('-o', '--output', help=f'Merged file ({", ".join(MERGED_SUFFIXES)}); '
                                               'default: merged_results/<folder>_<timestamp>.csv.gz')
    parser.add_argument('--subsets', nargs='+', metavar='PATTERN',
                        help="Only these subsets (glob patterns), e.g. 'no_group_*xx_*xx_*' for the subsets "
                             "combining two http codes (default: all)")
    parser.add_argument('--include-ignored', action='store_true',
                        help=f'Also merge the ignored models (families {sorted(IGNORED_MODEL_FAMILIES)}, '
                             f'folders {sorted(IGNORED_MODEL_FOLDERS)})')
    args = parser.parse_args()
    files = find_grid_files(args.root, include_ignored=args.include_ignored, subsets=args.subsets)
    if not files:
        raise SystemExit(f'No *_grid_search_*.csv found under {args.root}'
                         + (f' for the subsets {args.subsets}' if args.subsets else ''))
    path = Path(args.output or default_merged_path(args.root))
    paths = [f['path'] for f in files]
    if _suffix(path.name) == '.parquet':
        results = read_grid_files(paths)
        write_merged(results, path)
        rows, model_folders = len(results), results['model_folder'].nunique()
    else:
        rows, model_folders = write_merged_streaming(paths, path)
    print(f'Merged {len(files)} grid-search files ({rows:,} rows, {model_folders} model folders) into {path}' +
          ('' if args.include_ignored else f'; ignored: families {sorted(IGNORED_MODEL_FAMILIES)}, '
                                           f'folders {sorted(IGNORED_MODEL_FOLDERS)}'))


if __name__ == '__main__':
    main()
