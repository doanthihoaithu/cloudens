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


def main():
    parser = argparse.ArgumentParser(description='Merge the grid-search CSVs of a results folder into one file.')
    parser.add_argument('root', help='Results folder, e.g. trained_models_log1p')
    parser.add_argument('-o', '--output', help=f'Merged file ({", ".join(MERGED_SUFFIXES)}); '
                                               'default: merged_results/<folder>_<timestamp>.csv.gz')
    args = parser.parse_args()
    files = find_grid_files(args.root)
    if not files:
        raise SystemExit(f'No *_grid_search_*.csv found under {args.root}')
    results = read_grid_files([f['path'] for f in files])
    path = write_merged(results, args.output or default_merged_path(args.root))
    print(f'Merged {len(files)} grid-search files ({len(results):,} rows, '
          f'{results["model_folder"].nunique()} model folders) into {path}')


if __name__ == '__main__':
    main()
