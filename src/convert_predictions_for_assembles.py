"""
Convert the grid-search test predictions written as CSV by earlier versions
(<model>_predictions_for_assembles_<normalization>.csv, 0/1 text) to the boolean parquet files
the training now writes (about 20-50x smaller). Each parquet file is read back and compared with
the CSV before the CSV is removed.

From the project root:
    python src/convert_predictions_for_assembles.py trained_models_log1p --dry-run   # list the files only
    python src/convert_predictions_for_assembles.py trained_models_log1p
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_outputs import read_predictions_for_assembles, write_predictions_for_assembles

SUFFIX = '_predictions_for_assembles_'


def convert(csv_path):
    """Write the parquet file of a CSV and remove the CSV; returns the (CSV, parquet) sizes in bytes."""
    model_dir, name = os.path.split(csv_path)
    model, normalization = name.removesuffix('.csv').split(SUFFIX)
    csv_size = os.path.getsize(csv_path)
    predictions = pd.read_csv(csv_path)
    if not predictions.isin([0, 1]).all().all():
        raise ValueError(f'{csv_path} has values other than 0 / 1')
    # Written next to the CSV under a temporary name, so a failed run never leaves a partial parquet file
    tmp_dir = os.path.join(model_dir, '.converting')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = write_predictions_for_assembles(predictions, tmp_dir, model, normalization)
    if not read_predictions_for_assembles(tmp_dir, model, normalization).equals(predictions.astype('int64')):
        raise ValueError(f'The parquet file of {csv_path} differs from it')
    parquet_path = os.path.join(model_dir, os.path.basename(tmp_path))
    os.replace(tmp_path, parquet_path)
    os.rmdir(tmp_dir)
    os.remove(csv_path)
    return csv_size, os.path.getsize(parquet_path)


def main():
    parser = argparse.ArgumentParser(description='Convert the predictions_for_assembles CSVs to boolean parquet.')
    parser.add_argument('root', help='Results folder, e.g. trained_models_log1p')
    parser.add_argument('--dry-run', action='store_true', help='Only list the files')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, '**', f'*{SUFFIX}*.csv'), recursive=True))
    total_size = sum(os.path.getsize(f) for f in files)
    print(f'{len(files)} CSV files, {total_size / 1e9:.2f} GB')
    if args.dry_run:
        print('\n'.join(files))
        return
    saved = 0
    for i, csv_path in enumerate(files, 1):
        try:
            csv_size, parquet_size = convert(csv_path)
        except Exception as error:
            print(f'[{i}/{len(files)}] FAILED {csv_path}: {error}', flush=True)
            continue
        saved += csv_size - parquet_size
        print(f'[{i}/{len(files)}] {csv_path}: {csv_size / 1e6:.1f} MB -> {parquet_size / 1e6:.2f} MB', flush=True)
    print(f'Freed {saved / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
