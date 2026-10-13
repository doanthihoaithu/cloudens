"""
Convert the anomaly score CSVs of a results folder (<model>_anomaly_scores_<normalization>.csv, written by
earlier versions) to the parquet files written now: scores rounded to SCORE_DECIMALS, ~5-10x smaller. No model
is retrained: the CSV is read, rounded, written as parquet, read back and checked, and only then removed.

From the project root:
    python src/convert_anomaly_scores.py trained_models_log1p --dry-run   # what would be converted
    python src/convert_anomaly_scores.py trained_models_log1p             # convert, remove the CSVs
    python src/convert_anomaly_scores.py trained_models_log1p --keep-csv  # convert, keep the CSVs
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from model_outputs import read_anomaly_scores_file, round_scores, write_anomaly_scores_file

SCORE_CSV_PATTERN = re.compile(r'^.+_anomaly_scores_(global|per_node)\.csv$')


def find_score_csvs(root):
    return sorted(p for p in Path(root).rglob('*_anomaly_scores_*.csv') if SCORE_CSV_PATTERN.match(p.name))


def same_scores(expected, written):
    """The parquet file holds the rounded CSV: same columns and rows, equal values (NaN where NaN)."""
    if list(expected.columns) != list(written.columns) or len(expected) != len(written):
        return False
    for column in expected.columns:
        a, b = expected[column], written[column]
        if pd.api.types.is_float_dtype(a):
            if not np.array_equal(a.to_numpy(), b.to_numpy(dtype=float), equal_nan=True):
                return False
        elif column == 'timestamp':
            if not pd.to_datetime(a).equals(pd.to_datetime(b)):
                return False
        elif not a.equals(b):
            return False
    return True


def convert(csv_path, keep_csv=False):
    """Returns (status, bytes before, bytes after)."""
    parquet_path = csv_path.with_suffix('.parquet')
    before = csv_path.stat().st_size
    if before == 0:
        return 'empty CSV (left as is)', before, 0
    expected = round_scores(read_anomaly_scores_file(str(csv_path)))
    tmp_path = parquet_path.with_suffix('.parquet.tmp')
    try:
        write_anomaly_scores_file(expected, str(tmp_path))
        if not same_scores(expected, pd.read_parquet(tmp_path)):
            tmp_path.unlink()
            return 'MISMATCH (CSV kept, no parquet)', before, 0
        os.replace(tmp_path, parquet_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)   # e.g. disk quota exceeded: no partial parquet left behind
        raise
    after = parquet_path.stat().st_size
    if not keep_csv:
        csv_path.unlink()
    return 'converted', before, after


def main():
    parser = argparse.ArgumentParser(description='Convert the anomaly score CSVs of a results folder to parquet.')
    parser.add_argument('root', help='Results folder, e.g. trained_models_log1p')
    parser.add_argument('--dry-run', action='store_true', help='Only list the CSVs and their total size')
    parser.add_argument('--keep-csv', action='store_true', help='Keep the CSVs after a successful conversion')
    args = parser.parse_args()

    csvs = find_score_csvs(args.root)
    total = sum(p.stat().st_size for p in csvs)
    print(f'{len(csvs)} anomaly score CSVs under {args.root}, {total / 1e9:.1f} GB')
    if args.dry_run:
        return

    before_total = after_total = 0
    failed = []
    for i, csv_path in enumerate(csvs, 1):
        try:
            status, before, after = convert(csv_path, args.keep_csv)
        except Exception as e:   # e.g. a truncated CSV written when the disk quota was exceeded
            status, before, after = f'FAILED: {type(e).__name__}: {e}', 0, 0
        if status != 'converted':
            failed.append((csv_path, status))
        else:
            before_total += before
            after_total += after
        print(f'[{i}/{len(csvs)}] {status}: {csv_path} ({before / 1e6:.0f} MB -> {after / 1e6:.0f} MB)', flush=True)

    print(f'Converted {len(csvs) - len(failed)} / {len(csvs)}: {before_total / 1e9:.1f} GB -> {after_total / 1e9:.1f} GB')
    for path, status in failed:
        print(f'Not converted: {path}: {status}')


if __name__ == '__main__':
    main()
