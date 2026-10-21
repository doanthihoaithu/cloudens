"""
Time the construction of the operational-context graph on the node ids of every subset, e.g. no_group_4xx_count,
no_group_5xx_count and no_group_5xx_4xx_count, by two implementations giving the same edges:
    naive    utils.build_adjacency_matrix_no_group, as in preprocessing.prepare_5xx_4xx_features (the graphs of
             data/massaged/extracted/): every pair of nodes, through an N x N matrix
    grouped  build_adjacency_matrix_grouped: only the pairs of nodes sharing a component and an endpoint (the
             only ones with an edge), vectorized

The node ids are those of data/massaged/extracted/<subset>/meta_data.json. A single subset without its folder
(e.g. no_group_4xx_count) gets its node ids from a combined subset containing it (e.g. no_group_5xx_4xx_count):
its nodes of that http code, in the same order.

Writes to --output-dir only, never to data/massaged/extracted/ (its adjacency_matrix.npy files stay as they are):
    graph_construction_times.csv    one row per subset: nodes, edges, build times of both implementations
    <subset>_adjacency_matrix.npy   the graph built (as adjacency_matrix.npy: [source, target, weight] x E)

From the project root:
    python src/time_graph_construction.py
    python src/time_graph_construction.py --subsets no_group_4xx_count no_group_5xx_count --repeats 3
    python src/time_graph_construction.py --skip-naive   # the grouped implementation only (seconds)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import build_adjacency_matrix_no_group

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSETS_DIR = PROJECT_ROOT / 'data' / 'massaged' / 'extracted'
DEFAULT_SUBSETS = ['no_group_4xx_count', 'no_group_5xx_count', 'no_group_5xx_4xx_count']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'merged_results' / 'graph_construction'
# Tokens of a node id: datacenter1_CLIENT_component10_GET_400_endpoint865
COMMUNICATION_TYPE_TOKEN, COMPONENT_TOKEN, METHOD_TOKEN, HTTP_CODE_TOKEN, ENDPOINT_TOKEN = 1, 2, 3, 4, 5


def read_node_ids(subset):
    """Node ids of a subset and where they come from: its meta_data.json, or that of a combined subset holding
    it (no_group_4xx_count from no_group_5xx_4xx_count: the nodes of a 4xx code)."""
    meta_file = SUBSETS_DIR / subset / 'meta_data.json'
    if meta_file.is_file():
        return json.loads(meta_file.read_text())['node_ids'], str(meta_file.relative_to(PROJECT_ROOT))
    *codes, aggregation = subset.removeprefix('no_group_').split('_')
    if len(codes) != 1:
        raise FileNotFoundError(meta_file)
    code = codes[0]   # e.g. '4xx'
    for combined in sorted(SUBSETS_DIR.glob(f'no_group_*_{aggregation}')):
        combined_codes = combined.name.removeprefix('no_group_').split('_')[:-1]
        if code in combined_codes and len(combined_codes) > 1 and (combined / 'meta_data.json').is_file():
            node_ids = json.loads((combined / 'meta_data.json').read_text())['node_ids']
            node_ids = [n for n in node_ids if n.split('_')[HTTP_CODE_TOKEN].startswith(code[0])]
            return node_ids, f'{(combined / "meta_data.json").relative_to(PROJECT_ROOT)} ({code} nodes)'
    raise FileNotFoundError(f'{meta_file}, nor a combined subset with {code}')


def build_adjacency_matrix_grouped(node_ids):
    """
    Same edges, in the same order, as utils.build_adjacency_matrix_no_group(node_ids): an edge between two nodes
    of the same component and endpoint, weighted 0.8 with the same method, else 0.6 with the same communication
    type, else 0.2; both directions, [i, j, w] then [j, i, w] for every i < j in row-major order. Only the pairs
    inside each (component, endpoint) group are compared, without the N x N matrix.
    """
    tokens = [n.split('_') for n in node_ids]
    def codes(position):
        return pd.factorize(np.array([t[position] for t in tokens]))[0]
    communication_type, method = codes(COMMUNICATION_TYPE_TOKEN), codes(METHOD_TOKEN)
    group = pd.factorize(np.array([f'{t[COMPONENT_TOKEN]}_{t[ENDPOINT_TOKEN]}' for t in tokens]))[0]

    order = np.argsort(group, kind='stable')   # the nodes of each group, by increasing index
    starts = np.flatnonzero(np.r_[True, np.diff(group[order]) != 0])
    sources, targets = [], []
    for members in np.split(order, starts[1:]):
        if len(members) > 1:
            upper_i, upper_j = np.triu_indices(len(members), k=1)
            sources.append(members[upper_i])
            targets.append(members[upper_j])
    if not sources:
        return np.array([])   # as the naive implementation without edges
    i, j = np.concatenate(sources), np.concatenate(targets)
    pair_order = np.lexsort((j, i))   # row-major, as the loops of the naive implementation
    i, j = i[pair_order], j[pair_order]
    weight = np.where(method[i] == method[j], 0.8, np.where(communication_type[i] == communication_type[j], 0.6, 0.2))

    edges = np.empty((2 * len(i), 3))
    edges[0::2] = np.column_stack([i, j, weight])
    edges[1::2] = np.column_stack([j, i, weight])
    return edges


def timed_builds(build, node_ids, repeats, name, subset):
    """The graph built by build(node_ids).T (as prepare_5xx_4xx_features) and the time of each of its builds."""
    times = []
    for repeat in range(repeats):
        start = time.perf_counter()
        adjacency = build(node_ids).T
        times.append(time.perf_counter() - start)
        print(f'{subset}: {name} build {repeat + 1}/{repeats} in {times[-1]:.3f} s')
    return adjacency, times


def time_columns(times, name):
    if not times:
        return {f'build_time_{name}_mean_s': float('nan'), f'build_time_{name}_min_s': float('nan'),
                f'build_times_{name}_s': ''}
    return {f'build_time_{name}_mean_s': np.mean(times), f'build_time_{name}_min_s': np.min(times),
            f'build_times_{name}_s': ' '.join(f'{t:.4f}' for t in times)}


def same_as_existing(subset, adjacency):
    """True / False when the graph built equals the adjacency_matrix.npy of the subset, None without one."""
    existing = SUBSETS_DIR / subset / 'adjacency_matrix.npy'
    if not existing.is_file():
        return None
    existing = np.load(existing)
    return existing.shape == adjacency.shape and bool(np.allclose(existing, adjacency))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subsets', nargs='+', default=DEFAULT_SUBSETS)
    parser.add_argument('--repeats', type=int, default=1, help='Builds per subset (default: 1).')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--skip-naive', action='store_true',
                        help='Time the grouped implementation only (the naive one takes minutes on ~6500 nodes).')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subset in args.subsets:
        node_ids, source = read_node_ids(subset)
        print(f'{subset}: {len(node_ids)} nodes, from {source}')
        adjacency, grouped_times = timed_builds(build_adjacency_matrix_grouped, node_ids, args.repeats, 'grouped',
                                                subset)
        naive_times, same_as_naive = [], None
        if not args.skip_naive:
            naive, naive_times = timed_builds(build_adjacency_matrix_no_group, node_ids, args.repeats, 'naive',
                                              subset)
            same_as_naive = naive.shape == adjacency.shape and bool(np.array_equal(naive, adjacency))
        np.save(args.output_dir / f'{subset}_adjacency_matrix.npy', adjacency)
        rows.append({'subset': subset, 'node_ids_from': source, 'num_nodes': len(node_ids),
                     'num_edges': adjacency.shape[1] if adjacency.ndim == 2 else 0, 'repeats': args.repeats,
                     **time_columns(naive_times, 'naive'), **time_columns(grouped_times, 'grouped'),
                     'naive_dense_matrix_mb': len(node_ids) ** 2 * 8 / 2 ** 20,   # its N x N float64 matrix
                     'same_as_naive': same_as_naive, 'same_as_existing': same_as_existing(subset, adjacency)})

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / 'graph_construction_times.csv', index=False)
    print(table.drop(columns=['node_ids_from', 'build_times_naive_s', 'build_times_grouped_s']).to_string(index=False))
    print(f'Written to {args.output_dir}')


if __name__ == '__main__':
    main()
