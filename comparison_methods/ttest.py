import os
import pathlib

import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from ConnSearch.data_loading import load_data_5D
from ConnSearch.reporting.plotting_general import plot_pkg
from ConnSearch.utils import get_t_graph, clear_make_dir


def plot_t_test(dataset_num=0, N=50, atlas='power', corrected_alpha_thresh=.05,
                correction_method='fdr_bh'):  # 'holm-sidak'
    X, Y, coords, _ = load_data_5D(N=N, dataset_num=dataset_num, atlas=atlas)
    t_graph = get_t_graph(X, Y)
    tril_idxs = np.tril_indices(len(t_graph), -1)
    t_flat = t_graph[tril_idxs]
    p_flat = (1 - stats.t.cdf(abs(t_flat), df=N - 1)) * 2
    signif_flat, _, _, _ = multipletests(p_flat, alpha=corrected_alpha_thresh,
                                         method=correction_method)

    # 2D array shaped (#signif, 2) where each row is a pair of indices
    #   representing a significant edge
    signif_edges = np.array(tril_idxs)[:, signif_flat].T
    edge_weights = t_graph[signif_edges[:, 0], signif_edges[:, 1]]

    pkg = {'edges': signif_edges,
           'edge_weights': edge_weights,
           'coords_all': coords}
    root_dir = pathlib.Path(__file__).parent.parent
    dir_out = fr'{root_dir}/results/t-test/t-test_{atlas}_N{N}_d{dataset_num}'
    clear_make_dir(dir_out, clear_dir=False)
    clear_make_dir(os.path.join(dir_out, 'glass'), clear_dir=False)
    clear_make_dir(os.path.join(dir_out, 'chord'), clear_dir=False)

    method2title = {'fdr_bh': 'Benjamini-Hochberg',
                    'holm-sidak': 'Holm-Sidak',
                    'bonferroni': 'Bonferroni'}
    method_title = method2title[correction_method]
    fn_out = f't_test_p{corrected_alpha_thresh}_{correction_method}_' \
             f'edges{len(signif_edges)}.png'
    title = f't-tests ({method_title} corrected, p < {corrected_alpha_thresh}' \
            f'), n[edges] = {len(signif_edges)}'
    plot_pkg(pkg, dir_out, fn_out, title=title,
             chord_params={'black_BG': False})


if __name__ == '__main__':
    plot_t_test()
