import pathlib

import numpy as np
import scipy.stats as stats

from ConnSearch.data_loading import load_data_5D
from ConnSearch.utils import get_t_graph
from comparison_methods.reporting.plotting_connectomewide import connectomewide_plotting, count_edges_per_node


def plot_CPM(dataset_num=0, N=50, atlas='power', alpha_thresh=0.05):
    '''
    Creates the CPM plots shown in the manuscript. Amounts to performing t-tests
        filtering to just edges showing significant (p < .05) 2-back vs. 0-back
        differences, then plotting them.
    Interpretation uses the full dataset for the t-tests (no training set).
        This ensures interpretation is as representative as possible of the
        neural data. Although not reported in the present manuscript, we provide
        code elsewhere investigating accuracy, which does CPM filtering using
        just training set data.

    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param N: int, dataset sample size
    :param atlas: str, atlas of dataset
    :param alpha_thresh: float, p-value threshold for edge selection
    '''

    X, Y, coords, _ = load_data_5D(N=N, dataset_num=dataset_num, atlas=atlas)
    t_graph = get_t_graph(X, Y)
    t_cutoff = stats.t.ppf(1 - alpha_thresh / 2, N - 1)  # two-sided
    top_edges = np.argwhere(abs(t_graph) > t_cutoff)  # top_edges = significant

    # omit symmetric duplicates - e.g., keep only (0, 1) and not (1, 0))
    top_edges = np.array([edge for edge in top_edges if edge[0] < edge[1]])
    edge_weights = np.array([t_graph[i, j] for (i, j) in top_edges])

    # Organize the variables necessary for plotting. Plotting uses the same
    #   functions as for ConnSearch, and the data are organized similarly to the
    #   ConnSearch result pkgs.
    pkg = {'edges': top_edges,
           'edge_weights': edge_weights,
           'coords_all': coords}
    root_dir = pathlib.Path(__file__).parent.parent
    dir_out = fr'{root_dir}/results/CPM/' \
              f'CPM_{atlas}_a{alpha_thresh}_N{N}_d{dataset_num}'
    fn_out = f'CPM_a{alpha_thresh}_edges{len(top_edges)}.png'

    # The glass brain visuals contain info on the number of predictive edges
    #   (top_edges) associated with each ROI.
    node2num_d, node2num_ustd_d = count_edges_per_node(top_edges, len(coords))

    # Generates multiple plots
    title = f'Connectome Predictive Modeling, n[edges] = {len(top_edges)}'
    connectomewide_plotting(dir_out, fn_out,
                            pkg, node2num_ustd_d, node2num_d,
                            atlas=atlas, title=title,
                            make_table=True)


if __name__ == '__main__':
    plot_CPM(atlas='power', N=50, alpha_thresh=.05)
