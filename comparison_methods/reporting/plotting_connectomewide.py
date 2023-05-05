import os

from ConnSearch.reporting.plotting_general import plot_pkg, plot_ROI_scores
from ConnSearch.utils import clear_make_dir
from comparison_methods.reporting.connectomewide_tables import make_network_pair_table


def connectomewide_plotting(dir_out, fn_out,
                            pkg, node2num_ustd_d, node2num_std,
                            atlas='power', title='',
                            make_table=False,
                            ):
    '''
    Prepares the plots for the connectome-wide interpretation methods. Makes
        multiple plots.
    This uses plotting functions from ConnSearch.reporting.plotting_general.

    :param dir_out: str, directory where the plots will be saved
    :param fn_out: str, filename for the plot. This str will be modified
        depending on the specific plot.
    :param pkg: dict, contains the data necessary for plotting. Similar to
        the ConnSearch result pkgs.
    :param node2num_ustd_d: dict, maps ROI index to the ROI's unstandardized
        weight (e.g., number of predictive edges connected to a given ROI).
        Used for plotting ROI sizes.
    :param node2num_std: dict, maps ROI to the standardized weight
    :param atlas: str, name of the atlas used
    :param title: str, title for the glass brain + chord diagram plot
    :param make_table: bool, whether to make a table of the top edges.
        Only reported for recursive feature elimination in the manuscript.
    :param only_count: bool, whether
    :return:
    '''
    coords = pkg['coords_all']
    top_edges = pkg['edges']
    multiplier = 2.7
    total_size = sum(x * multiplier for x in node2num_std)
    size_mod = 9000 / total_size
    node_sizes = []
    for i in range(len(coords)):  # Size is based on area so this is squared
        node_sizes.append(size_mod * ((node2num_std[i]) ** (multiplier)))

    node2num_ustd_l = [node2num_ustd_d[i] for i in range(len(coords))]

    glass_params = {'alphas': 0.25,
                    'node_size': node_sizes,
                    'linewidths': 0}

    node2num_ustd_l_sorted = sorted(node2num_ustd_l)
    vmax = node2num_ustd_l_sorted[-30]
    vmin = node2num_ustd_l_sorted[0]

    clear_make_dir(dir_out, clear_dir=False)
    fp_ROIs = os.path.join(dir_out, fn_out.replace('.png', '_ROI_scores.png'))
    plot_ROI_scores(node2num_ustd_l, coords, fp_out=fp_ROIs,
                    show=False, vmin=vmin, vmax=vmax)

    clear_make_dir(os.path.join(dir_out, 'glass'), clear_dir=False)
    clear_make_dir(os.path.join(dir_out, 'chord'), clear_dir=False)

    fp_table = os.path.join(dir_out, fn_out.replace('.png', '.csv'))
    if make_table:
        print(f'Table: {fp_table=}')
        make_network_pair_table(top_edges, fp_table, coords,
                                atlas=atlas,
                                do_open=False)

    chord_params = {'alphas': 0.9,
                    'plot_count': True,
                    'norm_thickness': True,
                    'plot_abs_sum': True,
                    'magnify_thickness_difs': 2.0}
    plot_pkg(pkg, dir_out, fn_out.replace('.png', '_abssum_norm.png'),
             glass_params=glass_params,
             chord_params=chord_params, title=title)


def count_edges_per_node(top_edges, n_nodes):
    '''
    Score each ROI based on its number of linked top predictive edges

    :param top_edges: list of top (most predictive) edges
    :param n_nodes: int, number of ROIs in connectome
    :return:
      node2num_std, list, which is the standardized version of node2num (below)
      node2num, list, where node2num[i] is the number of top edges with ROI_i
    '''
    node2num = [0] * n_nodes
    for edge in top_edges:
        node2num[edge[0]] += 1
        node2num[edge[1]] += 1
    min_sum = min(node2num)
    max_sum = max(node2num)
    node2num_std = [(num - min_sum) / (max_sum - min_sum) for num in node2num]
    return node2num_std, node2num


def sum_edgeweight_mag_per_node(edges, edge_weights, n_nodes):
    '''
    Similar to count_edges_per_node, but instead of counting edges, it sums
        the absolute value of the edge weights. This is used for the Haufe
        approach visualizations.

    :param edges: list tuples, [(ROI0, ROI1), ...]
    :param edge_weights: list of floats, [weight0, weight1, ...], corresponding
                         to the edges
    :return:
      node2num_std, list, which is the standardized version of node2num (below)
      node2num, list, where node2num[i] is the absolute sum of the weights for
        edges connected to ROI_i
    '''
    node2num = [0] * n_nodes
    for edge, weight in zip(edges, edge_weights):
        node2num[edge[0]] += abs(weight)
        node2num[edge[1]] += abs(weight)
    min_sum = min(node2num)
    max_sum = max(node2num)
    node2num_std = [(num - min_sum) / (max_sum - min_sum) for num in node2num]
    return node2num_std, node2num
