import pathlib

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from ConnSearch.data_loading import load_data_2D_flat
from comparison_methods.reporting.plotting_connectomewide import \
    connectomewide_plotting, sum_edgeweight_mag_per_node


def get_haufe_transformed_weights(X, Y):
    '''
    Follows the procedures by Chen et al. (2022) for implementing the Haufe et
        al. (2014) transformation. A kernel ridge regression is fit, predicting
        Y from X. Then, for each edge, the between-example correlation is
        measured between the edge's connectivity and the ridge regressions
        predicted Y for that example.
    :param X: 2D array, where .shape=(examples, features)
    :param Y: list of binary labels (e.g., 1 = 2-back, 0 = 0-back)
    :return:
    '''
    clf = KernelRidge()
    clf.fit(X, Y)
    pred_y = clf.predict(X)
    edge_weights = []
    for i in range(X.shape[1]):
        edge_weights.append(np.corrcoef(X[:, i], pred_y)[0, 1])
    # Note that edge_weights is not the regression weights. Rather, it is the
    #   Haufe-transformed regression weights.
    return np.array(edge_weights)


def plot_Haufe(dataset_num=0, N=50, atlas='power'):
    '''
    Creates the Kernel Ridge Regression + Haufe transformation plots shown in
        the manuscript. Involves fitting a kernel ridge regression then taking,
        for each edge, the correlation between connectivity and the kernel
        ridge regression prediction. This implementation follows the strategy
        by Chen et al. (2022). For more details on this transformation, see
        Haufe et al. (2014).
    This plotting procedure deviates from RFE, CPM, andf NCFS. Haufe
        transformation yields a continuous predictiveness weight measure for
        every edge, rather than identifying a subset as highly predictive.
        Hence, plotting does not use a counts, but rather the sum of the
        absolute values of weights (between a pair of networks or with an ROI).
    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param N: int, dataset sample size
    :param atlas: str, atlas of dataset
    '''
    X_by_ex, Y_by_ex, groups, coords, tril_idxs = load_data_2D_flat(
        atlas=atlas, N=N, dataset_num=dataset_num)

    edge_weights = get_haufe_transformed_weights(X_by_ex, Y_by_ex)
    edges = list(zip(*tril_idxs))

    # Organize the variables necessary for plotting. Plotting uses the same
    #   functions as for ConnSearch, and the data are organized similarly to the
    #   ConnSearch result pkgs.
    pkg = {'edges': edges,
           'edge_weights': edge_weights,
           'coords_all': coords}

    # Score each ROI based on the absolute sum of its linked edges' weights
    #   node2num is a list. node2num[i] depend on the edges connected to ROI_i
    #   node2num_std is a standardized variant, conformed to span 0. to 1.
    node2num_std, node2num = \
        sum_edgeweight_mag_per_node(edges, edge_weights, len(coords))

    root_dir = pathlib.Path(__file__).parent.parent
    dir_out = fr'{root_dir}/results/Haufe/haufe_{atlas}_N{N}_d{dataset_num}'
    title = f'Ridge regression & Haufe transformation'
    fn_out = f'haufe_{atlas}_sum.png'
    connectomewide_plotting(dir_out, fn_out,
                            pkg, node2num, node2num_std,
                            atlas=atlas, title=title,
                            make_table=False)


if __name__ == '__main__':
    plot_Haufe(atlas='power', N=50)
