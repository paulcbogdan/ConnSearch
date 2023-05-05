import os
import pathlib

import numpy as np
from ncfs import NCFS
from numba import set_num_threads
from sklearn.svm import SVC

from ConnSearch.data_loading import load_data_2D_flat
from ConnSearch.utils import pickle_wrap, tril2flat_mappers
from comparison_methods.reporting.plotting_connectomewide import \
    connectomewide_plotting, count_edges_per_node


def get_NCFS_features(X, Y):
    X_means = np.mean(X, axis=0)
    X_stds = np.std(X, axis=0)
    X = (X - X_means) / X_stds
    ncfs = NCFS()
    ncfs.fit(X, Y)
    return ncfs, np.argsort(ncfs.coef_)[::-1]


def plot_NCFS(dataset_num=0, N=50, atlas='power', num_feats=398):
    '''
   Creates the NCFS plots shown in the manuscript. Note that NCFS is performed
       using the full dataset (e.g., all 50 participants), rather than just
       a training set. This ensures interpretation is as representative as
       possible.
   :param dataset_num: int, can be used to select among the five N=50 datasets
   :param N: int, dataset sample size
   :param atlas: str, atlas of dataset
   :param n_edges: int, cutoff for kept to edges (e.g., 398 means keep 398 best)
   '''
    # X imported here is a 2D array (examples, features). See load_data_2D_flat
    #   for more details.
    X_by_ex, Y_by_ex, groups, coords, tril_idxs = load_data_2D_flat(
        N=N, dataset_num=dataset_num, atlas=atlas)

    # NCFS is rather slow. To speed things up, we use a cache.
    # This is possible because get_NCFS_features(...) ranks every single feature
    #   from the connectome (i.e., many thousands), for a given dataset. This
    #   full ranking can be cached. This allows users to change the num_feats
    #   cutoff and quickly get new plots, without needing to redo the ranking.
    root_dir = pathlib.Path(__file__).parent.parent
    fp_pkl = os.path.join(root_dir, 'pickle_cache',
                          f'NCA_ranks_{atlas}_N{N}_d{dataset_num}.pkl')
    nca, coef_rankings = pickle_wrap(fp_pkl,
                                     lambda: get_NCFS_features(X_by_ex, Y_by_ex))

    # These dicts map an edge, (ROI0, ROI1), to a feature's index (edge_to_idx)
    #   and vice versa (idx_to_edge).
    edge_to_idx, idx_to_edge = tril2flat_mappers(tril_idxs)
    top_idxs = coef_rankings[:num_feats]
    top_edges = [tuple(idx_to_edge[idx]) for idx in top_idxs]
    clf = SVC(kernel='linear')  # SVM fit to get the mean SVM weights to plot
    clf.fit(X_by_ex[:, top_idxs], Y_by_ex)
    edge_weights = clf.coef_[0]

    pkg = {'edges': top_edges,
           'edge_weights': edge_weights,
           'coords_all': coords}

    # Score each ROI based on its number of linked top predictive edges
    #   node2num is a list. node2num[i] is the number of top edges with ROI_i
    #   node2num_std is a standardized variant, conformed to span 0. to 1.
    node2num_std, node2num = count_edges_per_node(top_edges, len(coords))

    dir_results = fr'{root_dir}/results/NCFS/NCFS_{atlas}_N{N}_d{dataset_num}'
    fn_out = f'NCFS_{atlas}_{num_feats}_count.png'
    title = f'Neighborhood Component Feature Selection, n[edges] = {num_feats}'
    connectomewide_plotting(dir_results, fn_out,
                            pkg, node2num, node2num_std,
                            atlas=atlas, title=title,
                            make_table=True)


if __name__ == '__main__':
    set_num_threads(1)
    plot_NCFS(N=50, atlas='power', num_feats=398)
