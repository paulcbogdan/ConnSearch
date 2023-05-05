import pathlib

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from ConnSearch.data_loading import load_data_2D_flat
from ConnSearch.utils import tril2flat_mappers, pickle_wrap
from comparison_methods.reporting.plotting_connectomewide import connectomewide_plotting, count_edges_per_node


class RFE_SVM(BaseEstimator, ClassifierMixin):
    '''
    Training this classifier involves: (a) performing recursive feature
        elimination to identify the 1000 most predictive features, by
        repeatedly dropping the lowest weight SVM edges, (b) training an SVM
        using only the identified 1000 features.
    Note that to properly assess this classifier's accuracy, both steps
        must be performed for each cross-validation split, based on only
        training data. The selected features will vary slightly between splits.
    '''

    def __init__(self, num_feats, rfe_step=0.1):
        self.num_feats = num_feats
        self.key_feats = None
        self.rfe_step = rfe_step
        self.clf = SVC(C=1, kernel='linear', gamma='auto')
        self.coef_ranks = None
        self.coef_ranking = None

    def fit(self, X, y):
        _, coef_ranking = self.get_feature_rankings(X, y)
        # np.random.shuffle(coef_rankings)
        self.key_feats = coef_ranking[:self.num_feats]
        X_pruned = X[:, self.key_feats]
        self.clf.fit(X_pruned, y)
        print('done')

    def predict(self, X, y=None):
        X_pruned = X[:, self.key_feats]
        return self.clf.predict(X_pruned)

    def get_feature_rankings(self, X, y):
        if self.coef_ranks is not None:
            return self.coef_ranks, self.coef_ranking
        rfe = RFE(self.clf, n_features_to_select=self.num_feats,
                  step=self.rfe_step, verbose=1)
        selector = rfe.fit(X, y)
        coef_ranks = selector.ranking_
        coef_ranking = np.argsort(coef_ranks)
        self.coef_ranks = coef_ranks
        self.coef_ranking = coef_ranking
        return coef_ranks, coef_ranking


def get_RFE_features(n_edges, rfe_step, X, Y):
    clf = RFE_SVM(n_edges, rfe_step=rfe_step)
    coef_ranks, coef_rankings = clf.get_feature_rankings(X, Y)
    clf.fit(X, Y)
    edge_weights = clf.clf.coef_[0]
    return coef_ranks, coef_rankings, edge_weights


def plot_RFE(dataset_num=0, N=50, atlas='power', n_edges=1000, rfe_step=.01):
    '''
    Creates the RFE plots shown in the manuscript. Note that RFE is performed
        using the full dataset (e.g., all 50 participants), rather than just
        a training set. This ensures interpretation is as representative as
        possible of the full dataset. Using just training data would only be
        worthwhile for investigations of classification accuracy.
    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param N: int, dataset sample size
    :param atlas: str, atlas of dataset
    :param n_edges: int, number of edges to select via RFE
    :param rfe_step: float, proportion of edges eliminated each RFE step
    '''
    # X imported here is a 2D array (examples, features). See load_data_2D_flat
    #   for more details.
    X_by_ex, Y_by_ex, groups, coords, tril_idxs = load_data_2D_flat(
        atlas=atlas, N=N, dataset_num=dataset_num)

    root_dir = pathlib.Path(__file__).parent.parent
    fp_pkl = fr'{root_dir}/pickle_cache/' \
             f'RFE_feats_{atlas}_N{N}_d{dataset_num}_{rfe_step}.pkl'
    coef_ranks, coef_rankings, edge_weights = \
        pickle_wrap(fp_pkl, lambda: get_RFE_features(n_edges, rfe_step,
                                                     X_by_ex, Y_by_ex))
    top_idxs = coef_rankings[:n_edges]  # list of ints
    # convert list of features to a list of edges as tuples [(ROI0, ROI1), ...]
    edge_to_idx, idx_to_edge = tril2flat_mappers(tril_idxs)
    top_edges = [idx_to_edge[idx] for idx in top_idxs]  # list of tuples

    # Organize the variables necessary for plotting. Plotting uses the same
    #   functions as for ConnSearch, and the data are organized similarly to the
    #   ConnSearch result pkgs.
    pkg = {'edges': top_edges,  # list of 2-element tuples [(ROI0, ROI1), ...]
           'edge_weights': edge_weights,  # list of weight for each top edge
           'coords_all': coords}  # list of all coordinates in the connectome

    # Score each ROI based on its number of linked top predictive edges
    #   node2num is a list. node2num[i] is the number of top edges with ROI_i
    #   node2num_std is a standardized variant, conformed to span 0. to 1.
    node2num_std, node2num = count_edges_per_node(top_edges, len(coords))

    title = f'Recursive Feature Elimination, n[edges] = {len(top_edges)}'
    dir_out = fr'{root_dir}/results/RFE/' \
              f'RFE_step{rfe_step}_{atlas}_N{N}_d{dataset_num}'
    fn_out = f'RFE_{atlas}_{n_edges}.png'
    connectomewide_plotting(dir_out, fn_out,
                            pkg, node2num, node2num_std,
                            atlas=atlas, title=title,
                            make_table=True)


if __name__ == '__main__':
    # plot_RFE(N=250, atlas='schaefer1000', n_edges=10000, rfe_step=.01)
    plot_RFE(N=50, atlas='power', n_edges=1000, rfe_step=.001)
