import os
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedStratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm

from ConnSearch.data_loading import load_data_2D_flat
from ConnSearch.utils import pickle_wrap, clear_make_dir
from comparison_methods.NCFS import get_NCFS_features


def test_NCFS_acc_contaminated(X, Y, groups, cache_str, n_splits=2, n_repeats=10,
                               num_feats=1000):
    '''
    See plot_erylimaz_accs(...). The present function tests the accuracy for a
        single feature cutoff. num_feats is the variable representing "Z"
        in the plot_erylimaz_accs(...) explanation.
    :param X:

    '''
    # This function will be run for many different values of num_feats. To make
    #   this much faster, the NCA ranks are cached. Note get_NCFS_features(...)
    #   ranks every single feature from the connectome and changing num_feats
    #   just shifts the cutoff for how many features to use.
    root_dir = pathlib.Path(__file__).parent.parent
    fp_pkl = os.path.join(root_dir, 'pickle_cache',
                          f'NCA_feature_rankings_{cache_str}.pkl')
    # see pickle_wrap(...) for details on how the caching works
    nca, coef_rankings = pickle_wrap(fp_pkl, get_NCFS_features, args=[X, Y],
                                     verbose=False)
    X = X[:, coef_rankings[:num_feats]]
    clf = SVC(kernel='linear')
    # same cross-validation as for ConnSearch
    cv = RepeatedStratifiedGroupKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=0)
    acc = cross_val_score(clf, X, Y, cv=cv, groups=groups, scoring='accuracy')
    return np.mean(acc)


def plot_NCFS_accs(dataset_num=0, N=50, atlas='power'):
    '''
    This function is used to identify the threshold of predictive edges for the
      NCFS interpretation (398 edges).
    To our knowledge, this procedure follows the original NCFS approach
      to interpetation by Eryilmaz et al., (2020). NFCS was fit using all
      examples as training data (e.g., all 50 subjects' data). This generates
      a ranking of all features based on their predictiveness. 1000 SVMs were
      trained/tested using the first Z features in the ranking (for all Z from
      0 to 1000). Then, a value of Z is identified (manually based on visual
      inspection of the plotted results), which yields high accuracy and the
      standard deviation in accuracy between Z:Z+100 is low. "Z" corresponds
      to "num_feats" below.
    For the Power atlas, the resulting Z threshold is Z = 398.
    Note that the resulting estimates of accuracy are very high (near 100%
      at X > 1000) but this would not generalize. This procedure involves
      "contaminating" the test set. Yet, this is okay because this procedure
      is purely for post hoc interpretation. See the paper by Eryilmaz et al.,
      (2020) for more detail on the procedure.

    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param N: int, dataset sample size
    :param atlas: str, atlas of dataset
    '''

    X_by_ex, Y_by_ex, groups, coords, tril_idxs = load_data_2D_flat(
        N=N, dataset_num=dataset_num, atlas=atlas)
    accs = []
    l_feats = np.arange(1, 1000, 1)
    rolling_std = []
    for num_feats in tqdm(l_feats, desc='Testing NCFS cutoffs'):
        accs.append(test_NCFS_acc_contaminated(X_by_ex, Y_by_ex, groups,
                                               f'{atlas}_N{N}_d{dataset_num}',
                                               num_feats=num_feats))
        if len(accs) > 100:
            rolling_std.append(np.std(accs[-100:]))

    root_dir = pathlib.Path(__file__).parent.parent
    dir_out = fr'{root_dir}/results/NCFS'
    clear_make_dir(dir_out, clear_dir=False)

    plt.plot(l_feats, accs)
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(dir_out, f'NCFS_{atlas}_{N}_accs.png'))

    plt.plot(l_feats[100:], rolling_std)
    plt.xlim(100, 1000)
    plt.xlabel('Number of features')
    plt.ylabel('Rolling std of accuracy')
    plt.savefig(os.path.join(dir_out, f'NCFS_{atlas}_{N}_stds.png'))


if __name__ == '__main__':
    plot_NCFS_accs(dataset_num=0, N=50, atlas='power')
