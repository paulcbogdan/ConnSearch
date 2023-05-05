import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC

from ConnSearch.data_loading import load_data_2D_flat
from comparison_methods.NCFS import get_NCFS_features

'''
Unused for final report
'''


class NCSF_SVM(BaseEstimator, ClassifierMixin):
    '''
    Training this classifier involves (a) fitting a neighborhood component
        feature selector, which ranks each feature based on the degree of
        within-label similarity and between-label distinctiveness (b) selecting
        the 1000 or so highest ranked edges, (c) fitting a classifier (e.g., an
        SVM) using just those highest ranked edges.
    Using this classifier, accuracy can be tested in an unbiased manner because
        the feature selector (and classifier) are trained using only training
        data. This contrasts the biased approach used during interpretation
        (see plot_erylimaz_accs)
    '''
    def __init__(self, num_feats):
        self.num_feats = num_feats
        self.top_feats = None
        self.clf = SVC(kernel='linear')

    def fit(self, X, y):
        ncfs, coef_rankings = get_NCFS_features(X, y)
        # np.random.shuffle(coef_rankings)
        self.top_feats = coef_rankings[:self.num_feats]
        X_pruned = X[:, self.top_feats]
        self.clf.fit(X_pruned, y)

    def predict(self, X, y=None):
        X_pruned = X[:, self.top_feats]
        return self.clf.predict(X_pruned)


def test_NCFS_acc_uncontaminated(dataset_num=0, N=50, atlas='power', n_folds=2,
                                 n_repeats=1, num_feats=1000):
    '''
    As part of preliminary analyses, we also examined the unbiased accuracy
      of a classifier based on NCFS + SVM. This classifier used only training
      data for fitting the NCFS model and SVM training.

    :param n_folds: int, number of folds for cross-validation
    :param n_repeats: int, number of times to repeat cross-validation
    :param num_feats: int, cutoff for top NCFS features to use
    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param N: int, dataset sample size
    :param atlas: str, atlas of dataset
    '''
    X_by_ex, Y_by_ex, groups, coords, tril_idxs = load_data_2D_flat(
        N=N, dataset_num=dataset_num, atlas=atlas)
    X = X_by_ex.reshape(-1, X_by_ex.shape[-1])
    Y = Y_by_ex.reshape(-1)
    clf = NCSF_SVM(num_feats)
    cv = RepeatedStratifiedGroupKFold(n_splits=n_folds, n_repeats=n_repeats,
                                      random_state=0)
    acc = cross_val_score(clf, X, Y, cv=cv, groups=groups,
                          scoring='accuracy', verbose=1)
    print(f'{clf=}')
    mean_acc = np.mean(acc)
    print(f'{mean_acc=}')