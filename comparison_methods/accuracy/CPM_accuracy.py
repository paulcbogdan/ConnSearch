import numpy as np
from scipy import stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC

from ConnSearch.data_loading import load_data_2D_flat

'''
Unused for final report
'''

class CPM(BaseEstimator, ClassifierMixin):
    '''
    Training this classifier involves: (a) performing a paired t-test for every
        connectivity edge, comparing the first condition (2-back) to the second
        condition (0-back). The t-tests are used to generate p-values associated
        with each edge. (b) Feature selection is done by omitting edges where
        p > .05 (uncorrected). (c) Using only the selected features, an SVM is
        trained.
    Note that to properly assess this classifier's accuracy, all three steps
        must be performed for each cross-validation split. Thus, the feature
        selected via t-tests will vary slightly between splits. Although our
        manuscript did not focus on accuracy, this was measured as part of
        preliminary analyses. For our actually reported interpretation (the
        plotted results), no cross-validation was performed and interpretation
        amounts to simply conducting paired t-tests and no machine learning.
    To conform this classifier to the standard sklearn style, its code is
        surprisingly intricate. The biggest challenge comes from the fact that
        paired t-tests are used, which means that the classifier must have info
        about which examples are from the same participant (i.e., the same
        group). To get around this, the last column of X is used as a code
        representing the example's group (e.g., 0 = first participant,
        1 = second participant, etc.).
    '''
    def __init__(self, clf, alpha_thresh=0.05):
        '''
        :param clf: sklearn clf (eg SVM) to be applied after feature selection
        :param alpha_thresh: float, p-value threshold to filter based on t-tests
        '''
        self.signif_edges = None # boolean array
        self.clf = clf
        self.alpha_thresh = alpha_thresh
        self.coef_ = None

    def fit(self, X, y=None):
        self.set_signif_edges(X, y) # run the paired t-tests and filter by p-val
        X = np.delete(X, -1, axis=1) # delete the group-indicator column

        X_pruned = X[:, self.signif_edges] # examine only significant edges
        X_final = X_pruned.reshape(-1, X_pruned.shape[-1])
        Y_final = y.reshape(-1)
        self.clf.fit(X_final, Y_final)
        if hasattr(self.clf, 'coef_'): # some sklearn clf will not have .coef_
            self.coef_ = self.clf.coef_

    def predict(self, X):
        X = np.delete(X, -1, axis=1)
        X_pruned = X[:, self.signif_edges]
        X_final = X_pruned.reshape(-1, X_pruned.shape[-1])
        return self.clf.predict(X_final)

    def set_signif_edges(self, X, y):
        groups_idxs = X[:, -1]
        X = np.delete(X, -1, axis=1)  # delete the group-indicator column

        # For paired t-tests, X must be averaged on a subject-by-subject basis,
        #   separately for y = 0 or y = 1
        X0_flat_subj_avg = avg_by_subj_via_groups(X, y, 0, groups_idxs)
        X1_flat_subj_avg = avg_by_subj_via_groups(X, y, 1, groups_idxs)

        # Then perform a paired t-test for each edge
        M_dif = np.mean(X1_flat_subj_avg - X0_flat_subj_avg, axis=0)
        std_dif = np.std(X1_flat_subj_avg - X0_flat_subj_avg, axis=0, ddof=1)
        se_dif = std_dif / np.sqrt(X0_flat_subj_avg.shape[0])
        t_vals = M_dif / se_dif

        # Filter by p-value
        t_cutoff = stats.t.ppf(1 - self.alpha_thresh / 2, # convert alpha to a
                               X0_flat_subj_avg.shape[0] - 1) # t-val threshold

        signif_edges = np.abs(t_vals) > t_cutoff
        self.signif_edges = signif_edges
        if np.sum(signif_edges) < 1:
            print('No significant edges found')
            raise ValueError
        else:
            n_signif = np.sum(signif_edges)
            print(f'{n_signif} significant edges found')


def avg_by_subj_via_groups(X_by_ex, Y_by_ex, label, groups):
    '''
    Extracts examples for a given label, then averages the examples by subject
        (group). This is necessary for paired t-tests, which involve averaging
        by subject.

    :param X_by_ex: array where the subjects, sessions, and examples are
        collapsed into just the 0th dimension (e.g., X_2D)
    :param Y_by_ex: 1D array indicating each example's label
    :param label: int, label to average within
    :param groups: groups, list mapping each example to a given subject (group)
    :return:
    '''
    X0 = X_by_ex[Y_by_ex == label]
    groups0 = groups[Y_by_ex == label]

    idx_cutoffs_dif = np.diff(groups0)
    idx_cutoffs = np.argwhere(idx_cutoffs_dif)
    idx_cutoffs = idx_cutoffs.reshape(-1) + 1

    X0_subj_avg = np.array([a.mean(0) for a in np.split(X0, idx_cutoffs)])
    return X0_subj_avg


def test_CPM_accuracy(alpha_thresh=.05, N=50, dataset_num=0, n_folds=5,
                      n_repeats=10, atlas='power'):
    '''
    CPM accuracy was not reported in the manuscript but was tested as part of
        preliminary analyses.

    :param alpha_thresh: float, p-value threshold for edge selection
    :param N: int, dataset sample size
    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param n_folds: int, number of folds for cross-validation
    :param n_repeats: int, number of times to repeat cross-validation
    :param atlas: str, atlas of dataset
    '''
    X_by_ex, Y_by_ex, groups_idxs, _, _ = \
        load_data_2D_flat(N=N, dataset_num=dataset_num, atlas=atlas)

    # Add groups as the last column of X_by_ex. This allows groups to be
    #   incorporated into fitting the CPM classifier, which is needed for the
    #   paired t-tests.
    X_by_ex = np.append(X_by_ex, np.expand_dims(groups_idxs, 1), axis=1)

    # Evaluate the CPM classifier using the cross-validation as for ConnSearch
    clf = CPM(SVC(kernel='linear'), alpha_thresh=alpha_thresh)
    cv = RepeatedStratifiedGroupKFold(n_splits=n_folds, n_repeats=n_repeats,
                                      random_state=2)
    acc = cross_val_score(clf, X_by_ex, Y_by_ex, cv=cv, groups=groups_idxs,
                          scoring='accuracy')
    print(f'{alpha_thresh=:.3f}')
    print('Mean accuracy: {:.1%} +/- {:.1%}'.format(np.mean(acc), np.std(acc)))

if __name__ == '__main__':
    test_CPM_accuracy()