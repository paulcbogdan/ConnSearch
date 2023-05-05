import numpy as np
from sklearn.model_selection import RepeatedStratifiedGroupKFold, cross_val_score

from ConnSearch.data_loading import load_data_2D_flat
from comparison_methods.RFE import RFE_SVM

'''
Unused for final report
'''

def test_RFE_accuracy(dataset_num=0, N=50, atlas='power', n_edges=1000,
                      rfe_step=.01, n_splits=5, n_repeats=100):
    X_by_ex, Y_by_ex, groups, coords, tril_idxs = load_data_2D_flat(
        atlas=atlas, N=N, dataset_num=dataset_num)
    X = X_by_ex.reshape(-1, X_by_ex.shape[-1])
    Y = Y_by_ex.reshape(-1)

    clf = RFE_SVM(n_edges, rfe_step=rfe_step)
    cv = RepeatedStratifiedGroupKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=0)
    acc = cross_val_score(clf, X, Y, cv=cv, groups=groups,
                          scoring='accuracy', verbose=1)
    print(f'{clf=}')
    mean_acc = np.mean(acc)
    print(f'{rfe_step=}')
    print(f'{mean_acc=}')
