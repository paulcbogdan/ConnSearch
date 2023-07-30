import warnings
from collections import defaultdict

import numpy as np
import sklearn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d
from sklearn.utils.multiclass import type_of_target

'''
In preparing this research, we identified an issue with
    sklearn.model_selection.StratifiedGroupKFold. It wasn't achieving optimal 
    stratification. The present code provides ConnSearch_StratifiedGroupKFold,
    which yields better stratification.
The code also provides RepeatedStratifiedGroupKFold, which repeats group k-fold
    then averages across the repetitions, much like sklearn's other repeated-
    k-fold classes.
'''


class RepeatedStratifiedGroupKFold(_RepeatedSplits):
    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            ConnSearch_StratifiedGroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )


class ConnSearch_StratifiedGroupKFold(StratifiedGroupKFold):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (n_smallest_class, self.n_splits),
                UserWarning,
            )
        n_classes = len(y_cnt)



        groups_unique, groups_inv, groups_cnt = np.unique(
            groups, return_inverse=True, return_counts=True
        )

        if self.shuffle:
            group2shuffled = np.array(range(len(groups_unique)))
            rng.shuffle(group2shuffled)
            groups_inv = group2shuffled[groups_inv]

        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)
        sorted_groups_idx = np.argsort(
            -np.std(y_counts_per_group, axis=1), kind="mergesort"
        )

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold,
                y_cnt=y_cnt,
                group_y_counts=group_y_counts,
            )
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        if not self._check_stratification_ideal(y_counts_per_fold, warn=False):
            self._optimize_stratification(y_counts_per_fold, y_cnt,
                                          y_counts_per_group,
                                          groups_per_fold)
        for i in range(self.n_splits):
            test_indices = [
                idx
                for idx, group_idx in enumerate(groups_inv)
                if group_idx in groups_per_fold[i]
            ]
            yield test_indices

    def _optimize_stratification(self, y_counts_per_fold, y_cnt,
                                 y_counts_per_group, groups_per_fold):
        # print('Optimizing stratification...')
        groups = np.array(range(y_counts_per_group.shape[0]))
        np.random.shuffle(groups)
        for group0 in groups:
            for fold0, groups_ in groups_per_fold.items():
                if group0 in groups_:
                    break
            for group1 in groups:
                if group0 == group1:
                    continue
                for fold1, groups_ in groups_per_fold.items():
                    if group1 in groups_:
                        break
                std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1),
                                       axis=0)

                y_counts_per_fold[fold0] -= y_counts_per_group[group0]
                y_counts_per_fold[fold1] += y_counts_per_group[group0]
                y_counts_per_fold[fold1] -= y_counts_per_group[group1]
                y_counts_per_fold[fold0] += y_counts_per_group[group1]

                std_per_class_new = np.std(
                    y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
                if np.mean(std_per_class_new) < np.mean(std_per_class):
                    groups_per_fold[fold0].remove(group0)
                    groups_per_fold[fold1].remove(group1)
                    groups_per_fold[fold1].add(group0)
                    groups_per_fold[fold0].add(group1)
                    if self._check_stratification_ideal(y_counts_per_fold,
                                                        warn=False):
                        # If the stratification is ideal, then looping stops
                        return
                    else:
                        # If the stratification is not, it runs again
                        return self._optimize_stratification(y_counts_per_fold,
                                                             y_cnt,
                                                             y_counts_per_group,
                                                             groups_per_fold)
                else:
                    y_counts_per_fold[fold0] += y_counts_per_group[group0]
                    y_counts_per_fold[fold1] -= y_counts_per_group[group0]
                    y_counts_per_fold[fold1] += y_counts_per_group[group1]
                    y_counts_per_fold[fold0] -= y_counts_per_group[group1]
        print('Failed to achieve ideal stratification')

    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = (
                fold_eval < min_eval
                or np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold

    def _check_stratification_ideal(self, y_counts_per_fold, warn=False):
        if y_counts_per_fold.shape[0] > 1:
            first_fold = y_counts_per_fold[0]
            for fold in y_counts_per_fold[1:]:
                if not np.array_equal(first_fold, fold):
                    if warn:
                        warnings.warn('Folds are not perfectly stratified.',
                                  UserWarning)
                    return False
        return True