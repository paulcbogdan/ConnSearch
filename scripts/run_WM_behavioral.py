import pathlib

import numpy as np
import pandas as pd
import scipy.stats as stats

'''
This script reports the 2-back and 0-back accuracies for each N = 50 dataset
    It also compares accuracies between each dataset using a Wilcoxon rank-sum
    test because the data are left skewed.
'''


def compare_all(accs):
    '''
    Used to compare the accuracy between the five groups of 50 participants
    :param accs: 2D array, dim 0 = group, dim 1 = participant, values = accuracy
    '''
    print('  ---- Descriptive stats ----')
    for i, l0 in enumerate(accs):
        m = np.nanmean(l0)
        sd = np.nanstd(l0)
        print(f'dataset {i}, M = {m:.3f} [SD = {sd:>6.3f}], '
              f'N = {len(l0)}, n-nans: {np.isnan(l0).sum()}')

    ar_all = np.reshape(np.array(accs), (-1, 1))
    m_all = np.nanmean(ar_all)  # Three participants have NaN for 2-back
    sd_all = np.nanstd(ar_all)

    print(f'All groups M = {m_all:.3f} [SD = {sd_all:>6.3f}]')
    print('  --------- t-tests ---------')
    for i, l0 in enumerate(accs):
        for j, l1 in enumerate(accs):
            if i >= j:
                continue
            t, p = stats.ranksums(l0, l1)  # Wilcoxon rank-sum test
            print(f' {i} vs. {j}, t = {t:>6.3f}, p = {p:.3f}')


if __name__ == '__main__':
    WM_accs = []
    WM_2bk_accs = []
    WM_0bk_accs = []

    root_dir = pathlib.Path(__file__).parent.parent
    for i in range(5):
        fn = fr'{root_dir}/in_data/behavioral_data/behavioral_d{i}.csv'
        df_i = pd.read_csv(fn)
        WM_accs.append(df_i['WM_Task_Acc'].values)
        WM_2bk_accs.append(df_i['WM_Task_2bk_Acc'].values)
        WM_0bk_accs.append(df_i['WM_Task_0bk_Acc'].values)

    print('\nScores for 0-back & 2-back average')
    compare_all(np.array(WM_accs))
    print('\n\n\t   Scores for 2-back')
    compare_all(np.array(WM_2bk_accs))
    print('\n\n\t   Scores for 0-back')
    compare_all(np.array(WM_0bk_accs))
