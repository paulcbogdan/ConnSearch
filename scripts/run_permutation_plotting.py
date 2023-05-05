import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import custom modules

from ConnSearch.permutation.permutation_manager import Permutation_Manager
from ConnSearch.data_loading import load_data_5D
from ConnSearch.components import get_connectivity_sets

from sklearn.svm import SVC
import matplotlib as mpl
from cycler import cycler

import io
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from copy import copy


def plot_permutation_distributions(atlas='power', N=50, n_perm=1000):
    '''
    Plots permutation-testing results, managed by Permutation_Manager.
        Plots shuffled accuracy to p-values. Different settings are compared.
        See Supplemental Figure S1.
    Permutation-testing should already have been performed before this function
        is run. This function will just use the cached results.
    The present code compares the effects of using 10 repeats vs. no repeats
        for 5-fold cross-validation on N = 50 datasets analyzed via ConnSearch.

    :param atlas: str, atlas
    :param N: int, number of subjects
    :param n_perm: int, number of permutations for each permutation-testing run
    '''

    def get_PM_settings():
        root_dir = pathlib.Path(__file__).parent.parent
        settings0 = {'comp_size': 16,  # Permutation-test parameters for setting0
                     'n_folds': 5,
                     'n_repeats': 1,
                     'wu_analysis': False,
                     'cache_dir': f'{root_dir}/ConnSearch/'
                                  f'permutation/permutation_saves',
                     'perm_strategy': 'within_subject',
                     'component_func': get_connectivity_sets}
        settings1 = copy(settings0)
        settings1['n_repeats'] = 10  # setting1 uses a different number of repeats
        settings = [settings1, settings0]
        names = ['5 splits, 10 repeats', '5 splits, no repeat']
        color_cycle = ['b', 'r']  # Color for each distribution
        return settings, names, color_cycle

    # load data, but only the dataset shape and number of coords are relevant
    X, y, coords, _ = load_data_5D(atlas=atlas, N=N)

    # prepare a list of two settings to compare:
    #   settings0['n_repeats'] = 0 vs. settings1['n_repeats'] = 10
    settings, names, color_cycle = get_PM_settings()

    # matploblib setup
    font = {'family': 'arial',
            'size': 14}
    mpl.rc('font', **font)
    mpl.rcParams['image.cmap'] = 'viridis'
    plt.rcParams['axes.prop_cycle'] = cycler(color=color_cycle)
    acc_space = np.linspace(0.5, 0.72, 1000)
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event',  # press ctrl+c to copy figure
                           lambda event: add_figure_to_clipboard(event, fig))

    # plot permutation-testing results for each setting
    for i, (s, name) in enumerate(zip(settings, names)):
        PM = Permutation_Manager(X, y, coords, SVC(kernel='linear'),
                                 n_perm=n_perm, **s)
        acc2p, p2acc, n_perms_done = PM.get_acc_pval_funcs()
        ps = list(map(acc2p, acc_space))
        ax.plot(acc_space, ps, linewidth=1, alpha=.8, label=name, zorder=5 - i)

    # plot a line representing p = 0.05
    plt.plot([min(acc_space), max(acc_space)], [0.05, 0.05], 'k--',
             linewidth=1, zorder=0)

    # matplotlib setup
    ax = plt.gca()
    ax.set_xlim((min(acc_space), max(acc_space)))
    ax.set_ylim((-.00, 1.01))
    plt.xticks([.5, .55, .6, .65, .7])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.legend(frameon=False, loc='best', fontsize=15)
    plt.xlabel('Accuracy', fontsize=16)
    plt.ylabel('p-value', fontsize=16)
    root_dir = pathlib.Path(__file__).parent.parent
    fp_out = fr'{root_dir}/ConnSearch/permutation/effect_of_repeats.png'
    plt.savefig(fp_out, dpi=300, bbox_inches='tight')
    plt.show()


def add_figure_to_clipboard(event, fig):
    '''
    Lets you conveniently copy a figure to the clipboard by pressing ctrl+c.
    Taken from: https://stackoverflow.com/questions/31607458/how-to-add-clipboard-support-to-matplotlib-figures

    :param event: matplotlib event, see usage in plot_settings(...)
    :param fig: matplotlib fig, see usage in plot_settings(...)
    :return:
    '''
    if event.key == "ctrl+c":
        with io.BytesIO() as buffer:
            fig.savefig(buffer)
            QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))


if __name__ == '__main__':
    plot_permutation_distributions()
