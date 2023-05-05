import os
import pickle
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ConnSearch.reporting.plotting_general import plot_pkg, plot_ROI_scores
from ConnSearch.utils import clear_make_dir

def plot_components(dir_results, dir_pics, clear_dir=True,
                    subject_specific=False):
    '''
    Plots the results saved in dir_results. ConnSearch saves each component
        result as a separate .pkl file in dir_results (pkg), and the present
        function loops over all .pkl files in dir_results, loads them, then
        plots them. This function saves the plots in dir_pics.
    The results will be plotted in multiple styles.
    This function can also be used to plot the significant "Wu-components"
        (vector of edges connected to a single ROI).

    :param dir_results: Directory containing the .pkl results files
    :param dir_pics: Directory to save the plots
    :param clear_dir: Whether to clear dir_pics before plotting
    :param subject_specific: bool, if True sets -1 and 1 as vmin/vmax for colors
    :return:
    '''

    clear_make_dir(dir_pics, clear_dir=clear_dir)
    clear_make_dir(os.path.join(dir_pics, 'glass'), clear_dir=clear_dir)
    clear_make_dir(os.path.join(dir_pics, 'chord'), clear_dir=clear_dir)

    chord_params = {}
    if subject_specific:
        chord_params['vmin'] = -1
        chord_params['vmax'] = 1
        glass_params = {'vmin': -1, 'vmax': 1}
    else:
        glass_params = {}


    chord_black_params = {'black_BG': True}

    fns = os.listdir(dir_results)
    print(f'Total number of results to plot: {len(fns)}')
    for fn in tqdm(fns, 'Plotting component results'):
        if fn[-4:] != '.pkl': # skips any other files that may mistakenly enter
            continue
        fp = os.path.join(dir_results, fn)
        with open(fp, 'rb') as f:
            pkg = pickle.load(f)
        title = f'Component Core ROI: {pkg["component"][0]}'

        # Uses plot_pkg which is a general function designed for plotting
        #   results from ConnSearch, Wu, and the connectome-wide methods.
        plot_pkg(pkg, dir_pics, fn, chord_params=chord_params,
                 glass_params=glass_params, title=title)
        plot_pkg(pkg, dir_pics, fn.replace('.pkl', '_black.pkl'),
                 chord_params=chord_black_params, title=title)

        sum_chord_params = {'alphas': 0.9, # For averaging network-pair edges
                            'plot_count': True,
                            'norm_thickness': True,
                            'plot_abs_sum': True}
        plot_pkg(pkg, dir_pics, fn.replace('.pkl', '_sum.pkl'),
                 chord_params=sum_chord_params, title=title)


def plot_ConnSearch_ROI_scores(dir_results, fp_out,
                               group_level=False,
                               vmin=None, vmax=None):
    '''
    Plots the results of all the ConnSearch component models. Each ROI is a dot
        on a glass brain, where its color indicates the score (accuracy or
        t-value) of the component for which the ROI is the Core ROI.
    :param dir_results: str, directory where results will be loaded from
    :param fp_out: str, filepath to save the plot
    :param group_level: if True, sets the default vmin/vmax for group-level
        plotting. if False, sets the default vmin/vmax for subject-level.
    :param vmin: float, minimum value for the colorbar
    :param vmax: float, maximum value for the colorbar
    :return:
    '''
    fns = os.listdir(dir_results)
    node_vals = []
    node_coords = []
    i2coord = {}
    i2accs = defaultdict(list)

    for fn in fns: # Each component result is saved in its own .pkl file
        if '.pkl' not in fn:
            continue
        fp = os.path.join(dir_results, fn)
        with open(fp, 'rb') as f:
            pkg = pickle.load(f)
        score = pkg['score']
        core_i = pkg['component'][0]
        core_roi_coord = pkg['coords_component'][0]
        i2coord[core_i] = core_roi_coord
        for j in pkg['component']:
            i2accs[j].append(score)

        node_vals.append(score)
        node_coords.append(core_roi_coord)

    node_vals_avg = []
    node_coords_avg = []
    for i in i2coord:
        node_vals_avg.append(np.mean(i2accs[i]))
        node_coords_avg.append(i2coord[i])
    node_vals = node_vals_avg
    node_coords = node_coords_avg

    if group_level:
        if vmax is None: vmax = .65
        if vmin is None: vmin = .555
    else:
        if vmax is None: vmax = 3.5
        if vmin is None: vmin = 1.7
    plot_ROI_scores(node_vals, node_coords, vmin=vmin, vmax=vmax,
                    fp_out=fp_out, title='', dpi=600)

