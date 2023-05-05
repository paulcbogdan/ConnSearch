import os

import numpy as np
from matplotlib import pyplot as plt, colors as mcolors

from nichord import convert_matrix, plot_chord, plot_glassbrain, combine_imgs, get_idx_to_label
from nilearn import plotting


def plot_pkg(pkg, dir_out, fn_out, chord_params=None,
             glass_params=None, title=None, cmap=None):
    '''
    Plots a given result, specified by pkg. This is used for both the ConnSearch
        components and the connectome-wide edge results.
    Plotting is, honestly, a bit slow and will often take more time than running
        ConnSearch itself.

    :param pkg: dict, contains the necessary info to plot a given result
    :param dir_out: str, specifies where the plot should be saved
    :param fn_out: str, specifies the filename of the plot
    :param chord_params: parameters for the chord diagram
    :param glass_params: parameters for the glass brain plot
    :param title: title for the glass + chord combined plot
    :param cmap: colormap to use for the plots
    '''
    if chord_params is None: chord_params = {}
    if glass_params is None: glass_params = {}
    print(f'Plotting: {fn_out=}')
    edges = pkg['edges']
    idx_to_label, network_order, network_colors = \
        get_network_info(pkg['coords_all'])

    if pkg['edge_weights'] is None:
        _, edge_weights = convert_matrix(pkg['adj']) # flattens matrix
    else:
        edge_weights = pkg['edge_weights'] # list of weights, length = #edges

    if ('code' in pkg) and ('wu' in pkg['code']):
        is_wu = True # appearance of plots changed slightly for Wu results
                     # (Wu results are, again, not part of the manuscript but
                     #  were probed for a peer review comment)
    else:
        is_wu = False

    dir_chord = os.path.join(dir_out, 'chord')
    fp_chord = os.path.join(dir_chord, fn_out.replace('.pkl', '_chord.png'))
    dir_glass = os.path.join(dir_out, 'glass')
    fp_glass = os.path.join(dir_glass, fn_out.replace('.pkl', '_glass.png'))
    fp_combined = os.path.join(dir_out, fn_out.replace('.pkl', '_combined.png'))

    # See NiChord (https://github.com/paulcbogdan/NiChord) for plotting details
    if cmap is None:
        cmap = 'turbo' if not all(w == 1 for w in edge_weights) else 'Greys'
    plot_chord(idx_to_label, edges, edge_weights=edge_weights,
               fp_chord=fp_chord,
               arc_setting=False if is_wu else True, # Helps Wu plots look nicer
               network_order=network_order, coords=pkg['coords_all'],
               network_colors=network_colors, cmap=cmap,
               **chord_params)

    plot_glassbrain(idx_to_label, edges, edge_weights, fp_glass,
                    pkg['coords_all'], network_colors=network_colors,
                    cmap=cmap,
                    **glass_params)
    combine_imgs(fp_glass, fp_chord, fp_combined, title=title)

    fp_combined = fp_combined[:-4] + '_1' + fp_combined[-4:]
    combine_imgs(fp_glass, fp_chord, fp_combined, title=title, only1glass=True,
                 fontsize=75 if (len(title) > 50) else 82)
    print(f'\tPlotted: {fp_combined=}')
    plt.close()


def get_network_info(coords):
    '''
    Returns info about the ROI network labels and some settings for plotting

    :param coords: list of ROI coordinates. shape (#ROIs, 3)
    :return:
        idx_to_label, dict mapping each ROI's index to a network label
        network_order, list of network labels in order to be listed on diagrams
        network_colors, dict mapping each network label to a color
    '''
    network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                     'Uncertain', 'VAN']
    network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                      'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                      'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

    idx_to_label = get_idx_to_label(coords, atlas='yeo',
                                    search_closest=True)
    labels = set(idx_to_label.values())
    do_pop = []
    for network in network_colors:
        if network not in labels:
            do_pop.append(network)
    for network in do_pop:
        del network_colors[network]
        network_order.remove(network)

    return idx_to_label, network_order, network_colors


def plot_ROI_scores(node_vals, node_coords, vmin=None, vmax=None,
                    fp_out=None, title='', show=False, dpi=600):
    '''
    Used for generating those plots where each ROI is a dot on a glass brain,
        where the dot's color indicates its score. Score can be the accuracy of
        its corresponding components (ConnSearch) or the number of highly
        predictive edges it is connected to (RFE, NCFS, CPM). This is a
        general function used for plotting the results of all the methods.

    :param node_vals: list of floats, the score for each ROI
    :param node_coords: list of tuples, the coordinates of each ROI
    :param vmin: float, minimum value for the colorbar
    :param vmax: float, maximum value for the colorbar
    :param fp_out: str, filepath to save the plot
    :param title: str, title of the plot
    :param show: bool, whether to show the plot immediately via
        matplotlib.pyplot.show(). Regardless of this setting, the plot is saved.
    :param dpi: int, resolution of the plot
    :return:
    '''
    if show: dpi = dpi * 2.5/6
    print(f'Plotting ROI scores glass brains: {fp_out=}')

    fig = plt.figure(figsize=(8, 2.79), dpi=dpi)

    node_vals_, node_coords_ = zip(*sorted(zip(node_vals, node_coords),  # plots
                                           key=lambda x: x[0])) # highest on top

    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap',
                                    plt.cm.CMRmap(np.linspace(0.05, 1, 1000)))

    # plots all three glass brain angles (sagittal, coronal, axial)
    plotting.plot_markers(node_vals_, node_coords_,
                          node_cmap=mymap, alpha=1,
                          node_vmin=vmin, node_vmax=vmax,
                          black_bg=True,
                          node_size=3.25,
                          title=title,
                          figure=fig)
    plt.savefig(fp_out)
    if show: plt.show()

    # plots only the sagittal view of the glass brain
    fig = plt.figure(figsize=(4, 4 / (2.6/2.3) * .71), dpi=dpi)
    plotting.plot_markers(node_vals_, node_coords_,
                          node_cmap=mymap, alpha=1,
                          node_vmin=vmin, node_vmax=vmax,
                          black_bg=True,
                          node_size=1.75,
                          title=title,
                          display_mode='x',
                          figure=fig)

    if show: plt.show()
    fp_x = fp_out[:-4] + '_x.png'
    plt.savefig(fp_x)
    plt.clf()
