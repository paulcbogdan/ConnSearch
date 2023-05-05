import os
import pathlib


def get_save_name(atlas='', proximal=False, no_components=False,
                  wu_analysis=False, comp_size=16, N=10, dataset_num=0,
                  n_folds=None, n_repeats=None,
                  acc_thresh=None,
                  rsa=False, fwe=False,
                  alpha_thresh=.05,
                  subtract_mean=False):
    '''
    This function just generates a name for the directory where the ConnSearch
        results will be saved or loaded from. The name depends on the parameters

    ConnSearch will save its results (significant components) as .pkl files.
        The .pkl files are placed in a directory within \results_data\. This
        function creates the name for the directory. For example:
        "\results_data\ConnSearch_Group\power_set16_nspl5_nrep10_N50_d0"
        This name is generated based on the various parameters.

    We also implemented the technique by Wu et al. (2021; Cerebral Cortex).
        Implementing this technique required modifying ConnSearcher.py somewhat
        but overall this addition is meant to show the codebase's flexibility.

    There are many parameters to be set.
    :param atlas: str, added at the start of the directory name
    :param proximal: boolean, unused for the present manuscript
    :param no_components: boolean, related to the Wu et al. (2021) technique
    :param wu_analysis: boolean, if True uses the Wu et al. (2021) technique
    :param comp_size: int, number of ROIs in each component
    :param N: sample size
    :param dataset_num: int, dataset number. Used for replicability analysis

    --- Group-level parameters ---
    :param n_folds: int, number of folds used during cross-validation
    :param n_repeats: int, number of repeats used during cross-validation
    :param acc_thresh: float, accuracy or t-value threshold for whether a
        component will be saved. If zero, all components are saved (useful
        then plotting all component results, e.g., Figure 5). If None, uses
        threshold needed for p < .05 established via permutation-testing).

    --- Subject-specific parameters ---
    :param rsa: boolean, if True then uses subject-specific ConnSearch
    :param fwe: boolean, applies only if rsa=True. If True, uses Holm-Sidak
        (FWE) correction. If False, uses Benjamani-Hochberg (FDR) correction.
    :param alpha_thresh:
    :param subtract_mean:
    :return: string, directory name
    '''
    # Helps the code work regardless of where it is run from
    root_dir = pathlib.Path(__file__).parent.parent
    if rsa:
        n_folds, n_repeats = None, None
    atlas += '_prox' if proximal else ''
    if not wu_analysis or not no_components:
        atlas += f'_comp{comp_size}'
    atlas += f'_folds{n_folds}' if n_folds is not None else ''
    atlas += f'_reps{n_repeats}' if n_repeats is not None else ''
    atlas += f'_N{N}'
    atlas += f'_d{dataset_num}' if dataset_num != 0 else ''
    if rsa:
        atlas += '_subM' if subtract_mean else ''
        if alpha_thresh == 1:
            atlas += '_all'  # It will save all components if alpha_thresh == 1
        else:
            atlas += f'_fwe{alpha_thresh}' if fwe else f'_bh{alpha_thresh}'
        if wu_analysis:
            dir_results = fr'results/Wu_SubjSpec/{atlas}'
        else:
            dir_results = fr'results/ConnSearch_SubjSpec/{atlas}'
    else:
        if acc_thresh == 0:
            atlas += '_all'
        elif acc_thresh is not None:
            atlas += f'_acc{acc_thresh:.3f}'
        if wu_analysis:
            dir_results = fr'results/Wu_Group/{atlas}'
        else:
            dir_results = fr'results/ConnSearch_Group/{atlas}'
    print(f'Saving results to: {dir_results}\n')
    dir_results = os.path.join(root_dir, dir_results, 'out_data')
    return dir_results
