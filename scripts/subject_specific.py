import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import custom modules

import os

from ConnSearch.reporting.ConnSearch_tables import generate_component_table
from ConnSearch.ConnSearcher import ConnSearcher
from ConnSearch.data_loading import load_data_5D
from ConnSearch.components import get_components
from ConnSearch.reporting.plotting_ConnSearch import plot_components, plot_ConnSearch_ROI_scores
from ConnSearch.names import get_save_name
from ConnSearch.utils import clear_make_dir


def run_subject_ConnSearch(alpha_thresh=.05, override_existing=True, fwe=True,
                           proximal=False, wu_analysis=False,
                           no_components=False,
                           comp_size=16, N=10, dataset_num=0, atlas='schaef',
                           subtract_mean=False,
                           make_component_plots=True,
                           make_table=True,
                           make_ROI_plots=True):
    '''
    Runs a ConnSearch subject-specific analysis. This function (a) prepares a
        directory to save results and plots, (b) loads data, (c) defines
        components, (d) creates the ConnSearcher object, (e)
        runs the ConnSearcher object's subject-specific analysis, (g) plots
        every significant component, Figure 7 top, (h) generates a table
        detailing significant components, Table 3, and (i) creates the ROI
        plots, showing the accuracy of each ROI's corresponding component,
        Figure 7 bottom.

    :param alpha_thresh: float, corrected p-value threshold for significance
    :param override_existing: bool, if True, the directory where results are
        saved will be overwritten
    :param fwe: bool, if True, uses Holm-Sidak (FWE) correction. If False,
        uses Benjamani-Hochberg (FDR) correction.
    :param proximal: bool, if True, components are defined based on proximity
        to a Core ROI. Not used for the final manuscript.
    :param wu_analysis:bool, if True, the Wu et al. (2021) approach or a
        modified version, which uses only edges within the component, is used.
        Not used for the manuscript.
    :param no_components: bool, if True, does the original Wu method, where
        classifiers are fit based on every single connection to an ROI.
        Not used for the manuscript.
    :param comp_size: int, number of ROIs in the component
    :param N: int, number of subjects
    :param dataset_num: int, which dataset to use. The manuscript needed this to
        select amongst the five 50-participant datasets.
    :param atlas: str, which atlas to use ("power" or "schaefer1000")
    :param subtract_mean: bool, if True, subtracts the mean group-level 2-back
        vs. 0-back connectivity from each edge before subject-specific analysis.
        See Supplemental Methods 1.3.
    :param make_component_plots: bool, if True, plots significant components
    :param make_table: bool, if True, generates table of significant components
    :param make_ROI_plots: bool, if True, generates ROI plots
    :return:
    '''

    # Directory where results will be saved. Name is generated from parameters.
    #    e.g., "\results_data\ConnSearch_Group\power_set16_nspl5_nrep10_N50_d0"
    dir_results = get_save_name(atlas=atlas, proximal=proximal,
                                no_components=no_components,
                                wu_analysis=wu_analysis,
                                comp_size=comp_size, N=N,
                                dataset_num=dataset_num,
                                rsa=True,
                                fwe=fwe,
                                alpha_thresh=alpha_thresh,
                                subtract_mean=subtract_mean)
    clear_make_dir(dir_results, clear_dir=override_existing)

    # Load data. 5D np.array with shape = (#subjects, #sessions,
    #                                      #examples-per-session, #ROIs, #ROIs)
    X, Y, coords, name = load_data_5D(N=N, dataset_num=dataset_num,
                                      atlas=atlas)

    # Define components. The manuscript only defined components based on
    #   connectivity but alludes to the possibility of defining components
    #   based on proximity to a Core ROI (if proximal=True). We also implemented
    #   the technique by Wu et al. (2021; Cerebral Cortex), which is used by
    #   setting no_components=True.
    components = get_components(proximal=proximal, no_components=no_components,
                                X=X, comp_size=comp_size, coords=coords)

    CS = ConnSearcher(X, Y, coords, components, dir_results,
                      wu_analysis=wu_analysis)
    CS.do_subject_specific_analysis(alpha_thresh, FWE=fwe,
                                    subtract_mean=subtract_mean)

    if make_table:
        fn_csv = f'ConnSearch_Subject-specific_{atlas}_{N}.csv'
        dir_table = pathlib.Path(dir_results).parent
        fp_csv = os.path.join(dir_table, fn_csv)
        generate_component_table(fp_csv, dir_results)

    if make_component_plots or make_ROI_plots:
        dir_pics = os.path.join(pathlib.Path(dir_results).parent, 'pics')
        if make_component_plots:
            plot_components(dir_results, dir_pics)

        if make_ROI_plots:
            fp_ROI_plots = os.path.join(dir_results, 'ROI_plots.png')
            plot_ConnSearch_ROI_scores(dir_results, fp_ROI_plots,
                                       group_level=True,
                                       vmin=2.5, vmax=5.5, avg_ROIs=True)
            # If avg_ROIs=True, the plots assign ROIs scores as the average of
            #   all the components they contributed to.
            # If avg_ROIs=False, assigns ROIs scores based solely on the
            #   component for which they are the Core ROI.


def run_subject_Power_50_dataset(comp_size=16, subtract_mean=False):
    '''
    All the parameters are set for the subject-specific analysis of the
        50-participant Power atlas dataset.
    :param comp_size: int, number of ROIs per component
    '''
    run_subject_ConnSearch(N=50,
                           dataset_num=0,
                           comp_size=comp_size,
                           atlas='power',
                           fwe=True,
                           alpha_thresh=.05,
                           subtract_mean=subtract_mean,
                           make_component_plots=False,
                           make_table=True,
                           make_ROI_plots=False)


def run_subject_Schaefer1000_250_dataset(comp_size=16, subtract_mean=False):
    '''
    All the parameters are set for the subject-specific analysis of the
        50-participant Power atlas dataset.
    :param comp_size: int, number of ROIs per component
    '''
    run_subject_ConnSearch(N=250,
                           dataset_num=0,
                           comp_size=comp_size,
                           atlas='schaefer1000',
                           fwe=False,
                           alpha_thresh=1,  # Save all components for ROI plotting
                           subtract_mean=subtract_mean,
                           make_component_plots=False,
                           make_table=False,
                           make_ROI_plots=True)


if __name__ == '__main__':
    run_subject_Power_50_dataset()
    # run_subject_Schaefer1000_250_dataset()
