import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import custom modules
import os

from sklearn.svm import SVC

from ConnSearch.ConnSearcher import ConnSearcher
from ConnSearch.data_loading import load_data_5D
from ConnSearch.components import get_components
from ConnSearch.names import get_save_name
from ConnSearch.reporting.plotting_ConnSearch import plot_components, plot_ConnSearch_ROI_scores
from ConnSearch.reporting.ConnSearch_tables import generate_component_table
from ConnSearch.permutation.permutation_manager import Permutation_Manager
from ConnSearch.utils import clear_make_dir, colors


def run_group_ConnSearch(n_splits=2, n_repeats=10, acc_thresh=None,
                         override_existing=True,
                         proximal=False, wu_analysis=False, no_components=False,
                         comp_size=16, N=10, dataset_num=0, atlas='power',
                         make_component_plots=True,
                         make_table=True,
                         make_ROI_plots=True):
    '''
    Runs a ConnSearch group-level analysis. This function (a) prepares a
        directory to save results and plots, (b) loads data, (c) defines
        components, (d) creates the ConnSearcher object, (e) loads the
        permutation-testing accuracy threshold via Permutation_Manager, (f)
        runs the ConnSearcher object's group-level analysis, (g) plots every
        significant component, Figure 3, (h) generates a table detailing
        significant components, Table 1, and (i) creates the ROI plots, showing
        the accuracy of each ROI's corresponding component, Figure 5.

    Many of these steps are optional. Additionally, instead of being used for
        a ConnSearch analysis, the ConnSearcher object can also be used for the
        analytic technique described in Wu et al. (2021; Cerebral Cortex).
        However, this feature was not used for the final manuscript.

    :param n_splits: int, number of cross-validation folds.
    :param n_repeats: int, number of cross-validation repeats. The type of cross
        validation used is repeated k-fold group stratified cross-validation.
    :param acc_thresh: float None, or 0. If None, the accuracy threshold for
        significance is determined via permutation-testing. If 0, all components
        are considered significant (i.e., all components' results are saved)
    :param override_existing: bool, if True, the directory where results are
        saved will be overwritten
    :param proximal: bool, if True, components are defined based on proximity
        to a Core ROI. Not used for the final manuscript beyond a footnote.
    :param wu_analysis: bool, if True, the Wu et al. (2021) approach or a
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
    :param make_component_plots: bool, if True, plots significant components
    :param make_table: bool, if True, generates table of significant components
    :param make_ROI_plots: bool, if True, generates ROI plots
    :return:
    '''
    assert not no_components or wu_analysis, f'Can\'t do ConnSearch if: {no_components=}'

    # dir_results is the directory where results will be saved.
    #   Name is generated from parameters. For example:
    # dir_results = ".../ConnSearch_Group/power_set16_nspl5_nrep10_N50_d0/data"
    dir_results = get_save_name(atlas=atlas, proximal=proximal,
                                no_components=no_components,
                                wu_analysis=wu_analysis, comp_size=comp_size,
                                N=N, dataset_num=dataset_num, n_folds=n_splits,
                                n_repeats=n_repeats, acc_thresh=acc_thresh,
                                rsa=False)
    clear_make_dir(dir_results, clear_dir=override_existing)

    # Load data. 5D np.array with shape = (#subjects, #sessions,
    #                                      #examples-per-session, #ROIs, #ROIs)
    X, Y, coords, _ = load_data_5D(N=N, dataset_num=dataset_num, atlas=atlas)
    # Define components. The manuscript only defined components based on
    #   connectivity but alludes to the possibility of defining components
    #   based on proximity to a Core ROI (if proximal=True). We also implemented
    #   the technique by Wu et al. (2021; Cerebral Cortex), which is used by
    #   setting no_components=True.
    components = get_components(proximal=proximal, no_components=no_components,
                                X=X, comp_size=comp_size, coords=coords)

    # Defining ConnSearcher. This involves passing it a sklearn-style classifier
    #   which will be trained/tested for each component. We used an SVM
    #   See the ConnSearcher object.
    clf = SVC(kernel='linear')
    CS = ConnSearcher(X, Y, coords, components, dir_results, n_splits=n_splits,
                      n_repeats=n_repeats, clf=clf, wu_analysis=wu_analysis)

    # Run group-level analysis. acc_thresh determines which component results
    #   are saved. If acc_thresh=None, then components are saved if their
    #   accuracy surpasses the threshold found by permutation-testing (p < .05).
    # If acc_thresh is a float (0.0-1.0), components will be saved if their
    #   accuracy surpasses that threshold. If acc_thresh=0, then all components
    #   are saved (useful for plotting all component results, e.g., Figure X)
    if acc_thresh is None:
        # See permutation.Permutation_Manager. This class manages the
        #   permutation-testing. The actual permutation-testing, which takes
        #   over an hour usually, is done elsewhere (see run_permutations.py). Here, the
        #   PM object simply retrieves those cached permutation-testing results.
        #   The retrieved results account for the current dataset size,
        #   cross-validation, etc. settings. These settings are retrieved from
        #   the ConnSearcher object (CS).
        root_dir = pathlib.Path(__file__).parent.parent
        PM = Permutation_Manager(connsearcher=CS, n_perm=1000,
                                 cache_dir=f'{root_dir}/ConnSearch/'
                                           f'permutation/permutation_saves')
        # Based on permutation-testing, the p-value associated with a given
        #   accuracy can be calculated. PM creates a function acc2p, which
        #   takes an accuracy and returns the associated p-value. The opposite
        #   is done by p2acc. An error will be raised if permutation-testing
        #   results are not found for the current ConnSearcher settings.
        #   Note that the p-values correspond to FWE corrected p-values.
        acc2p, p2acc, _ = PM.get_acc_pval_funcs()
        print(f'\nAccuracy threshold: '
              f'{colors.GREEN}{p2acc(0.05):.1%}{colors.ENDC}\n')
        CS.do_group_level_analysis(acc2p=acc2p)  # Run group-level analysis
        # save components where p < .05
    else:  # You can also specify a threshold directly. This is viable if you
        #   want quick results and do not want to wait for permutation-tests.
        CS.do_group_level_analysis(acc_thresh=acc_thresh)

    # At this point, dir_results will be populated with .pkl files. Each
    #   .pkl file contains the results for a single saved component.
    # The upcoming functions plot the results and/or make a table of results.
    # Each of these functions below operate by loading .pkls from dir_results,
    #    then generating a .csv or .png(s). These are saved nearby dir_results.

    if make_component_plots:
        dir_pics = os.path.join(pathlib.Path(dir_results).parent, 'pics')
        plot_components(dir_results, dir_pics)

    if make_table:
        fn_csv = f'ConnSearch_Group_{atlas}_{N}.csv'
        dir_table = pathlib.Path(dir_results).parent
        fp_csv = os.path.join(dir_table, fn_csv)
        generate_component_table(fp_csv, dir_results)

    if make_ROI_plots:
        dir_pics = os.path.join(pathlib.Path(dir_results).parent, 'pics')
        fp_ROI_plots = os.path.join(dir_pics, 'ROI_plots.png')
        plot_ConnSearch_ROI_scores(dir_results, fp_ROI_plots, group_level=True)


def run_group_Power_50_dataset(comp_size=16):
    '''
    All the parameters are set for the group-level analysis of the
        50-participant Power atlas dataset. This is used to create Figure 3.
    A component is only saved and plotted if its classifier's accuracy surpasses
        the p < .05 threshold found by permutation-testing.

    :param comp_size: int, number of ROIs per component
    '''
    run_group_ConnSearch(n_splits=5,
                         n_repeats=10,
                         N=50,
                         dataset_num=0,
                         comp_size=comp_size,
                         atlas='power',
                         acc_thresh=None,  # Uses permutation-testing threshold
                         make_component_plots=True,
                         make_table=True,
                         make_ROI_plots=False,
                         override_existing=False
                         )


def run_group_Schaefer1000_250_dataset(comp_size=16):
    '''
    All the parameters are set for the group-level analysis of the
        250-participant Schaefer atlas dataset. This is used to create Figure 7.
    For this analysis, we save the results for all components, regardless of
        their accuracy, as both accurate and inaccurate results are plotted
        on the glass brain. Permutation-testing is not needed to be done
        before this to establish a significant accuracy threshold, although it
        may still be informative.

    :param comp_size: int, number of ROIs per component
    '''
    run_group_ConnSearch(n_splits=5,
                         n_repeats=10,
                         N=250,
                         dataset_num=0,
                         comp_size=comp_size,
                         atlas='schaefer1000',
                         acc_thresh=0,  # Save all components for ROI plotting
                         make_component_plots=False,
                         make_table=False,
                         make_ROI_plots=True,
                         override_existing=False
                         )


def run_Wu_Power_50_dataset(comp_size=None):
    '''
    This was not used for the analysis but shows how the code can be used to
        conduct the analysis by Wu et al. (2021, Cerebral Cortex).

    :param comp_size: int, number of ROIs per component; or None if you want
        to do the original Wu analysis, which fits classifiers based on all
        edges to a given ROI.
    '''
    run_group_ConnSearch(n_splits=5,
                         n_repeats=10,
                         N=50,
                         dataset_num=0,
                         wu_analysis=True,
                         comp_size=comp_size,
                         atlas='power',
                         acc_thresh=.633,  # Uses permutation-testing threshold
                         make_component_plots=True,
                         make_table=True,
                         make_ROI_plots=False,
                         override_existing=False
                         )


if __name__ == '__main__':
    # run_Wu_Power_50_dataset()
    run_group_Power_50_dataset()
