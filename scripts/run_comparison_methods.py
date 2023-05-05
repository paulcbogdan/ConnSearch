import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import modules

from comparison_methods.CPM import plot_CPM
from comparison_methods.NBS import plot_NBS
from comparison_methods.NCFS import plot_NCFS
from comparison_methods.NCFS_feature_cutoff import plot_NCFS_accs
from comparison_methods.RFE import plot_RFE
from comparison_methods.Haufe import plot_Haufe
from comparison_methods.ttest import plot_t_test

if __name__ == '__main__':
    DATASET_NUM = 0
    N_GLOBAL = 50
    ATLAS = 'power'
    # This script creates the interpretation plots for the four connectome-wide
    #   classification + interpretation methods. Given the current settings,
    #   this script will generate the plots Figure 4 and Table 2. To generate
    #   the plots for Figure 6, set N_GLOBAL = 250 and ATLAS = 'schaefer1000'.

    # Notably, the plot_ functions (plot_RFE, plot_CPM, etc.) all have similar
    #   structures, where a set of edges is defined, edge_weights are defined,
    #   and ROIs are scored based on the extent to which they are connected to
    #   highly predictive edges. After these variables are calculated they are
    #   submitted to the same plotting functions. This differs slightly for
    #   plot_Haufe, as that method involves scoring each edge's predictiveness
    #   as a continuous measure rather than defining a specific set of "top
    #   predictive edges". Nonetheless, its plotting remains largely the same.

    # ---- Machine Learning Methods ----
    # Create the plots for the RFE interpretation
    RFE_N_EDGES = 1000  # Target number of edges to keep
    RFE_STEP = .001  # Proportion of features dropped each RFE step
    plot_RFE(dataset_num=DATASET_NUM, N=N_GLOBAL, atlas=ATLAS,
             n_edges=RFE_N_EDGES, rfe_step=RFE_STEP)

    if ATLAS != 'schaefer1000':  # NCFS is too slow for schaefer1000 and N = 250
        # For the erylimaz approach, plot the (contaminated) accuracy for every
        #   feature threshold from 1 to 1000. This is used to identify the feature
        #   threshold (NCFS_FEATS) for interpretation. See function for more details
        plot_NCFS_accs(DATASET_NUM, N_GLOBAL, ATLAS)

        # Create the plots for the NCFS interpretation
        NCFS_FEATS = 398
        plot_NCFS(DATASET_NUM, N_GLOBAL, ATLAS, num_feats=NCFS_FEATS)

    # Create the plots for the Haufe et al. (2014) interpretation
    plot_Haufe(DATASET_NUM, N_GLOBAL, ATLAS)

    # Create the plots for CPM interpretation (similar to Shen et al., 2017)
    CPM_ALPHA_THRESH = .05  # Edges kept if p < alpha_thresh for paired t-tests
    # comparing 0-back vs. 2-back connectivity
    plot_CPM(DATASET_NUM, N_GLOBAL, ATLAS, alpha_thresh=CPM_ALPHA_THRESH)

    # ---- Frequentist methods ----
    # Creates plots for t-test results
    T_TEST_CORRECTED_ALPHA_THRESH = .05  # Alpha threshold after correction
    T_TEST_CORRECTION_METHOD = 'holm-sidak'  # Correction method
    plot_t_test(DATASET_NUM, N_GLOBAL, ATLAS,
                corrected_alpha_thresh=T_TEST_CORRECTED_ALPHA_THRESH,
                correction_method=T_TEST_CORRECTION_METHOD)

    # Create plots for Network-based statistic (Zalensky et al., 2010)
    NBS_ALPHA_THRESH = .0005  # Primary-level threshold
    MIN_CLUSTER_SIZE = 10  # Cluster-extent threshold.
    # Can be found for p < .05 via permutation-testing
    plot_NBS(DATASET_NUM, N_GLOBAL, ATLAS, alpha_thresh=NBS_ALPHA_THRESH,
             min_cluster_size=MIN_CLUSTER_SIZE)
