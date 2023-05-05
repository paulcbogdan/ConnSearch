import os
import pickle
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, \
    RepeatedStratifiedGroupKFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ConnSearch.apply_sklearn_monkeypatch import do_monkey_patch
from ConnSearch.data_loading import get_groups
from ConnSearch.fast_corrsim import corrsim_all_components
from ConnSearch.utils import print_list_stats, get_t_graph

do_monkey_patch()  # fixes issue in sklearn.model_selection.RepeatedStratifiedGroupKFold

np.random.seed(0)

class ConnSearcher:
    def __init__(self, X, Y, coords, components, dir_out, n_splits=None,
                 n_repeats=None, clf=None, wu_analysis=False):
        '''
        This class is used for both group-level and subject-specific ConnSearch.
            The two analytic branches use the same methods to organize the data
            into components and for plotting. However, the two use different
            methods for fitting the models.
        This class can also be used for the Wu et al. (2021; Cerebral Cortex)
            analysis, which can likewise be done in a group-level or subject-
            specific manner. Running the Wu technique uses the same modeling
            methods as for ConnSearch but different methods for organizing the
            data into "components" and for plotting. However, this Wu technique
            was not featured in the final manuscript.
        Future work may refactor this class into a Super class and use
            inheritance for group/subject and ConnSearch/Wu analyses.

        X is a 5D matrix that contains every participant's connectome from
            every session and every example within that session. It contains
            data from all conditions.
        X is organized as: (subject, session, example, ROI0, ROI1)
            The dimensions of the final two axes are the same size, and
            correspond to total number of ROIs in the connectome (e.g., there
            are 264 ROIs for the Power et al. atlas). For the 50-participant
            datasets used in the paper, X.shape = (50, 2, 2, 264, 264).
        Y provides labels (0, 1, etc.) for examples and has a similar structure
            Y is organized as: (subject, session, example). For the
            50-participant datasets used in the paper, Y.shape = (50, 2, 2).

        The components parameter is a numpy array or list of lists of size:
            (n_components, n_ROIs), where n_components corresponds to the number
            of components, and n_ROIs corresponds to the number of ROIs in each
            (i.e., the components' size). Currently, all components must be the
            same size. This restriction helps with optimization as it permits
            efficient use of matrix multiplication via numpy.

        In __init__, the 5D X matrix will be organized into components, creating
            self.X_components, which is a 6D matrix. It is organized as:
                (component, subject, session, example, n_ROI0, n_ROI1).
            Each component is a small matrix of size n_ROIs x n_ROIs.

        The other parameters are relevant to saving, plotting, or defining
            the classifier to be fit separately for each component.

        :param X: 5D np array (subject, session, example, ROI0, ROI1)
        :param Y: 3D np array (subject, session, example)
        :param coords: coordinates for each ROI. Used for plotting
        :param components: component definitions applied to the connectomes
        :param dir_out: where to save results
        :param n_splits: parameter used for RepeatedStratifiedGroupKFold
        :param n_repeats: parameter used for RepeatedStratifiedGroupKFold
        :param clf: sklearn-style classifier fit for each component
        :param wu_analysis: If true, the analysis will be performed as in
            Wu et al. (2021). The "Wu components" can either use edges with all
            ROIs (is so the components paremeter should be set to None) or can
            be trimmed to just the edges with ROIs specified by the components
            parameter
        '''
        self.X = X  # (subject, session, example, ROI0, ROI1)
        self.Y = Y
        self.coords = coords
        self.components = components
        if components is None:  # Defines "components" to mirror the Wu et al.
            # (2021) analysis, which fit classifiers using
            # all edges to a given ROI.
            self.components = []
            for i in range(X.shape[3]):
                self.components.append([i] +
                                       [j for j in range(X.shape[3]) if j != i])
            self.components = np.array(self.components)
        self.comp_size = len(self.components[0])
        self.dir_out = dir_out
        self.wu_analysis = wu_analysis
        self.X_components_flat = None
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.clf = clf
        self.sort_X_0_to_1()

        if wu_analysis:
            self.save_component = self.save_component_Wu
            self.organize_X_into_Wu()
        else:
            self.save_component = self.save_component_ConnSearch
            self.organize_X_into_components()

    def sort_X_0_to_1(self):
        '''
        Examples are organized within-subject such that those with the
            lowest label (Y) are first, and those with the highest label last
        :return:
        '''
        for subject_i in range(self.X.shape[0]):
            for sess_j in range(self.X.shape[1]):
                idx = np.argsort(self.Y[subject_i, sess_j, :])
                self.X[subject_i, sess_j, :] = self.X[subject_i, sess_j, idx]

    def organize_X_into_components(self):
        '''
        Organizes self.X into components, creating self.X_components
        :return:
        '''
        component_size = len(self.components[0])
        X_components = np.empty((len(self.components), self.X.shape[0],
                                 self.X.shape[1], self.X.shape[2],
                                 component_size, component_size))
        for i, component in enumerate(self.components):
            X_components[i] = \
                self.X[:, :, :, component, :][:, :, :, :, component]
        if np.isnan(X_components).any():
            raise ValueError("X_components contains NaN")
        self.X_components = X_components

    def organize_X_into_Wu(self):
        '''
        This method was used for analyses that did not make it into the final
            paper.
        The method organizes self.X into components based on the approach by
            Wu et al. (2021). The authors did not use the word "components",
            although their analysis involved identifying ROIs i and fitting
            classifiers based on connections between ROI i with every other ROI
            in the connectome.
        This method can do this and organizes the "Wu components" into
            self.X_organized as for the main ConnSearch analysis of the previous
            method. However, here, self.X_components is a 5D array, organized
            as: (component, subject, session, example, feature)
        This method can also go beyond just what Wu et al. (2021) did. It can
            also "trim" the edges so that "Wu components" are defined as the
            edges between ROI i and the n other ROIs specified in
            self.components. For example, if self.components specifies 16 ROIs,
            the "trimmed Wu components" will contain 15 edges. The shape would
            be: (component, subject, session, example, 15).
        '''
        print('Organizing into Wu...')
        self.X_components_flat = np.empty((self.X.shape[3], self.X.shape[0],
                                           self.X.shape[1], self.X.shape[2],
                                           len(self.components[0])))
        for i, component in enumerate(self.components):
            self.X_components_flat[i] = self.X[:, :, :, component[0], component]
        self.X_components = self.X_components_flat

    def do_group_level_analysis(self, acc_thresh=None, acc2p=None, alpha=0.05,
                                verbose=True):
        '''
        Does the group-level analyses. Notably, as each participant contributes
            multiple examples, it is important to use group k-fold to ensure
            that a given participant's examples are either entirely within the
            training set or entirely within the test set for each fold.
            Additionally, as we detail in the paper, using repeated k-fold permits
            greater sensitivity in the analyses. Further, we must stratify the
            data (equal proportions of Y in the training and testing sets).
            This is accomplished via sklearn's RepeatedStratifiedGroupKFold
        To determine what the group-level accuracy threshold is needed for
            saving a component result, you can either pass an accuracy threshold
            (acc_thresh) or pass acc2p and an alpha threshold. acc2p is a
            method, mapping accuracies to p-values. It can be created using
            permutation-testing (the Permutation_Manager class).
        alpha is the p-value threshold used to determine which components are
            significant. Significant components are saved. If alpha is set to 1,
            then all components are saved.

        If you pass neither acc_thresh nor acc2p, then no results will be saved
            and this method will simply return the accuracies associated with
            each component.

        :param acc_thresh: accuracy threshold to determine what components will
            be saved
        :param acc2p: method mapping accuracies to p-values. It can be created
            using the Permutation_Manager class
        :param alpha: p-value threshold to determine what components will be
            saved, if acc2p is passed
        :return: accuracies associated with each component classifier
        '''

        if acc_thresh is not None and acc2p is not None:
            warnings.warn('You provided both acc_thresh and acc2p. acc2p will '
                          'be used and acc_thresh will be ignored.')

        # X_components is a 6D array, (component, subj, sess, ex, ROI0, ROI1)
        groups = get_groups(self.X_components,  # List specifying which subject
                            sn_dim=1, sess_dim=2, ex_dim=3)  # each example belongs to
        if self.X_components_flat is None:
            self.flatten_X_components()  # The last two dimensions (ROI0, ROI1)
            # specify the component matrices. We flatten the bottom
            # triangle of each matrix into a vector.

        X_reshaped = self.X_components_flat.reshape(  # Three dims (subject,
            self.X_components_flat.shape[0], -1,  # session, example) are
            self.X_components_flat.shape[-1])  # flattened into one dim (example)
        # X_reshaped is now a 3D array,
        # (component, example, features)
        Y_reshaped = self.Y.reshape(-1)  # Flatten 3 dims into 1 for Y too
        cv = RepeatedStratifiedGroupKFold(n_splits=self.n_splits,
                                          n_repeats=self.n_repeats,
                                          random_state=0)
        scores = []  # List of accuracies for each component
        self.t_graph = get_t_graph(self.X, self.Y)  # Only used for plotting.
        # The figures plot each edge
        #   as a t-value.
        if verbose:  # Prints a progress bar to the console
            iterator = tqdm(enumerate(X_reshaped),
                            desc='Fitting component classifiers')
        else:
            iterator = enumerate(X_reshaped)
        for i, X_component in iterator:
            # Looping over each component
            # X_component is a 2D array (example, features), which can be used
            #   as input to a classifier.
            acc = cross_val_score(clone(self.clf), X_component, Y_reshaped,
                                  cv=cv, groups=groups)
            acc = np.mean(acc)
            comp_matrix = \
                self.t_graph[self.components[i], :][:, self.components[i]]  # plot
            if acc2p is not None:
                p = acc2p(acc)  # Map accuracy to p-value
                if p < alpha:  # If p-value is significant, save the component
                    self.save_component(i, acc, comp_matrix=comp_matrix)
            elif acc_thresh is not None:
                if acc > acc_thresh:
                    self.save_component(i, acc, comp_matrix=comp_matrix)
            scores.append(acc)
        scores = sorted(scores)
        print('-***- Classifier accuracies -***-')
        print_list_stats(scores)
        return scores

    def flatten_X_components(self):
        '''
        This method flattens the component matrices for self.X_components.
            For example, if component size = 16, then self.X_components last
            two axes specify 16x16 matrices. The bottom triangle (no diagonal)
            of the matrices are flattened into a vector of length 120.
            Flattening is necessary for both the group-level and subject-
            specific analyses.
        The new flattened data are stored in self.X_components_flat, which
            has one fewer axis than self.X_components.
        self.X_components_flat is a 5D numpy array with axes:
            (component, subject, session, trial, flattened component edges)
        :return:
        '''
        component_size = self.X_components.shape[4]
        tril_idxs = np.tril_indices(component_size, k=-1)
        self.X_components_flat = self.X_components[:, :, :, :,
                                 tril_idxs[0], tril_idxs[1]]

    def do_subject_specific_analysis(self, alpha, FWE=True,
                                     subtract_mean=False,
                                     multiple_tests_method=None):
        '''
        Performs subject-specific ConnSearch analysis based on the data (X & Y)
            provided in __init__. This analysis uses correlational similarity.
        The bulk of the code is done using the methods in
            fast_corrsim.py. Said code leverages np arrays and
            does every calculation in a vectorized manner. This allows the code
            to run quickly.

        :param alpha: required alpha level (corrected for FWE or FDR)
        :param FWE: Executes common multiple-hypothesis correction techniques.
            For FWE=True, uses Holm-Sidak correction. For FWE=False, uses FDR
            correction (Benjamini-Hochberg). These are two defaults, implemented
            in statsmodels.stats.multitest import multipletests. However,
            you can alternatively pass a string via the multiple_tests_method
            parameter, and it will just use that instead
        :param multiple_tests_method: String to override the default correction
            procedures for FWE=True or False. For example, specifying
            multiple_tests_method='simes-hochberg' will use the Simes-Hochberg
            correction. See multipletests from the statsmodels package
            for other options.
        :param subtract_mean: If True, subtracts group-level effects from each
            subject before calculating correlational similarity.
        :return:
        '''
        # Note:
        #   X_components is a 6D array, (component, subj, sess, ex, ROI0, ROI1)
        #   Y is a 3D array, (subj, sess, ex)
        ts, ps, euclidean_adj_matrix = \
            corrsim_all_components(self.X_components, self.Y,
                                   subtract_mean=subtract_mean)
        # Euclidean_adj_matrix is used for visualization only

        if multiple_tests_method is None:
            if FWE:  # Using multiple-hypothesis correction via statsmodels
                multiple_tests_method = 'holm-sidak'
            else:
                multiple_tests_method = 'fdr_bh'
        print(f'{multiple_tests_method} correction applied')
        bools, ps_FDR, alpha_sidak, alpha_bon = multipletests(ps,
                                                              method=multiple_tests_method,
                                                              alpha=alpha)
        print(f'\tNumber of significant components: {sum(bools)}')
        for i, p in enumerate(ps_FDR):
            if p < alpha or alpha >= 1:  # Save significant components
                self.save_component(i, ts[i],  # If alpha >= 1, save all
                                    comp_matrix=euclidean_adj_matrix[i],
                                    code_xtra='_SubjSpec')

    def save_component_ConnSearch(self, component_i, score, dir_out=None,
                                  comp_matrix=None, code_xtra=''):
        '''
        Each component is saved to its own .pkl. Data are saved to self.dir_out
            by default, but you can specify a different directory using dir_out.
            Each .pkl contains all the information needed for plotting the
            component results.

        This method is used for both group-level and subject-specific
            ConnSearch.
        :param component_i: int, index of component
        :param score: float, accuracy (group-level) or t-val (subject-specific)
        :param dir_out: str, directory to save component .pkl
        :param comp_matrix: 2D numpy array, of the edge values to be plotted
        :param code_xtra: str, extra code, which may be useful later for
            identifying the analysis used to generate the component result.
        '''

        if dir_out is None:
            dir_out = self.dir_out
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        fn_out = os.path.join(dir_out, f'component_{component_i}_score{score:.3f}.pkl')
        component = self.components[component_i]
        coords_component = self.coords[component]
        i_to_node_idx = dict((i, node_idx) for i, node_idx in enumerate(component))
        edges = zip(*np.tril_indices(len(component), k=-1))
        edges = [(i_to_node_idx[i], i_to_node_idx[j]) for (i, j) in edges]
        pkg_out = {
            'score': score,  # can be accuracy (group) or a t-value (subject)
            'coords_component': coords_component,  # coordinates for ROIs in component
            'coords_all': self.coords,  # coordinates for all ROIs in connectome
            'adj': comp_matrix,  # values assigned to edges for plotting
            'component': component,  # indices (ROI_i) of ROIs in component.
            # component[0] is the Core ROI's index
            'edges': edges,  # list of pairs of indices, [(ROI_i, ROI_j), ...]
            'edge_weights': None,  # determines colors of edges in plot
            'code': 'connsearch' + code_xtra  # identifier used for plotting
        }

        with open(fn_out, 'wb') as f:
            pickle.dump(pkg_out, f)

    def save_component_Wu(self, component_i, score, dir_out=None,
                          comp_matrix=None,
                          code_xtra=''):
        '''
        Used for plotting the results of the Wu et al. analysis. This
            wasn't used for the final manuscript.

        :param component_i: int, index of component
        :param score: float, accuracy (group-level) or t-val (subject-specific)
        :param dir_out: str, directory to save component .pkl
        :param comp_matrix: unused, here for consistency w/ self.save_component
        :param code_xtra: str, extra code, which may be useful later for
            identifying the analysis used to generate the component result.
        :return:
        '''
        if dir_out is None:
            dir_out = self.dir_out
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        fn_out = os.path.join(dir_out,
                              f'component_{component_i}_score{score:.3f}.pkl')
        component = self.components[component_i]
        edges = [(j, component_i) for j in component]
        coords_component = self.coords[component]
        if len(comp_matrix.shape) == 1:
            t_component = edge_weights = comp_matrix
            edge_weights[0] = 0
        else:
            t_component = np.empty(self.t_graph[component, :][:, component].shape)
            t_component[:] = np.nan
            t_component[0, :] = self.t_graph[component_i, component]
            edge_weights = self.t_graph[component_i, component]
            edge_weights[0] = 0

        pkg_out = {'score': score, 'coords_component': coords_component,
                   'adj': t_component,
                   'coords_all': self.coords, 'component': component,
                   'edges': edges, 'edge_weights': edge_weights,
                   'code': 'wu' + code_xtra
                   }

        with open(fn_out, 'wb') as f:
            pickle.dump(pkg_out, f)
