import numpy as np
from scipy import stats as stats

'''
The subject specific analyses (correlational similarity) can be computed much
    faster if we analyze all components and all subjects at once using numpy
    arrays instead of for-looping over individual components and subjects.This 
    makes the code more elaborate and complicated. This functionality has been
    but in this module, as opposed to in ConnSearcher.py. 
'''


def corrsim_all_components(X_components, Y, get_similarity=False,
                           subtract_mean=True):
    '''
    Calculates correlational similarity for every component (X)
        with respect to labels Y

    X_components contains all data for all components and for both Y = 0 & Y = 1
        The Y vector is used to extract X[Y = 0] and X[Y = 1]

    After this, correlational similarity is calculated for each component

    Setting subtract_mean=True causes group-level effects (i.e., differences
        between the two conditions seen across all participants) to be regressed
        out of each subject's data. This is done on a component by component
        and edge-by-edge basis for every participant before calculation of
        correlational similarity.
        Setting subtract_mean=True ensures that any effects found via subject-
        specific ConnSearch indeed are unique to that specific subject and do
        not apply to everybody.
        The subtract_mean=True subtraction and code are somewhat obtuse. To
        regress out group-level effects on subject i, we must also omit
        subject i's contribution to the group-level effects. Doing this in an
        unbiased way requires the code below. For detFails, please see
        Supplemental Materials 1.4 of the paper.

    After calculating correlational similarity on a subject-by-subject and
        component-by-component basis, a t-test is performed across-subjects to
        assess whether the correlational similarity effect is significant.
        The results of the t-tests (t-values and p-values) are returned.

    :param X_components:
    :param Y:
    :param get_similarity:
    :param subtract_mean:
    :return:
    '''

    component_size = X_components.shape[4]
    tril_idxs = np.tril_indices(component_size, k=-1)

    if len(X_components.shape) == 6:
        X0_flat = extract_cond_X(0, X_components, Y)[:, :, :, :,
                  tril_idxs[0], tril_idxs[1]]
        X1_flat = extract_cond_X(1, X_components, Y)[:, :, :, :,
                  tril_idxs[0], tril_idxs[1]]
    else:
        X0_flat = extract_cond_X(0,
                                 np.expand_dims(X_components, 5), Y).squeeze(axis=-1)
        X1_flat = extract_cond_X(1,
                                 np.expand_dims(X_components, 5), Y).squeeze(axis=-1)

    if subtract_mean:  # Subtract group-level effects following the equations
        # Provided in Supplemental Materials 1.3.
        X0_flat_M = X0_flat.mean(axis=(1, 2))
        X1_flat_M = X1_flat.mean(axis=(1, 2))
        X0_flat_subj_M = X0_flat.mean(axis=2)[:, :, np.newaxis]
        X1_flat_subj_M = X1_flat.mean(axis=2)[:, :, np.newaxis]

        X0_flat_M_prime = (X0_flat_M[:, np.newaxis, np.newaxis, :] -
                           X0_flat_subj_M / (X_components.shape[1] * 2)) * \
                          X_components.shape[1] * 2 / \
                          (X_components.shape[1] * 2 - 1)
        X1_flat_M_prime = (X1_flat_M[:, np.newaxis, np.newaxis, :] -
                           X1_flat_subj_M / (X_components.shape[1] * 2)) * \
                          X_components.shape[1] * 2 / \
                          (X_components.shape[1] * 2 - 1)

        X0_flat = X0_flat - X0_flat_M_prime
        X1_flat = X1_flat - X1_flat_M_prime

    euclidean_adj_matrix = get_euclidean_edge_distances(X0_flat, X1_flat,
                                                        X_components)
    corr_same_00_10 = corr_all(X0_flat[:, :, 0], X0_flat[:, :, 1], axis=3)
    corr_same_10_11 = corr_all(X1_flat[:, :, 0], X1_flat[:, :, 1], axis=3)
    corr_dif_00_11 = corr_all(X0_flat[:, :, 0], X1_flat[:, :, 1], axis=3)
    corr_dif_10_01 = corr_all(X1_flat[:, :, 0], X0_flat[:, :, 1], axis=3)

    similarity = corr_same_00_10 + corr_same_10_11 - \
                 corr_dif_00_11 - corr_dif_10_01
    similarity = np.squeeze(similarity)
    if get_similarity:
        return similarity, euclidean_adj_matrix

    M_similarity = np.mean(similarity, axis=1)
    SE_similarity = np.std(similarity, axis=1, ddof=1) / \
                    np.sqrt(similarity.shape[1])
    ts = M_similarity / SE_similarity
    ps = stats.t.sf(ts, similarity.shape[1] - 1)
    return ts, ps, euclidean_adj_matrix


def extract_cond_X(cond, X_components, Y):
    '''
    Extracts examples from X_components for a given condition (Y = cond)

    for X_components:
        axis=0 is component, axis=1 is subject, axis=2 is session,
        axis=3 is example, axes=4 & 5 are ROIs.
        See elaboration in the documentation for class ConnSearcher
    for Y: axis=0 is subject, axis=1 is session, axis=2 is example

    Assumes that every session only has a single example of Y = cond

    :param cond: value of Y to extract
    :param X_components: 6D numpy array, (component, subject, session, example,
                                          ROI, ROI)
    :param Y: 3D numpy array, (subject, session, example)
    :return: 6D numpy, with the same structure as X_components
    '''
    Y_bool = Y == cond
    # The code below basically takes Y_bool and repeats it across the three
    #   axes of X_components, which Y does not have
    # Y_bool_repeated is then used to extract X_components_cond
    Y_bool_repeated = np.repeat(np.repeat(np.repeat(
        Y_bool[np.newaxis, :, :, :], X_components.shape[0], axis=0)
                                          [:, :, :, :, np.newaxis], X_components.shape[4], axis=4)
                                [:, :, :, :, :, np.newaxis], X_components.shape[5], axis=5)
    X_components_cond = np.where(Y_bool_repeated, X_components, np.nan)
    X_components_cond = np.nanmean(X_components_cond, axis=3, keepdims=True)
    # This code may seem a bit strange.
    #   However, I'm not sure whether there is a better way to do this. I can't
    #   find any dedicated numpy function for this nor a more elegant approach
    #   on searching StackOverflow.
    return X_components_cond


def corr_all(ar0, ar1, axis=1):
    '''
    For all i: correlates column i in 2D matrix ar0 x column i in 2D matrix ar1.
        Then, Fisher z-transforms the correlations

    This is equivalent to:
        corrXY = np.empty(ar.shape[1])
        for i in range(ar0.shape[1]):
            r, _ = stats.pearsonr(ar0[:, i], ar1[:, i])
            corrXY[i] = np.arctanh(r)
        return corrXY

    However, the code below is much faster than that above, as it calculates
        all the correlations at once via numpy arrays.

    Can be switched to correlating rows instead of columns by setting axis=0

    :param ar0: 2D array
    :param ar1: 2D array of same shape as ar0
    :param axis: axis to correlate along
    :return: 1D array of Fisher z-transformed correlations
    '''
    eXY = np.mean(ar0 * ar1, axis=axis)
    eXeY = ar0.mean(axis=axis) * ar1.mean(axis=axis)
    covXY = eXY - eXeY
    sdX = np.std(ar0, axis=axis)
    sdY = np.std(ar1, axis=axis)
    corrXY = covXY / (sdX * sdY)
    corrXY = np.arctanh(corrXY)  # Fisher z-transform
    return corrXY


def get_euclidean_edge_distances(X0_flat, X1_flat, X_components):
    '''
    Calculates Euclidean edge distances between X0 and X1. Takes in data after
        the connectivity matrices have been flattened into vectors of features.
    It then converts the difference vectors back into matricies corresponding
        to the connectivity edges.
    These euclidean edge distances are used for plotting.

    :param X0_flat: 5D array, (component, subject, session, example, features)
    :param X1_flat: 5D array, (component, subject, session, example, features)
    :param X_components: Its shape is used to rebuild the matrices
    :return: 3D array, (component, ROI0, ROI1)
    '''

    euclidean = abs(X0_flat[:, :, 0] - X0_flat[:, :, 1]) + \
                abs(X1_flat[:, :, 0] - X1_flat[:, :, 1]) - \
                abs(X0_flat[:, :, 0] - X1_flat[:, :, 1]) - \
                abs(X1_flat[:, :, 0] - X0_flat[:, :, 1])

    euclidean = -euclidean
    if len(X_components.shape) == 5:  # For the Wu et al. (2021) analysis
        euclidean = np.mean(euclidean, axis=1)
        max_abs = np.max(abs(euclidean), axis=2)
        euclidean = np.squeeze(euclidean)
        max_abs = np.squeeze(max_abs)
        euclidean = euclidean.T / max_abs.T
        euclidean = euclidean.T
        return euclidean

    # Rebuild connectivity matrices
    euclidean_adj_matrix = np.zeros((euclidean.shape[0], euclidean.shape[1],
                                     X_components.shape[-2],
                                     X_components.shape[-1]))
    for component_i in range(euclidean.shape[0]):
        for subj_j in range(euclidean.shape[1]):
            euclidean_adj_matrix[component_i, subj_j] = \
                vector2matrix(euclidean[component_i, subj_j, 0, :])
    euclidean_adj_matrix = np.mean(euclidean_adj_matrix, axis=1)
    max_abs = np.max(abs(euclidean_adj_matrix), axis=(1, 2))
    euclidean_adj_matrix = euclidean_adj_matrix.T / max_abs.T  # Norm to [-1, 1]
    euclidean_adj_matrix = euclidean_adj_matrix.T
    return euclidean_adj_matrix


def vector2matrix(a):
    '''
    Takes in a vector, e.g., shape = (120, ), then creates a matrix, e.g.,
        shape = (16, 16). Indices of the vector map to edges of matrix following
        the numpy procedures (see np.tri). The previous functions likewise
        used numpy to convert matrices to vectors (see np.tril_indices).
    :param a: 1D numpy array
    :return: 2D numpy array
    '''
    n = int(np.sqrt(len(a) * 2)) + 1
    mask = np.tri(n, dtype=bool, k=-1)  # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n, n), dtype=float)
    out[mask] = a
    out = out + out.T - np.diag(out.diagonal())  # copy bottom triangle to top
    return out
