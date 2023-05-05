from copy import deepcopy

from scipy.spatial import distance
import numpy as np

def get_components(X=None, comp_size=10, coords=None,
                   proximal=False, no_components=False):
    '''
    Calls the component function below. Usually returns 2D numpy arrays, shaped
        (#ROIs, comp_size). Each row i contains the indices for the component
        built using Core ROI_i.

    :param X: 5D numpy array representing the dataset, shape =
        (subjects, sessions, examples, ROI0, ROI1).
    :param comp_size: int, size of components
    :param coords: list of 3-element tuples, representing each ROI's coordinate.
                   Only used if proximal=True.
    :param proximal: bool, if True, defines components based on proximity
        to a Core ROI. If False, defines components based on connectivity.
        The manuscript only uses components based on connectivity.
    :param no_components: bool, if True, returns None.
    :return:
    '''
    if no_components or comp_size is None:
        return None
    elif proximal:
        return get_proximal_sets(X=X, comp_size=comp_size, coords=coords)
    else:
        return get_connectivity_sets(X=X, comp_size=comp_size, coords=coords)


def get_connectivity_sets(X=None, comp_size=10, coords=None):
    '''
    This function defines components based on connectivity, as reported in the
        manuscript.

    :param X: 5D numpy array representing the dataset, shape =
              (subjects, sessions, examples, ROI0, ROI1).
    :param comp_size: int, size of components
    :param coords: required for consistency with other the component function
                   below. Not used.
    :return: 2D numpy array, shaped (#ROIs, comp_size). Each row i contains the
             indices for the component built using Core ROI_i.
    '''
    sets = []
    avg_graph = X.mean(axis=(0, 1, 2))
    for i in range(avg_graph.shape[0]):
        i_connectivities = [(i, 1)] # a node's "connectivity" with itself is set to 1
        for j in range(avg_graph.shape[0]):
            avg = avg_graph[i][j] # taking the abs(...) has virtually no impact
            i_connectivities.append((j, avg))
        i_connectivities.sort(key=lambda x:x[1])
        i_connectivities.reverse()
        set_i = [j[0] for j in i_connectivities[0:comp_size]]
        sets.append(set_i)
    sets = np.array(sets)
    return sets


def get_proximal_sets(X=None, comp_size=10, coords=None):
    '''
    Not used for the manuscript. This function defines components based on
        proximity to a given Core ROI.
    Permutation_Manager requires that X, size, and coords can all be taken as
        arguments.

    :param X: required argument for consistency with function above. Not used
    :param comp_size: int, size of components
    :param coords: list of 3-element tuples, representing each ROI's coordinate
    :return: 2D numpy array, shaped (#ROIs, comp_size). Each row i contains the
        indices for the component built using Core ROI_i.
    '''
    sets = []
    for i, coord_i in enumerate(deepcopy(coords)):
        i_distances = [(i, 0)] # a node's "distance" to itself is set to 0
        for j, coord_j in enumerate(deepcopy(coords)):
            if i == j:
                continue
            i_distances.append((j, distance.euclidean(coord_i, coord_j)))
        i_distances.sort(key=lambda x:x[1])
        set_i = [j[0] for j in i_distances[0:comp_size]]
        sets.append(set_i)
    sets = np.array(sets)
    return sets



def get_none_components(X=None, comp_size=10, coords=None):
    '''
    When components are None, this causes ConnSearcher to carry out the analysis
        by Wu et al. (2021).
    This function exists because Permutation_Manager can be used for the Wu
        approach. Running Permutation_Manager requires a function to define the
        components.

    :param X: required argument for consistency with the other functions
    :param comp_size: required argument for consistency with the other functions
    :param coords: required argument for consistency with the other functions
    :return: None
    '''
    return None