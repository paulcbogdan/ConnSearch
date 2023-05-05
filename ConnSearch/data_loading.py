import os.path
import pathlib
import pickle

import numpy as np
import shutil



def load_data_5D(N=50, dataset_num=0, atlas='power'):
    '''
    X is returned as a 5D array, (subjects, sessions, examples, ROI0, ROI1). For
        example, for the N = 50 Power atlas dataset, X's shape is (50, 2, 2, 
        264, 264).
    Y is returned as a 3D array, (subjects, sessions, examples).

    :param N: int, dataset sample size
    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param atlas: str, atlas of dataset
    '''
    root_dir = pathlib.Path(__file__).parent.parent
    if atlas == 'power':
        if N == 50:
            fp = fr'{root_dir}\in_data\Power\power_n50_d{dataset_num}.pkl'
            if not os.path.isfile(fp):
                shutil.unpack_archive(fp.replace('.pkl', '.zip'), os.path.dirname(fp))
                quit()
            with open(fp, 'rb') as file:
                X, Y, coords = pickle.load(file)  # X.shape = (50, 2, 2, 264, 264)
            return X, Y, coords, 'power'
        else:  # N up to 250 can be attained by combining the five N=50 Power datasets
            X_so_far = None
            Y_so_far = None
            for d in range(5):
                fp = fr'{root_dir}\in_data\Power\power_n50_d{d}.pkl'
                with open(fp, 'rb') as file:
                    X, Y, coords = pickle.load(file)
                    if X_so_far is None:
                        X_so_far, Y_so_far = X, Y
                    else:
                        X_so_far = np.concatenate((X_so_far, X), axis=0)
                        Y_so_far = np.concatenate((Y_so_far, Y), axis=0)
                if X_so_far.shape[0] >= N:
                    X = X_so_far[:N]
                    Y = Y_so_far[:N]
                    return X, Y, coords, 'power'
    elif atlas == 'schaefer1000':
        fp = fr'{root_dir}\in_data\Schaefer1000\schaefer1000_n250.pkl'
        with open(fp, 'rb') as file:
            X, Y, coords = pickle.load(file)  # X.shape = (250, 2, 2, 1000, 1000)
        return X[:N], Y[:N], coords, 'schaefer1000'
    else:
        raise ValueError(f'Invalid atlas: {atlas=}')

def load_data_2D_flat(N=50, dataset_num=0, atlas='power'):
    '''
    X is returned as a 2D array (examples, features). This contrasts how X
      is imported for ConnSearch as a 5D array (subjects, sessions, examples,
      ROI, ROI). The 2D array has flattened the first three dimensions into
      the first dimension. The ROIxROI matrix has been reduced to the second
      dimension (features) by taking the bottom left triangle. 
    The function notably also returns groups and tril_idxs, which provide the
      information lost when going a 2D array The groups variable here is a list
      which specifies which subject a given example is associated with. e.g.,
      the first example is for subject 0, so groups[0] = 0. tril_idxs is a list 
      that maps a given feature to its edge pair, e.g., the 3rd feature 
      corresponds to (0, 3) for (ROI, ROI), so tril_idxs[2] = (1, 2).
      See np.tril_indices(...) for more information.

    :param N: int, dataset sample size
    :param dataset_num: int, can be used to select among the five N=50 datasets
    :param atlas: str, atlas of dataset
    '''
    X, Y, coords, name = load_data_5D(N=N, dataset_num=dataset_num, atlas=atlas)
    groups = get_groups(X, sn_dim=0, sess_dim=1, ex_dim=2)
    tril_idxs = np.tril_indices(X.shape[4], k=-1)
    X_flat = X[:, :, :, tril_idxs[0], tril_idxs[1]]
    X_2D = X_flat.reshape((-1, X_flat.shape[-1]))
    Y_2D = Y.reshape(-1)
    return X_2D, Y_2D, groups, coords, tril_idxs


def get_groups(X, sn_dim=0, sess_dim=1, ex_dim=2):
    '''
    For the 5D X, see load_data_5D(...), the first three dimensions are 
        (subject, session, example). These get reshaped into a single dimension
        when X is returned as a 2D array. group_idxs is a list that maps the
        examples to their associated subject. e.g., the first example is from
        participant 0, so group_idxs[0] = 0.
    This function will, notably, also work for other types of X (e.g., 6D array
        where the 0th dimension correspond to a given component)
     
    :param X: 5D np array of X, where (subjects, sessions, examples, ROI0, ROI1)
    :param sn_dim: int, specify which dimension corresponds to the subject
    :param sess_dim: int, specify which dimension corresponds to the subject
    :param ex_dim: int, specify which dimension corresponds to the subject
    :return: groups_idx, list
    '''
    if sess_dim is None:
        groups = np.repeat(list(range(X.shape[sn_dim])), X.shape[ex_dim])
    else:
        groups = np.repeat(list(range(X.shape[sn_dim])),
                           X.shape[sess_dim] * X.shape[ex_dim])
    return groups
