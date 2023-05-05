import numpy as np


'''
Various ways of shuffling datasets during permutation-testing. 
    shuffle_X_within_subject was used for the present dataset, although other
    datasets may call for other strategies. This permutation-testing method
    was selected based on available literature and preliminary testing showing
    how it gives rise to unbiased classifiers (median and average accuracies of 
    exactly 0.5).
'''

def shuffle_X_within_subject(X, Y):
    '''
    Shuffles the examples within-subject. This function shuffles X and keeps Y
       unchanged, but this is equivalent to shuffling Y and keeping X unchanged.

    :param X: 5D numy array, shaped (subject, session, example, ROI0, ROI1)
    :param Y: list of labels. Returned unchanged, but passing Y is necessary
        for consistency with the other shufflers.
    :return:
    '''
    X_mod = X.reshape(X.shape[0], X.shape[1] * X.shape[2],
                      X.shape[3], X.shape[4]) # flatten 2 sessions x 2 examples
                                              # into just a 4 example dimension
    X_permuted = np.zeros(X_mod.shape)
    for subj_i in range(X.shape[0]): # shuffle each participant individually
        if X.shape[1] != 2:
            raise ValueError('X must have two sessions')
        idx = np.random.permutation(X_mod.shape[1]) # shuffle list of 4
        X_permuted[subj_i, :] = X_mod[subj_i, idx]
    X_permuted = X_permuted.reshape(X.shape) # reshape back to original shape
    return X_permuted, Y


def shuffle_X_within_session(X, Y):
    X_permuted = np.zeros(X.shape)
    for subj_i in range(X.shape[0]):
        if X.shape[1] != 2:
            raise ValueError('X must have two sessions')
        for sess_j in range(X.shape[1]):
            idx = np.random.permutation(X.shape[2])
            X_permuted[subj_i, sess_j, :] = X[subj_i, sess_j, idx]
    return X_permuted, Y



def shuffle_X_within_half(X, Y):
    valid_permutations = [[0, 1, 3, 2],
                          [1, 0, 2, 3],
                          [0, 2, 1, 3],
                          [3, 1, 2, 0]]
    X_mod = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4])
    X_permuted = np.zeros(X_mod.shape)
    for subj_i in range(X.shape[0]):
        if X.shape[1] != 2:
            raise ValueError('X must have two sessions')
        # idx = np.random.permutation(X_mod.shape[1])
        idx = valid_permutations[np.random.randint(0, len(valid_permutations))]
        # print(f'{idx=}')
        # quit()
        X_permuted[subj_i, :] = X_mod[subj_i, idx]
    X_permuted = X_permuted.reshape(X.shape)
    return X_permuted, Y


def shuffle_Y_fully(X, Y):
    print('Scramble_U_fully...')
    Y_mod = Y.reshape(-1)
    np.random.shuffle(Y_mod)
    Y_mod = Y_mod.reshape(Y.shape)
    return X, Y_mod


def scramble_X_fully(X, Y):
    print('Scramble_X_fully...')
    X_mod = X.reshape(-1)
    np.random.shuffle(X_mod)
    X_mod = X_mod.reshape(X.shape)
    return X_mod, Y
