import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import custom modules

from ConnSearch.permutation.permutation_manager import Permutation_Manager
from ConnSearch.data_loading import load_data_5D
from ConnSearch.components import get_connectivity_sets

from sklearn.svm import SVC

'''
This code runs Permutation_Manager. It is used to perform permutation-testing.
    permutation-testing is meant to be done before searching for significant
    components using group-level ConnSearch. The results of permutation 
    are cached, then retrieved by ConnSearch (see group_level.py). 
The permutation-testing results can be plotted (see run_permutation_plotting.py)  
'''

if __name__ == '__main__':
    ATLAS = 'power'
    N = 50
    X_GLOBAL, Y_GLOBAL, COORDS, NAME = load_data_5D(atlas=ATLAS, N=N)
    N_PERM = 1000  # Number of shuffled datasets to analyze
    N_SPLITS = 5  # Number of splits for cross-validation
    N_REPEATS = 10  # Number of repeats for cross-validation
    COMP_SIZE = 16 # Component size
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    CACHE_DIR = f'{ROOT_DIR}/ConnSearch/permutation/permutation_saves'
    PERMUTATION_SCHEME = 'within_subject'  # Specifies how data will be shuffled
    COMPONENT_FUNC = get_connectivity_sets  # Use connectivity sets as components

    PM = Permutation_Manager(X_GLOBAL, Y_GLOBAL, COORDS, SVC(kernel='linear'),
                             comp_size=COMP_SIZE,
                             n_folds=N_SPLITS,
                             component_func=COMPONENT_FUNC,
                             n_repeats=N_REPEATS,
                             n_perm=N_PERM,
                             perm_strategy=PERMUTATION_SCHEME,
                             cache_dir=CACHE_DIR)
    PM.run_permutations()
