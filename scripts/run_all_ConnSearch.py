import pathlib
import sys

sys.path.append(f'{pathlib.Path(__file__).parent.parent}')  # to import custom modules

from group_level import run_group_Power_50_dataset, \
    run_group_Schaefer1000_250_dataset
from subject_specific import run_subject_Power_50_dataset, \
    run_subject_Schaefer1000_250_dataset

if __name__ == '__main__':
    # The ConnSearcher parameters set and run in the below functions.

    # ---- group-level analyses ----
    # Generate plots for Figure 3, and generate Table 1
    run_group_Power_50_dataset(comp_size=16)  # Table 1 and plots for Figure 3
    # Generate plots for Figure 5
    run_group_Schaefer1000_250_dataset(comp_size=16)  # plots for Figure 5

    # ---- subject-specific analyses ----
    # Generate plots for Figure 7 top and generate Table 3
    run_subject_Power_50_dataset(comp_size=16, subtract_mean=False)
    # Run the analysis after subtracting group-level effects
    run_subject_Power_50_dataset(comp_size=16, subtract_mean=True)
    # Generate plots for Figure 7 bottom
    run_subject_Schaefer1000_250_dataset(comp_size=16, subtract_mean=True)
