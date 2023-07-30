# ConnSearch

The present repo provides the code necessary to reproduce the primary results for the following manuscript:

Bogdan, P., Iordan, A. D., Shobrook, J., & Dolcos, F. (2023). ConnSearch: A 
    Framework for Functional Connectivity Analysis Designed for Interpretability
    and Effectiveness at Limited Sample Sizes. *NeuroImage*, 120274.

## Code overview

The ConnSearch analyses are done using the `ConnSearch.ConnSearcher` class. To run all the analyses used for the manuscript and to generate its visuals, a script has been provided, `scripts.run_all_ConnSearch.py`. It calls additional scripts in `scripts.group_level.py` and `scripts.subject_specific.py`.

## Modeling ConnSearch components

`ConnSearcher` takes connectome matrices as X and binary labels as Y. The X input is organized as a 5D array `(subject, session, example, ROI0, ROI1)`, note that "ROI" means "Region of Interest". Each subject completed two sessions and each session yielded two examples, and these aspects are represented by the first three dimensions of the 5D array. Each example's data is a symmetric connectome matrix, which is represented by the last two dimensions. For instance, for the 50-participant dataset parcellated with the Power atlas, which has 264 ROIs, `X.shape = (50, 2, 2, 264, 264)`. The Y input is organized as a 3D array, `(subject, session, example)` made up of 1s and 0s, which has the same structure as the first three dimensions of X. These data are loaded via the functions in `ConnSearch.data_loading.py`. 

`ConnSearcher` organizes X into components. Then, each component is analyzed independently. 

[//]: # (This involves creating `X_component`, which is a 6D array organized as `&#40;component_i, subject, session, example, ROI0_i, ROI1_i&#41;`. As detailed in the manuscript, one component is created per ROI. For the component size-16 analysis of the dataset above, `X_component.shape = `)

For group-level ConnSearch, a classifier is trained and tested for each component. The classifier, here, is a support vector machine (SVM) but this can be easily changed. The classifier use, as features, all the edges within a given component as features. `sklearn` is used wherever possible. Components are saved and plotted (see below) based on whether their classifier's surpass a specified accuracy threshold. By default, the scripts attempt to use an accuracy threshold established via permutation-testing. This threshold is generated using the `Permutation_Manager` class in `ConnSearch.permutation.Permutation_Manager.py`. Generally, permutation-testing is slow, and so you will want to run it beforehand, as its distributions will be cached. The scripts made available for running group-level ConnSearch will automatically load and use those cached permutation-testing distributions. To run the permutation-testing, use `scripts.run_permutation_testing.py`. To plot the permutation-testing distributions for different settings, use `scripts.plot_permutation_testing.py` (used to make Supplemental Figure S1).  

Subject-specific ConnSearch involves measuring correlational similarity between examples of the same label and subtracting the correlational similarity between examples of opposite labels. The latter analyses are largely done in `ConnSearch.fast_corrsim.py`, which uses `numpy` for fast matrix operations.

## Saving and reporting ConnSearch results

The ConnSearch results are saved in `results/`. Based on the different parameters set for the analysis (e.g., the sample size, component size, etc.), ConnSearch will yield different results. For each set of parameters, the ConnSearch outputs are saved in a unique directory. The directory name is generated automatically using `ConnSearch.names.get_save_name`. For example, the directory name may be `results/ConnSearch_Group/power_comp16_nspl5_nrep10_N50`, indicating that the analysis was group-level ConnSearch using the N = 50 Power atlas dataset, that the component size was 16, and that cross-validation used 5 folds then averaged across 10 repetitions. 

The ConnSearch results are plotted in two ways: (1) Each component is plotted individually, using the [`NiChord`](https://github.com/paulcbogdan/NiChord) package, which we developed for this research. These plots can be seen in Figure 3 and Figure 7. (2) Every component's score, its classifier accuracy or t-value, is plotted by coloring ROIs on a glass brain. These plots can be seen in Figure 5 and Figure 7. The picture below shows both the component plots (top) and the glass brain plots (bottom). In addition to using `NiChord`, several functions in `ConnSearch.reporting.plotting_ConnSearch.py` and `ConnSearch.reporting.plotting_general.py` help with plotting. The former module is used solely for the ConnSearch plotting. The latter module is also used for plotting the methods to which ConnSearch was compared, in the manuscript. Notably, the plotting functions automatically generate plots using multiple styles (e.g., to create the non-ConnSearch Figure 4 plots). 

![example_results.png](example_results.png)

Along with these visuals, the ConnSearch components are also reported as tables, using the functions in `ConnSearch.reporting.ConnSearch_tables.py`. These functions automatically assign network, regions, and Brodmann Area labels for each ROI based on its center coordinate (in MNI space). Tables 1 and 3 in the manuscript were made using these functions.


## Methods implemented for comparisons

The neural patterns implicated by ConnSearch were compared those found by six other methods. Their code is `comparison_methods/`. These other methods include four techniques where users fit a connectomewide classifier and then interpret the classifier using various post hoc techniques. These consistent of recursive feature elimination (`RFE.py`), neighborhood component feature selection (`NCFS.py` & `NCFS_feature_cutoff.py`), connectome predictive modeling (`NBS.py`), kernel ridge regression with Haufe transformation (`Haufe.py`). Two frequentist methods were also implemented: network-based statistic (`NBS.py`) and paired t-tests (`ttest.py`). All of these implementations follow a similar structure. The focus is only on the plots and tables generated via interpretation analysis and not the accuracy of classifiers.

The comparison methods can all be run using `scripts.run_comparison_methods.py`

## Additional techniques

The code also provides some features not emphasized in the manuscript but used for preliminary analyses. Two are notable.

First, the scripts allow defining components based on proximity to a Core ROI (i.e., components are groups of ROIs close together), which is useful for analyses where effects are localized but MVPA searchlights do not span enough volume. This can be done by simply setting `proximal=True` in `group_level.py` or `subject_specific.py`.

Second, the ConnSearch can be used to implement the analyses by Wu et al. (2021; Cerebral Cortex) in the context of task classification. Wu et al. (2021) describe a method where, for each ROI, a classifier is fit using all connections to that ROI as features - e.g., for a 264-ROI atlas, it fit 264 classifiers each with 263 features. The `ConnSearcher` class is capable of doing this analysis, and the functions provided in the scripts `group_level.py` and `subject_specific.py` can be used to run the analysis, by simply changing `wu_analysis=True` and `comp_size=None` (see also docstrings). Just below, we illustrate one significant result using the Wu et al. (2021) method, applied to the present dataset and 2-back vs. 0-back classification problem.

![results/Wu_Group/power_compNone_folds5_reps10_N50_acc0.633/pics/component_111_score0.644_black_combined.png](results/Wu_Group/power_compNone_folds5_reps10_N50_acc0.633/pics/component_111_score0.644_black_combined.png)

Wu, J., Eickhoff, S. B., Hoffstaedter, F., Patil, K. R., Schwender, H., Yeo, B. T., & Genon, S. (2021). A connectivity-based psychometric prediction framework for brain–behavior relationship studies. Cerebral Cortex, 31(8), 3732-3751.

## Notes

Files needed for the 50-participant analyses of the Power atlas are provided, here. The input dataset (`in_data/Power/power_n50_d0.pkl`, 106 MB) squeezes just under the 100 MB limit for GitHub if it is zipped (`in_data/Power/power_n50_d0.zip`, 98 MB). Running the scripts will automatically unzip it for you. Along with just this data, our permutation-testing results are also provided along with various other cache files to help things run quickly (e.g., dictionaries mapping ROI coordinates to brain regions).

This repo sought to use commonly-used libraries wherever possible. However, all plotting requires the [`NiChord`](https://github.com/paulcbogdan/NiChord) package, which is publicly available.

The plots used for the manuscript use a modified version of `NiChord` (e.g., to generate the colorbar titles and to adjust colors/arcs to better show differences). These visual aspects required hardcoding, and said version of `NiChord` has not been pushed, meaning that your plots may look slightly different than those in the manuscript.

We found that `sklearn.model_selection.StratifiedGroupKFold` has an issue in that it often doesn't find the most ideal stratification for each fold. This is particularly important for research likes ours where an emphasis is placed on whether accuracy beats chance. The present code fixes this in the new class `ConnSearch.crossval.ConnSearch_StratifiedGroupKFold`. In addition, the code provides `ConnSearch.crossval.RepeatedStratifiedGroupKFold`, which repeats the stratified group-k-fold and averages test accuracy across the repetitions.

The manuscript reports behavioral data on how well participants completed the N-back task. `scripts/run_WM_behavioral.py` generates these data. The script compares participant's accuracy across the five 50-participant groups used for the paper.
