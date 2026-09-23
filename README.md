Analysis workflow
The scripts are numbered according to the main analysis workflow.
01. Algorithm benchmarking
01_algorithm_benchmarking.py
Compares four gradient-boosting algorithms under default hyperparameters:
- CatBoost
- XGBoost
- LightGBM
- scikit-learn HistGradientBoostingClassifier
Performance is evaluated across the training, internal tuning, temporal evaluation, and spatial evaluation partitions.
02. CatBoost hyperparameter tuning
02_catboost_hyperparameter_tuning.py
Performs CatBoost hyperparameter optimization using Optuna with 30 trials and five-fold stratified cross-validation.
The optimized parameters are saved as:
config/catboost_best_params.json
03. Spatiotemporal recursive feature elimination
03_spatiotemporal_rfe.py
Performs recursive feature elimination using the mean AUC across the temporal and spatial evaluation partitions as the selection criterion.
The minimum-complexity rule selects the smallest feature subset within 0.01 of the maximum joint AUC.
04. Parallel model evaluation
04_parallel_model_evaluation.py
Fits and evaluates four CatBoost models:
- M1: full model
- M2: meteorological-only model
- M3: calendar-seasonality model
- M4: socio-geographical model
05. SHAP analysis
05_shap_analysis.py
Computes SHAP values using TreeExplainer and summarizes predictor importance and predictor-class contributions.
06. SHAP contribution decomposition
06_shap_contribution_decomposition.py
Summarizes SHAP contributions for the meteorological, calendar-temporal, and socio-geographical predictor classes.
07. DeLong, NRI, and IDI analyses
07_delong_nri_idi.py
Calculates descriptive pairwise model comparisons using:
- DeLong tests for correlated AUCs;
- net reclassification improvement (NRI); and
- integrated discrimination improvement (IDI).
08. Lead-time model comparison
08_lead_time_model_comparison.py
Compares:
- MM: meteorological-only model;
- SM: recent-surveillance-only model; and
- CM: combined meteorological and surveillance model
across prediction horizons of 0–4 weeks.
Data sources
The study uses publicly available aggregate data from:
1. CDC FluView Interactive:
   National, Regional, and State Level Outpatient Illness and Viral Surveillance.
2. ECMWF ERA5-Land reanalysis data accessed through Google Earth Engine.
3. U.S. Census Bureau population and population-density data.
4. CDC FluSight Forecast Hub data for the post hoc downstream analysis.
Detailed data-source definitions, temporal coverage, spatial coverage, and processing procedures are provided in the manuscript and Supplementary Information.
No individual-level or identifiable participant data were used.
Data availability
Raw source datasets are publicly available from their respective data providers.
[CHOOSE ONE OF THE FOLLOWING BEFORE PUBLIC RELEASE]
Option A:
The analysis-ready dataset used by the scripts in this repository is available in the data/ directory.
Option B:
The analysis-ready dataset is not redistributed in this repository. Instructions for obtaining and preparing the public source datasets are provided in data/README.md.
Software
The analyses were conducted using Python 3.10.
Principal Python packages include:
- pandas
- numpy
- scipy
- scikit-learn
- catboost
- xgboost
- lightgbm
- optuna
- shap
- matplotlib
Exact package versions are provided in requirements.txt.
Reproducibility
The analysis uses a fixed random seed of 123 where applicable.
The repository should be run in numerical script order.
Because the temporal and spatial evaluation partitions contributed to model and feature selection, reported performance estimates are exploratory and selection-influenced, as described in the manuscript.
Citation
If you use this code, please cite:
Zhang S, Lu K, Liu Y, Long C, Ma R, Li T.
Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study.
Manuscript submitted for publication.
Contact
For questions regarding the study, please contact the corresponding author listed in the manuscript.
