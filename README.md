# Influenza Meteorology Machine-Learning Analysis

This repository contains the core analysis code accompanying the manuscript:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

## Overview

This study investigated the predictive contribution of meteorological information to high-activity influenza weeks across 52 U.S. study areas from 2010 to 2026.

The analysis used an exploratory spatiotemporal machine-learning framework with temporally and geographically separated evaluation partitions.

The repository contains core code for:

- gradient-boosting algorithm benchmarking;
- CatBoost hyperparameter optimization;
- spatiotemporal recursive feature elimination;
- parallel predictor-set model evaluation;
- SHAP-based model interpretation;
- SHAP contribution decomposition;
- DeLong, net reclassification improvement (NRI), and integrated discrimination improvement (IDI) comparisons; and
- retrospective 0–4-week lead-time model comparisons.

## Repository Structure

```text
influenza-meteorology-ml/
│
├── README.md
├── requirements.txt
│
├── code/
│   ├── 01_algorithm_benchmarking.py
│   ├── 02_catboost_hyperparameter_tuning.py
│   ├── 03_spatiotemporal_rfe.py
│   ├── 04_parallel_model_evaluation.py
│   ├── 05_shap_analysis.py
│   ├── 06_shap_contribution_decomposition.py
│   ├── 07_delong_nri_idi.py
│   └── 08_lead_time_model_comparison.py
│
├── config/
│   ├── catboost_best_params.json
│   └── rfe_selected_features.csv
│
└── data/
    └── README.md
```

## Analysis Workflow

The scripts are numbered according to the main analysis workflow.

### 01. Algorithm Benchmarking

`code/01_algorithm_benchmarking.py`

Compares CatBoost, XGBoost, LightGBM, and scikit-learn HistGradientBoostingClassifier under default hyperparameters within the exploratory spatiotemporal evaluation framework.

The spatial evaluation partition contributed to algorithm selection; therefore, the resulting comparison is selection-influenced and should not be interpreted as independent evidence of algorithm superiority.

### 02. CatBoost Hyperparameter Optimization

`code/02_catboost_hyperparameter_tuning.py`

Performs CatBoost hyperparameter optimization using Optuna with 30 trials and five-fold stratified cross-validation.

The selected hyperparameters are stored in:

`config/catboost_best_params.json`

### 03. Spatiotemporal Recursive Feature Elimination

`code/03_spatiotemporal_rfe.py`

Performs recursive feature elimination using CatBoost feature importance.

At each iteration, model discrimination is evaluated in the temporal and spatial evaluation partitions. The selection objective is the mean of the temporal- and spatial-partition AUCs.

The minimum-complexity rule selects the smallest feature subset within 0.01 of the maximum joint AUC.

The selected feature set is stored in:

`config/rfe_selected_features.csv`

Because the temporal and spatial evaluation partitions contributed directly to feature selection, subsequent performance estimates from these partitions are selection-influenced.

### 04. Parallel Model Evaluation

`code/04_parallel_model_evaluation.py`

Fits and evaluates four parallel CatBoost models:

- **M1:** Full model
- **M2:** Meteorological-only model
- **M3:** Calendar-seasonality model
- **M4:** Socio-geographical model

The models are evaluated across the training, internal tuning, temporal evaluation, and spatial evaluation partitions.

### 05. SHAP Analysis

`code/05_shap_analysis.py`

Performs SHAP-based interpretation of the fitted CatBoost models using TreeExplainer and summarizes predictor-level model contributions.

SHAP values are interpreted as model-dependent attribution measures and not as causal effects or variance explained.

### 06. SHAP Contribution Decomposition

`code/06_shap_contribution_decomposition.py`

Summarizes SHAP contributions across three predictor classes:

- meteorological predictors;
- calendar-temporal predictors; and
- socio-geographical predictors.

Contribution shares are based on mean absolute SHAP magnitudes within the fitted model.

### 07. DeLong, NRI, and IDI Analyses

`code/07_delong_nri_idi.py`

Performs descriptive pairwise model comparisons using:

- DeLong tests for correlated AUCs;
- net reclassification improvement (NRI); and
- integrated discrimination improvement (IDI).

Because model and feature selection used information from the evaluation partitions, these comparisons are interpreted descriptively rather than as confirmatory statistical inference.

### 08. Lead-Time Model Comparison

`code/08_lead_time_model_comparison.py`

Performs retrospective lead-time analyses across prediction horizons from 0 to 4 weeks.

Three predictor sets are compared:

- **MM:** meteorological-only model;
- **SM:** recent-surveillance-only model; and
- **CM:** combined meteorological and recent-surveillance model.

Recent-surveillance predictors include influenza positivity rates from the preceding 1–3 weeks.

The analysis uses retrospective analytic data and does not reconstruct archived real-time data vintages, reporting delays, or later data revisions.

## Study Design

The study covers the 50 U.S. states, the District of Columbia, and Puerto Rico.

The primary study period extends from epidemiological week 40 of 2010 through week 24 of 2026.

The exploratory evaluation framework includes:

1. a training partition;
2. an internal tuning partition;
3. a temporally separated evaluation partition; and
4. a geographically separated evaluation partition.

A fixed random seed of `123` is used where specified in the analysis scripts.

The temporal and spatial evaluation partitions were excluded from final model fitting but contributed to model or feature selection at different stages of the workflow. Accordingly, reported evaluation performance should be interpreted as exploratory and selection-influenced rather than as independent external validation.

## Data Sources

The study used publicly available aggregate, area-level data from:

- CDC FluView Interactive;
- ECMWF ERA5-Land through Google Earth Engine;
- the U.S. Census Bureau; and
- the CDC FluSight Forecast Hub for the post hoc contextual analysis.

Detailed information on source datasets, study periods, variables, spatial coverage, and processing procedures is provided in the manuscript, Supplementary Information, and `data/README.md`.

No individual-level or identifiable participant data were used.

## Software

The primary analyses were conducted using Python 3.10 and R 4.5.1.

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

Python package dependencies are listed in `requirements.txt`.

## Reproducibility

The scripts are numbered according to the main machine-learning workflow:

```text
Algorithm benchmarking
        ↓
CatBoost hyperparameter optimization
        ↓
Spatiotemporal recursive feature elimination
        ↓
Parallel model evaluation
        ↓
SHAP interpretation
        ↓
SHAP contribution decomposition
        ↓
DeLong / NRI / IDI comparisons
        ↓
0–4-week lead-time model comparison
```

This repository documents the core computational workflow supporting the manuscript.

Additional sensitivity and supplementary analyses are described in the manuscript and Supplementary Information.

## Interpretation

This study is retrospective and exploratory.

The temporal and spatial evaluation partitions contributed to feature selection, and the spatial evaluation partition also contributed to algorithm selection.

Therefore, performance estimates from these partitions are selection-influenced and should not be interpreted as independent external validation.

The analysis does not establish causal meteorological effects or operational forecasting readiness. Prospective evaluation using a locked modelling pipeline would be required before operational use.

## Manuscript

This repository accompanies the manuscript:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

Authors:

Shunshun Zhang, Kailai Lu, Yirong Liu, Chaodong Long, Runmei Ma, and Tiantian Li.

## Citation

If you use code from this repository, please cite the associated manuscript:

> Zhang S, Lu K, Liu Y, Long C, Ma R, Li T.  
> *Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study.*  
> Manuscript submitted for publication.

The citation information can be updated after publication.

## Contact

Questions regarding the study, data processing, or analysis should be directed to the corresponding author identified in the manuscript.
