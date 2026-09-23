# Influenza Meteorology Machine-Learning Analysis

This repository contains the core analysis code accompanying the manuscript:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

## Overview

This study investigated the predictive contribution of meteorological information to high-activity influenza weeks across 52 U.S. study areas from 2010 to 2026.

The analysis used an exploratory spatiotemporal machine-learning framework with temporally and geographically separated evaluation partitions.

The repository contains the core code used for:

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

File: `code/01_algorithm_benchmarking.py`

Compares four gradient-boosting algorithms under default hyperparameters:

- CatBoost
- XGBoost
- LightGBM
- scikit-learn HistGradientBoostingClassifier

The candidate algorithms are evaluated within the exploratory spatiotemporal partitioning framework described in the manuscript.

The spatial evaluation partition contributed to algorithm selection; therefore, the reported algorithm-comparison results should not be interpreted as independent evidence of algorithm superiority.

### 02. CatBoost Hyperparameter Optimization

File: `code/02_catboost_hyperparameter_tuning.py`

Performs CatBoost hyperparameter optimization using Optuna with 30 trials and five-fold stratified cross-validation.

The selected hyperparameters are saved in:

`config/catboost_best_params.json`

### 03. Spatiotemporal Recursive Feature Elimination

File: `code/03_spatiotemporal_rfe.py`

Performs recursive feature elimination using CatBoost feature importance.

At each iteration, model discrimination is evaluated in the temporal and spatial evaluation partitions.

The feature-selection objective is defined as the mean of the temporal- and spatial-partition AUCs.

The minimum-complexity rule selects the smallest feature subset within 0.01 of the maximum joint AUC.

The final selected feature set is stored in:

`config/rfe_selected_features.csv`

Because the temporal and spatial evaluation partitions contributed directly to feature selection, subsequent performance estimates from these partitions are selection-influenced.

### 04. Parallel Model Evaluation

File: `code/04_parallel_model_evaluation.py`

Fits and evaluates four parallel CatBoost models using different predictor sets:

- **M1:** Full model
- **M2:** Meteorological-only model
- **M3:** Calendar-seasonality model
- **M4:** Socio-geographical model

The models are evaluated across the study's training, internal tuning, temporal evaluation, and spatial evaluation partitions.

The parallel-model comparisons are descriptive comparisons between predictor sets and should not be interpreted as estimates of causal or independent incremental effects.

### 05. SHAP Analysis

File: `code/05_shap_analysis.py`

Performs SHAP-based interpretation of the fitted CatBoost models using TreeExplainer.

The analysis summarizes feature-level model contributions and produces SHAP-based importance outputs for the fitted models.

SHAP values are interpreted as model-dependent attribution measures and do not represent variance explained, independent effects, or causal effects.

### 06. SHAP Contribution Decomposition

File: `code/06_shap_contribution_decomposition.py`

Summarizes SHAP contributions according to three predictor classes:

- meteorological predictors;
- calendar-temporal predictors; and
- socio-geographical predictors.

The contribution shares are calculated from mean absolute SHAP magnitudes within the fitted model.

These values describe the distribution of prediction magnitude within the fitted model and should not be interpreted as causal contributions.

### 07. DeLong, NRI, and IDI Analyses

File: `code/07_delong_nri_idi.py`

Performs descriptive pairwise model comparisons using:

- DeLong tests for correlated receiver operating characteristic AUCs;
- net reclassification improvement (NRI); and
- integrated discrimination improvement (IDI).

The analyses compare the parallel predictor-set models within the completed exploratory workflow.

Because feature and model selection used information from the evaluation partitions, these comparisons are interpreted descriptively rather than as confirmatory statistical inference.

### 08. Lead-Time Model Comparison

File: `code/08_lead_time_model_comparison.py`

Performs the retrospective lead-time analysis across prediction horizons from 0 to 4 weeks.

Three predictor sets are compared:

- **MM:** meteorological-only model;
- **SM:** recent-surveillance-only model; and
- **CM:** combined meteorological and recent-surveillance model.

Recent-surveillance predictors include influenza positivity rates from the preceding 1–3 weeks.

The analysis uses retrospective analytic data and does not reconstruct archived real-time data vintages, reporting delays, or later data revisions.

## Study Design

The study covers the 50 U.S. states, the District of Columbia, and Puerto Rico.

The primary study period extends from epidemiological week 40 of 2010 through week 24 of 2026.

The exploratory evaluation framework includes four analytic partitions:

1. training partition;
2. internal tuning partition;
3. temporally separated evaluation partition; and
4. geographically separated evaluation partition.

A fixed random seed of `123` is used where specified in the analysis scripts.

The temporal and spatial evaluation partitions were excluded from final model fitting but contributed to model or feature selection at different stages of the workflow.

Accordingly, the reported evaluation performance is considered exploratory and selection-influenced rather than independently externally validated.

## Data Sources

The study used publicly available, aggregate, area-level data.

### CDC FluView Interactive

Weekly influenza virological surveillance data were obtained from the CDC FluView Interactive application:

**National, Regional, and State Level Outpatient Illness and Viral Surveillance**

The study used weekly surveillance information for the 50 U.S. states, the District of Columbia, and Puerto Rico.

The primary study period was:

**2010 epidemiological week 40 through 2026 epidemiological week 24**

Variables used included weekly numbers of specimens tested, influenza-positive specimens, and weekly influenza test positivity.

Access date:

**22 July 2026**

### ERA5-Land

Meteorological data were obtained from the ECMWF ERA5-Land daily aggregated reanalysis product through Google Earth Engine.

Meteorological data covered 2000–2026.

The period 2000–2009 was used to construct climatological baseline values.

The period 2010–2026 was used for the primary study analyses.

Meteorological variables included measures related to:

- air temperature;
- relative humidity;
- absolute humidity;
- surface wind speed;
- surface pressure;
- solar radiation; and
- precipitation.

Detailed meteorological variable construction and temporal-lag definitions are provided in the manuscript and Supplementary Information.

### U.S. Census Bureau

Population-density information was obtained from 2020 U.S. Census Bureau data.

The corresponding population-density variable was used as a socio-geographical predictor.

Population-density information was obtained from 2020 U.S. Census Bureau data and was used as a socio-geographical predictor. Additional source and processing details are provided in the manuscript and Supplementary Information.

### CDC FluSight Forecast Hub

Archived influenza hospitalization forecasts and corresponding finalized hospitalization target data used in the post hoc contextual analysis were obtained from the CDC FluSight Forecast Hub.

The FluSight analysis was used only as a post hoc downstream contextual analysis.

It was not used for primary model development, algorithm selection, hyperparameter optimization, or recursive feature elimination.

## Data Availability

The source datasets used in this study are publicly available from their respective data providers.

Raw copies of the complete source databases are not redistributed in this repository.

Information required to identify the public data sources and study-specific subsets is provided in:

`data/README.md`

The analysis-ready dataset was derived from the public sources described above.

No individual-level or identifiable participant data were used.

## Software

The primary analyses were conducted using:

- Python 3.10
- R 4.5.1

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

Exact Python package versions used for the analysis environment are provided in:

`requirements.txt`

## Reproducibility

The scripts are numbered according to the main analysis workflow.

Where applicable, the same random seed and partitioning rules described in the manuscript are retained in the code.

The primary sequence of the machine-learning workflow is:

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

The repository is intended to document the core computational workflow supporting the manuscript.

Some additional sensitivity and supplementary analyses are described separately in the manuscript and Supplementary Information.

## Interpretation

This study is retrospective and exploratory.

The temporal and spatial evaluation partitions contributed to feature selection, and the spatial evaluation partition also contributed to algorithm selection.

Therefore, performance estimates from these partitions are selection-influenced and should not be interpreted as independent external validation.

The analysis does not establish causal meteorological effects or operational forecasting readiness.

Prospective evaluation using a locked modelling pipeline would be required before operational use.

## Manuscript

The repository accompanies the manuscript:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

Authors:

Shunshun Zhang, Kailai Lu, Yirong Liu, Chaodong Long, Runmei Ma, and Tiantian Li.

## Citation

If you use code from this repository, please cite the associated manuscript:

> Zhang S, Lu K, Liu Y, Long C, Ma R, Li T.  
> *Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study.*  
> Manuscript submitted for publication.

The citation information should be updated after publication.

## Contact

Questions regarding the study, data processing, or analysis should be directed to the corresponding author identified in the manuscript.
