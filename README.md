# Influenza Meteorology Machine-Learning Analysis

This repository contains the core analysis code accompanying the manuscript:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

## Overview

This study examines the predictive contribution of meteorological information to high-activity influenza weeks across 52 U.S. study areas from 2010 to 2026.

The analysis uses an exploratory spatiotemporal machine-learning framework with geographically and temporally separated evaluation partitions.

The repository contains code for:

- gradient-boosting algorithm benchmarking;
- CatBoost hyperparameter optimization;
- spatiotemporal recursive feature elimination;
- parallel predictor-set model evaluation;
- SHAP-based model interpretation;
- SHAP contribution decomposition;
- DeLong, NRI, and IDI comparisons; and
- retrospective 0–4-week lead-time model comparisons.

## Repository structure

```text
code/
    01_algorithm_benchmarking.py
    02_catboost_hyperparameter_tuning.py
    03_spatiotemporal_rfe.py
    04_parallel_model_evaluation.py
    05_shap_analysis.py
    06_shap_contribution_decomposition.py
    07_delong_nri_idi.py
    08_lead_time_model_comparison.py

config/
    catboost_best_params.json
    rfe_selected_features.csv

data/
    README.md
