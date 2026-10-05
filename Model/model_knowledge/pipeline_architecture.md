# Pipeline Architecture

This document describes the end-to-end structure of the regression pipeline for predicting `remaining_useful_cycles`.

---

## High-Level Flow

```
Raw CSVs
   |
   v
Data Loading
   |-- battery_observations.csv  (172,779 rows x 31 columns)
   |-- battery_metadata.csv      (200 rows x 7 columns)
   |-- train/val/test battery ID lists
   |
   v
Data Cleaning
   |-- Drop rows where target is null  (removes 43,256 rows)
   |-- Merge metadata by battery_id
   |-- Impute pack_temp_c (median per battery, fallback to global median)
   |-- Drop leaky columns: remaining_useful_efc, eol_reached
   |-- Drop identifiers: battery_id, timestamp
   |
   v
Feature Engineering
   |-- soc_swing, charge_discharge_ratio, temp_stress
   |-- energy_per_efc, voltage_window, cycle_age_ratio, soh_degradation_rate
   |
   v
Battery-Level Train/Val/Test Split
   |-- Train:  140 batteries (~88,000 rows)
   |-- Val:     30 batteries (~19,000 rows)
   |-- Test:    30 batteries (~22,000 rows)
   |
   v
Preprocessing (fit on train only)
   |-- Numeric: SimpleImputer(median) -> RobustScaler
   |-- Categorical: SimpleImputer(most_frequent) -> OrdinalEncoder
   |-- Applied via sklearn ColumnTransformer
   |
   v
Baseline Model Evaluation (8 models, default params)
   |-- Ridge, Lasso, ElasticNet
   |-- ExtraTrees, RandomForest
   |-- XGBoost, LightGBM, CatBoost
   |-- Evaluated on validation set: MAE, RMSE, MAPE, R2
   |
   v
Hyperparameter Tuning (Optuna TPE, 60 trials each)
   |-- XGBoost study
   |-- LightGBM study
   |-- CatBoost study
   |
   v
Final Model Training (on train + val combined)
   |-- best_xgboost  (best params from study_xgb)
   |-- best_lightgbm (best params from study_lgb)
   |-- best_catboost (best params from study_cat)
   |
   v
Weighted Ensemble
   |-- Weights = inverse validation RMSE, normalized
   |-- Final prediction = weighted average of 3 model outputs
   |-- Predictions clipped to >= 0
   |
   v
Test Set Evaluation
   |-- Individual tuned models + ensemble
   |-- MAE, RMSE, MAPE, R2
   |-- Residual analysis
   |-- Error breakdown by segment (degradation status, chemistry)
   |
   v
Explainability
   |-- Native gain-based importance (XGBoost, LightGBM)
   |-- SHAP TreeExplainer on LightGBM (3,000 test samples)
   |-- Summary plot, bar chart, dependence plots
   |
   v
Artifact Export
   |-- Fitted preprocessor, trained models, ensemble weights
   |-- Evaluation metrics, model summary JSON
   |-- All plots as PNG files
```

---

## Component Details

### Preprocessor

```
ColumnTransformer
├── num: Pipeline
│   ├── SimpleImputer(strategy='median')
│   └── RobustScaler()
└── cat: Pipeline
    ├── SimpleImputer(strategy='most_frequent')
    └── OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
```

### Ensemble Prediction Function

```python
def ensemble_predict(X_proc):
    p_xgb = clip(best_xgb.predict(X_proc), 0)
    p_lgb = clip(best_lgb.predict(X_proc), 0)
    p_cat = clip(best_cat.predict(X_proc), 0)
    return w_xgb * p_xgb + w_lgb * p_lgb + w_cat * p_cat

# where w_i = (1 / RMSE_i) / sum(1 / RMSE_j)
```

### Inference on New Data

To use the saved pipeline on new observations:

```python
import joblib, json, numpy as np

preprocessor = joblib.load('model_artifacts/preprocessor.joblib')
best_xgb     = joblib.load('model_artifacts/best_xgboost.joblib')
best_lgb     = joblib.load('model_artifacts/best_lightgbm.joblib')
best_cat     = joblib.load('model_artifacts/best_catboost.joblib')

with open('model_artifacts/ensemble_weights.json') as f:
    w = json.load(f)

X_new_proc = preprocessor.transform(X_new_df[feature_cols])

p_xgb = np.clip(best_xgb.predict(X_new_proc), 0, None)
p_lgb = np.clip(best_lgb.predict(X_new_proc), 0, None)
p_cat = np.clip(best_cat.predict(X_new_proc), 0, None)

predictions = w['xgboost'] * p_xgb + w['lightgbm'] * p_lgb + w['catboost'] * p_cat
```

Note: `X_new_df` must contain all columns listed in `feature_cols` (raw features + engineered features, excluding the target and dropped columns).

---

## Design Decisions

### Why battery-level split?

Battery sensor readings within a single battery are strongly autocorrelated over time. A random row-level split would place cycles from the same battery in both train and test, causing the model to "memorize" battery-specific patterns rather than generalize to new batteries. Battery-level splitting is the only methodologically sound approach for this type of time-series data.

### Why RobustScaler instead of StandardScaler?

C-rate values and temperature readings contain physical outliers that are not errors (e.g., fast-charge events with very high C-rates). StandardScaler uses the mean and standard deviation, which are sensitive to these outliers. RobustScaler uses the median and IQR, providing stable scaling that does not compress the majority of the data toward zero.

### Why OrdinalEncoder for categoricals?

The three boosting models (XGBoost, LightGBM, CatBoost) build decision trees. Trees split on thresholds, so they can naturally handle ordinal integer codes. One-hot encoding would multiply the feature dimensionality and make some splits harder to learn (especially for high-cardinality features like manufacturer). OrdinalEncoder is both more compact and more interpretable for tree models.

### Why inverse-RMSE weighting for the ensemble?

A simple average weights all models equally regardless of quality. Inverse-RMSE weighting is a principled, parameter-free method that automatically favors the best-performing model on the validation set. A more sophisticated approach would use a stacking meta-learner, but that requires a held-out fold for fitting the meta-model.

### Why exclude remaining_useful_efc?

`remaining_useful_efc` measures remaining life in equivalent full cycle units, which is a linear monotonic transformation of `remaining_useful_cycles`. Including it would allow the model to trivially predict the target without learning any real degradation patterns.
