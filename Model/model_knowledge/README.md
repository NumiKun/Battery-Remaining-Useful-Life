# Model Knowledge Base — Battery Remaining Useful Life Regression

## Overview

This knowledge base documents the complete machine learning pipeline built to predict `remaining_useful_cycles` for lithium-ion battery packs. It is intended to serve as a reference for reproducing the work, understanding design decisions, and extending the pipeline in future iterations.

---

## Dataset

### Source Files

| File | Rows | Description |
| --- | --- | --- |
| `battery_observations.csv` | 172,779 | Per-cycle operational sensor readings for 200 batteries |
| `battery_metadata.csv` | 200 | Static battery-level attributes |
| `train_batteries.csv` | 140 IDs | Batteries assigned to training |
| `validation_batteries.csv` | 30 IDs | Batteries assigned to validation |
| `test_batteries.csv` | 30 IDs | Batteries assigned to final testing |

### Key Observations

- The split is **battery-level**, not row-level. This prevents cycles from the same physical battery leaking across train/val/test, which would inflate metrics on correlated time-series data.
- After removing rows with null `remaining_useful_cycles`, usable rows drop from 172,779 to approximately 129,523 (the 43,256 dropped rows belong to batteries that exceeded their observation window without reaching EOL).
- `pack_temp_c` has approximately 1,727 missing values (~1%), imputed per-battery using the median.

### Target Variable Statistics (Training Set)

| Statistic | Value |
| --- | --- |
| Mean | ~446 cycles |
| Median | ~391 cycles |
| Std | ~310 cycles |
| Min | 0 cycles |
| Max | 1,469 cycles |
| Skewness | ~0.4 (mild right skew) |

The mild skew did not warrant a log-transform. Tree-based models are robust to non-normal targets.

---

## Features

### Raw Features Used

All columns from `battery_observations.csv` except:

- `battery_id` (identifier, not predictive)
- `timestamp` (identifier)
- `remaining_useful_efc` (direct leakage from target)
- `eol_reached` (direct leakage from target)
- `remaining_useful_cycles` (target)

Metadata columns merged in from `battery_metadata.csv`:

- `chemistry` (LFP, NMC, NCA)
- `manufacturer`
- `climate_zone` (Temperate, Desert, Nordic, Equatorial)
- `nominal_capacity_kwh`
- `usage_strategy` (Balanced, Conservative, Aggressive, Fast-Charge Heavy)

### Engineered Features

| Feature | Formula | Rationale |
| --- | --- | --- |
| `soc_swing` | `soc_start_pct - soc_end_pct` | Effective depth of usage per cycle |
| `charge_discharge_ratio` | `charge_c_rate / (discharge_c_rate + 1e-6)` | Captures asymmetric charging stress |
| `temp_stress` | `abs(pack_temp_c - 25) * (1 + cumulative_high_temp_hours / 100)` | Weighted thermal deviation from optimal operating point |
| `energy_per_efc` | `energy_discharged_kwh / (equivalent_full_cycles + 1e-6)` | Per-cycle energy efficiency proxy |
| `voltage_window` | `pack_voltage_charge_v - pack_voltage_discharge_v` | Voltage spread narrows with rising internal resistance |
| `cycle_age_ratio` | `cycle_index / (battery_age_days + 1e-6)` | Charge frequency over lifetime |
| `soh_degradation_rate` | `state_of_health_pct / (cycle_index + 1)` | Rolling SoH degradation rate approximation |

---

## Preprocessing

A `sklearn.compose.ColumnTransformer` applies:

- **Numeric features**: `SimpleImputer(strategy='median')` followed by `RobustScaler()`. RobustScaler uses the IQR, making it insensitive to the outliers present in C-rate and temperature readings.
- **Categorical features**: `SimpleImputer(strategy='most_frequent')` followed by `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`. Tree-based models split on ordinal integer codes efficiently without one-hot expansion.

The preprocessor is fitted exclusively on the training set and applied identically to validation and test sets.

**Saved artifact**: `model_artifacts/preprocessor.joblib`

---

## Models Evaluated

### Baseline (Default Hyperparameters)

| Model | Val MAE | Val RMSE | Val R2 |
| --- | --- | --- | --- |
| Ridge | High | High | Low |
| Lasso | High | High | Low |
| ElasticNet | High | High | Low |
| ExtraTrees | — | — | — |
| RandomForest | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |
| CatBoost | — | — | — |

*(Exact values depend on run; tree-based ensembles consistently outperform linear models by a substantial margin due to the non-linear, interaction-heavy nature of battery degradation physics.)*

### Top 3 Models Selected for Tuning

1. **XGBoost** — Gradient-boosted trees with L1/L2 regularization
2. **LightGBM** — Leaf-wise gradient boosting with histogram approximation
3. **CatBoost** — Symmetric gradient boosting with ordered boosting

---

## Hyperparameter Tuning

### Framework: Optuna with TPE Sampler

- **Algorithm**: Tree-structured Parzen Estimators (TPE)
- **Objective**: Minimize validation RMSE
- **Trials per model**: 60
- **Seed**: 42 (reproducible)

### Search Spaces

**XGBoost**

- `n_estimators`: 200 to 1000
- `max_depth`: 3 to 10
- `learning_rate`: 0.01 to 0.3 (log scale)
- `subsample`: 0.5 to 1.0
- `colsample_bytree`: 0.5 to 1.0
- `reg_alpha`: 1e-4 to 10.0 (log scale)
- `reg_lambda`: 1e-4 to 10.0 (log scale)
- `min_child_weight`: 1 to 10

**LightGBM**

- `n_estimators`: 200 to 1000
- `max_depth`: 3 to 12
- `learning_rate`: 0.01 to 0.3 (log scale)
- `num_leaves`: 16 to 256
- `subsample`: 0.5 to 1.0
- `colsample_bytree`: 0.5 to 1.0
- `reg_alpha`: 1e-4 to 10.0 (log scale)
- `reg_lambda`: 1e-4 to 10.0 (log scale)
- `min_child_samples`: 5 to 100

**CatBoost**

- `iterations`: 200 to 1000
- `depth`: 4 to 10
- `learning_rate`: 0.01 to 0.3 (log scale)
- `l2_leaf_reg`: 0.01 to 10.0 (log scale)
- `subsample`: 0.5 to 1.0
- `colsample_bylevel`: 0.5 to 1.0

---

## Ensemble Strategy

A **weighted average ensemble** combines predictions from all three tuned models. Weights are derived from the inverse of each model's validation RMSE:

```
weight_i = (1 / RMSE_i) / sum(1 / RMSE_j for all j)
```

This ensures the best-performing model has the highest influence. Weights are saved to `model_artifacts/ensemble_weights.json`.

After tuning, all three models are **retrained on the combined train+validation set** before final evaluation, giving them access to more data.

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
| --- | --- | --- |
| MAE | mean( | y - y_hat | ) | Average absolute error in cycle units |
| RMSE | sqrt(mean((y - y_hat)^2)) | Error metric that penalizes large deviations more heavily |
| MAPE | mean( | y - y_hat | / y) * 100 | Relative error as a percentage |
| R2 | 1 - SS_res / SS_tot | Proportion of variance explained (1.0 = perfect) |

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to interpret the LightGBM model post-hoc:

- **Summary plot**: Shows which features contribute most across all test predictions, and in which direction.
- **Bar chart**: Ranks features by mean absolute SHAP value for a global importance view.
- **Dependence plots**: Show how the SHAP value for each top feature changes as that feature varies, exposing non-linear patterns and interactions.

A subsample of 3,000 test observations is used for SHAP computation to balance precision and runtime.

---

## Saved Artifacts

All outputs are stored in `Model/model_artifacts/`:

```
model_artifacts/
├── preprocessor.joblib               # Fitted sklearn ColumnTransformer
├── best_xgboost.joblib               # Tuned XGBoost model
├── best_lightgbm.joblib              # Tuned LightGBM model
├── best_catboost.joblib              # Tuned CatBoost model
├── ensemble_weights.json             # Inverse-RMSE weights for the ensemble
├── test_evaluation_results.csv       # Per-model metrics on test set
├── model_summary.json                # Best model summary + top SHAP features
├── target_distribution.png           # Target histogram, log transform, Q-Q plot
├── target_by_category.png            # Boxplots of target by metadata categories
├── feature_correlation.png           # Pearson correlation with target
├── degradation_trajectories.png      # SoH and RUC curves for sample batteries
├── baseline_comparison.png           # RMSE and R2 comparison across baselines
├── optuna_history.png                # Optimization convergence for all 3 studies
├── residual_analysis.png             # Residuals vs predicted, histogram, actual vs predicted
├── error_by_segment.png              # Error by degradation status and chemistry
├── feature_importance_native.png     # Gain-based importance (XGBoost + LightGBM)
├── shap_summary.png                  # SHAP beeswarm plot
├── shap_bar.png                      # Mean absolute SHAP bar chart
└── shap_dependence.png               # Dependence plots for top 4 features
```

---

## Reproducing the Results

1. Ensure all Python dependencies are installed (see requirements below).
2. Open `remaining_useful_cycles_regression.ipynb` in the `Model/` directory.
3. Run all cells in order from top to bottom.
4. Artifacts will be written to `Model/model_artifacts/` automatically.

### Required Libraries

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
joblib
```

All were verified available in the project environment at the time of development.

---

## Known Limitations and Future Work

1. **No early stopping**: Models are trained with fixed iteration counts. Adding early stopping with the validation set as the evaluation set during training could reduce overfitting and training time.
2. **Static feature engineering**: The engineered features do not use rolling window statistics (e.g., 10-cycle rolling mean of SoH). Temporal features could capture trends not visible in a single cycle snapshot.
3. **No cross-validation on batteries**: A full K-fold cross-validation at the battery level would give a more stable estimate of generalization error, at the cost of significantly higher training time.
4. **Target nulls excluded**: The 43,256 rows with null `remaining_useful_cycles` are discarded. A semi-supervised or survival analysis approach could extract information from these censored observations.
5. **Single-step prediction**: The current model predicts RUC for an individual cycle observation. A sequence model (LSTM, Transformer) trained on full discharge histories could leverage temporal dependencies.
