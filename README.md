# Battery Remaining Useful Life (RUL) Prediction

## Industrial Machine Learning Pipeline for Remaining Useful Cycle Estimation

An end-to-end, production-oriented machine learning regression pipeline designed to predict the remaining useful life (`remaining_useful_cycles`) of lithium-ion battery packs from multi-sensor operational telemetry and static pack metadata.

---

## Executive Summary

Accurate estimation of Remaining Useful Life (RUL) in lithium-ion battery systems is critical for electric vehicle (EV) fleet operations, stationary battery energy storage systems (BESS), and predictive maintenance scheduling. Degradation mechanisms in battery cells—including solid electrolyte interphase (SEI) growth, lithium plating, and active material loss—exhibit severe non-linear behavior influenced by thermal gradients, cycling depth, and operational stress.

This project implements an end-to-end regression pipeline developed using industry best practices:
- **Leakage-Free Validation**: Strict battery-level partitioning (140 train / 30 validation / 30 test) ensuring no cycle observations from the same physical battery cross validation boundaries.
- **Physics-Informed Feature Engineering**: Domain-driven feature construction targeting electro-thermal stress, cycling intensity, and degradation rates.
- **Systematic Benchmarking**: Comparative analysis of 8 distinct regression algorithms spanning regularized linear models, tree bagging, and gradient-boosted decision trees.
- **Bayesian Hyperparameter Optimization**: Automated tuning using Optuna with Tree-structured Parzen Estimator (TPE) across XGBoost, LightGBM, and CatBoost.
- **Weighted Ensemble Architecture**: Multi-model blending via inverse validation RMSE weights.
- **Model Explainability**: Post-hoc global and local interpretability using TreeSHAP (SHapley Additive exPlanations).

---

## Repository Structure

```
.
|-- Dataset/
|   |-- battery_observations.csv       # Multi-cycle operational telemetry (172,779 rows x 31 cols)
|   |-- battery_metadata.csv           # Static battery pack specifications (200 rows x 7 cols)
|   |-- train_batteries.csv            # Training battery identifier index (140 batteries)
|   |-- validation_batteries.csv       # Validation battery identifier index (30 batteries)
|   `-- test_batteries.csv             # Final test battery identifier index (30 batteries)
|-- Model/
|   |-- remaining_useful_cycles_regression.ipynb   # Complete 15-section executable notebook
|   |-- model_knowledge/
|   |   |-- README.md                  # Comprehensive pipeline knowledge base
|   |   |-- feature_reference.md       # Technical feature catalog and domain rationale
|   |   `-- pipeline_architecture.md   # Architectural blueprint, inference code, and decisions
|   `-- model_artifacts/               # Generated upon notebook execution (models, plots, metrics)
|-- LICENSE                            # MIT License
|-- README.md                          # Project documentation
`-- requirements.txt                   # Pinned production dependencies
```

---

## Dataset & Problem Formulation

### Source Data

The dataset comprises operational telemetry from 200 lithium-ion battery packs representing diverse cell chemistries, operating environments, and charging profiles:

| Dataset Component | Dimensions | Description |
| --- | --- | --- |
| `battery_observations.csv` | 172,779 rows, 31 columns | Cycle-by-cycle sensor readings including voltages, currents, temperatures, C-rates, and state indicators |
| `battery_metadata.csv` | 200 rows, 7 columns | Battery-level metadata: chemistry (`LFP`, `NMC`, `NCA`), manufacturer, nominal capacity, climate zone, and usage strategy |
| Battery ID Partitions | 140 / 30 / 30 batteries | Pre-allocated train, validation, and test split indices |

### Data Cleaning and Target Handling

1. **Target Censoring**: Observations where `remaining_useful_cycles` is null (43,256 rows) represent battery packs that did not reach End of Life (EOL) within the observation window. These censored rows are excluded from supervised training to prevent target ambiguity.
2. **Missing Value Imputation**: The sensor feature `pack_temp_c` contains ~1% missing entries. Imputation is performed on a per-battery basis using each battery's individual median, with a global training median fallback.
3. **Leakage Elimination**: The fields `remaining_useful_efc` (a direct linear monotonic proxy of the target) and `eol_reached` (a binary trigger at 0 cycles) are stripped from feature sets prior to modeling. System identifiers (`battery_id`, `timestamp`) are dropped from predictive features.

---

## Physics-Informed Feature Engineering

To capture electro-thermal fatigue and aging dynamics beyond raw sensor outputs, domain-specific features were engineered:

| Engineered Feature | Formulation | Domain Rationale |
| --- | --- | --- |
| `soc_swing` | `soc_start_pct - soc_end_pct` | Measures effective depth of usage per cycle; larger swings amplify mechanical stress on electrode lattices |
| `charge_discharge_ratio` | `charge_c_rate / (discharge_c_rate + 1e-6)` | Captures charging asymmetry; elevated charge C-rates accelerate lithium plating and SEI layer growth |
| `temp_stress` | `abs(pack_temp_c - 25) * (1 + cumulative_high_temp_hours / 100)` | Quantifies acute deviation from optimal 25 deg C coupled with cumulative lifetime thermal exposure |
| `energy_per_efc` | `energy_discharged_kwh / (equivalent_full_cycles + 1e-6)` | Evaluates energy throughput efficiency, which systematically degrades as internal impedance climbs |
| `voltage_window` | `pack_voltage_charge_v - pack_voltage_discharge_v` | Terminal voltage spread; narrows as cell internal resistance increases over lifetime |
| `cycle_age_ratio` | `cycle_index / (battery_age_days + 1e-6)` | Cycling frequency proxy; identifies accelerated cycling regimes versus calendar aging |
| `soh_degradation_rate` | `state_of_health_pct / (cycle_index + 1)` | Longitudinal rate of capacity retention relative to cycle progression |

---

## Preprocessing Pipeline

Data transformation is orchestrated via an isolated `sklearn.compose.ColumnTransformer` fitted strictly on training data:

```
Raw Features
   |
   |-- Numeric Pipeline (30 features)
   |     |-- SimpleImputer(strategy='median')
   |     `-- RobustScaler(quantile_range=(25.0, 75.0))
   |
   `-- Categorical Pipeline (5 features)
         |-- SimpleImputer(strategy='most_frequent')
         `-- OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
```

- **RobustScaler** was selected over standard z-score normalization because operational sensor telemetry contains genuine operational spikes (e.g., peak fast-charging currents, high-temperature transients) that distort mean and variance estimates.
- **OrdinalEncoder** preserves split-efficiency for tree-based algorithms without incurring the high dimensionality and sparsity of one-hot representations.

---

## Model Exploration & Benchmarking

Eight candidate regression architectures across three model families were benchmarked under default hyperparameter configurations against the battery-level validation partition:

1. **Regularized Linear**: Ridge Regression, Lasso, ElasticNet
2. **Bagging Ensembles**: Extra Trees Regressor, Random Forest Regressor
3. **Gradient Boosting**: XGBoost, LightGBM, CatBoost

### Architectural Insights

- **Linear Models Underperform**: Due to severe multi-factor interactions between temperature, C-rate, and state of health, linear models exhibit high bias and inadequate variance explanation.
- **Gradient Boosting Dominance**: XGBoost, LightGBM, and CatBoost demonstrated the lowest validation RMSE and MAE. Tree boosting effectively partitions the multi-dimensional degradation surface and captures abrupt non-linear inflection points (e.g., knee points in battery aging curves).

The three top-performing architectures (**XGBoost**, **LightGBM**, and **CatBoost**) were advanced to systematic Bayesian hyperparameter optimization.

---

## Bayesian Hyperparameter Optimization

Optuna with the Tree-structured Parzen Estimator (TPE) algorithm was employed to minimize validation RMSE over 60 trials per model:

| Model | Hyperparameter Search Space |
| --- | --- |
| **XGBoost** | `n_estimators` (200-1000), `max_depth` (3-10), `learning_rate` (0.01-0.3, log), `subsample` (0.5-1.0), `colsample_bytree` (0.5-1.0), `reg_alpha` (1e-4-10.0, log), `reg_lambda` (1e-4-10.0, log), `min_child_weight` (1-10) |
| **LightGBM** | `n_estimators` (200-1000), `max_depth` (3-12), `num_leaves` (16-256), `learning_rate` (0.01-0.3, log), `subsample` (0.5-1.0), `colsample_bytree` (0.5-1.0), `reg_alpha` (1e-4-10.0, log), `reg_lambda` (1e-4-10.0, log), `min_child_samples` (5-100) |
| **CatBoost** | `iterations` (200-1000), `depth` (4-10), `learning_rate` (0.01-0.3, log), `l2_leaf_reg` (0.01-10.0, log), `subsample` (0.5-1.0), `colsample_bylevel` (0.5-1.0) |

Following study completion, optimal configurations were retrained on the aggregated train and validation sets (170 batteries, ~107,000 observations) to maximize training volume prior to final test evaluation.

---

## Ensemble Strategy

To enhance generalization and reduce prediction variance, predictions from the three tuned models are blended using an inverse validation RMSE weighting mechanism:

$$w_i = \frac{1 / \text{RMSE}_i}{\sum_{j} (1 / \text{RMSE}_j)}$$

$$\hat{y}_{\text{ensemble}} = \max\left(0, \sum_{i \in \{\text{XGB}, \text{LGB}, \text{CAT}\}} w_i \cdot \hat{y}_i\right)$$

This strategy provides an objective, data-driven weight allocation that favors the most accurate individual model while retaining regularizing diversity from the other boosting paradigms.

---

## Evaluation Framework

Model performance is assessed on the completely unseen 30-battery test set across four standard regression metrics:

- **Mean Absolute Error (MAE)**: Measures average absolute magnitude of errors in cycle units.
- **Root Mean Squared Error (RMSE)**: Penalizes severe prediction outliers.
- **Mean Absolute Percentage Error (MAPE)**: Assesses relative percentage error against true cycle life.
- **Coefficient of Determination ($R^2$)**: Proportion of variance explained by model predictions.

Diagnostic assessments include:
- Prediction vs. actual scatter alignment
- Residual distribution normality and heteroskedasticity checks
- Segmented error breakdowns across battery chemistries (`LFP`, `NMC`, `NCA`) and degradation states (`Normal`, `Moderate`, `Severe`, `Critical`)

---

## Model Explainability (SHAP Analysis)

Model interpretability is established using TreeSHAP on a 3,000-sample test cohort:

- **Global Feature Ranking**: Identifies the primary drivers governing degradation predictions across all battery packs.
- **Feature Impact Polarity (Beeswarm)**: Evaluates directional contributions (e.g., lower state of health and higher internal resistance driving lower predicted remaining cycles).
- **Non-Linear Dependence**: Isolates interaction effects between primary operating stressors (temperature, charging rate) and degradation response curves.

---

## Deployment & Inference Pipeline

All fitted estimators and transformation components are persisted in `Model/model_artifacts/`. The following production code demonstrates programmatic inference on new operational cycles:

```python
import json
import joblib
import numpy as np
import pandas as pd

# 1. Load serialised artifacts
preprocessor = joblib.load('Model/model_artifacts/preprocessor.joblib')
model_xgb    = joblib.load('Model/model_artifacts/best_xgboost.joblib')
model_lgb    = joblib.load('Model/model_artifacts/best_lightgbm.joblib')
model_cat    = joblib.load('Model/model_artifacts/best_catboost.joblib')

with open('Model/model_artifacts/ensemble_weights.json', 'r') as f:
    weights = json.load(f)

# 2. Transform new telemetry data
# Expected: DataFrame containing all raw and engineered feature columns
X_new_transformed = preprocessor.transform(new_telemetry_df[feature_columns])

# 3. Model inference with zero-floor clipping
pred_xgb = np.clip(model_xgb.predict(X_new_transformed), 0, None)
pred_lgb = np.clip(model_lgb.predict(X_new_transformed), 0, None)
pred_cat = np.clip(model_cat.predict(X_new_transformed), 0, None)

# 4. Weighted ensemble prediction
final_ruc_predictions = (
    weights['xgboost']  * pred_xgb +
    weights['lightgbm'] * pred_lgb +
    weights['catboost'] * pred_cat
)
```

---

## Getting Started

### Prerequisites

Python 3.10+ is recommended. Ensure pip and virtual environment tools are installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NumiKun/Battery-Remaining-Useful-Life.git
   cd Battery-Remaining-Useful-Life
   ```

2. Create and activate an isolated virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

Launch Jupyter Lab or Notebook and run the modeling pipeline:

```bash
jupyter lab Model/remaining_useful_cycles_regression.ipynb
```

Executing all notebook cells sequentially will perform data loading, preprocessing, model benchmarking, hyperparameter search, ensemble generation, evaluation, and SHAP explainability, saving all resulting artifacts into `Model/model_artifacts/`.

---

## Engineering Limitations & Roadmap

1. **Survival and Censoring Integration**: Current pipeline excludes un-failed batteries (43,256 observations). Integrating Cox Proportional Hazards or Accelerated Failure Time (AFT) models would leverage right-censored data points.
2. **Temporal Windowing**: The existing feature representation is cycle-static. Implementing rolling statistical aggregations (e.g., 5-cycle rolling standard deviation of cell voltage) would better encode dynamic degradation trajectories.
3. **Sequential Architectures**: Exploring Recurrent Neural Networks (LSTM, GRU) or Temporal Convolutional Networks (TCN) trained on historical cycle sequences to model long-term memory effects.
4. **Grouped Cross-Validation**: Expanding from single train/validation split to repeated battery-level K-Fold cross-validation for tighter confidence bounds on model error.

---

## License

This project is distributed under the terms of the MIT License. See the [LICENSE](LICENSE) file for complete details.
