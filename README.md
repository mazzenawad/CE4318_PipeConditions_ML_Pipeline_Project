# Underground Pipe Condition Predictor

A machine learning pipeline that classifies the structural condition of underground pipes (1–5 condition rating) from physical, environmental, and contextual features. The pipeline is built around an XGBoost classifier with a leak-proof preprocessing and resampling stack, and ships with a custom robustness suite — bootstrap confidence intervals and noise-perturbation testing — for evaluation that goes beyond headline accuracy.

The repository is structured as a reproducible end-to-end workflow: raw data → EDA → preprocessing → class balancing → training → evaluation, with all intermediate artifacts saved to disk for inspection. It is intended as a teaching-grade reference implementation rather than a production deployment.

## Project purpose

This project demonstrates an end-to-end ML workflow for civil infrastructure condition assessment:

1. ingest the raw pipe inventory dataset and run targeted exploratory analysis,
2. preprocess heterogeneous features (skewed numerics, standard numerics, and categoricals) inside a single `ColumnTransformer`,
3. apply SMOTE strictly on the training fold via an `imblearn` pipeline to prevent leakage into validation or test data,
4. train and tune an XGBoost classifier on the resampled training data,
5. evaluate the model with bootstrap-based confidence intervals and a noise-perturbation robustness check, and
6. export figures, metrics, and the serialized model to disk.

## Workflow diagram
```mermaid
graph TD
    %% Define Styles
    classDef data fill:#e1f5fe,stroke:#039be5,stroke-width:2px,color:#000
    classDef process fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000
    classDef model fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#000
    classDef eval fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#000

    %% Nodes
    A[Raw Data<br/>pipe_condition.csv]:::data
    B[Feature Engineering<br/>Age_x_Soil_PH]:::process
    C{Train/Test Split}:::process
    
    D[ColumnTransformer Preprocessing<br/>Median/Mean Imputation & Scaling<br/>OHE for Categoricals]:::process
    
    E[SMOTE Resampling<br/>Strictly on Training Data]:::process
    F[XGBoost Regressor<br/>reg:squarederror]:::model
    
    G[Prediction Output]:::data
    H[Regress-and-Round<br/>Round & Clip to 1-5 Scale]:::process
    
    I[Ordinal Evaluation<br/>QWK & Mean Absolute Error]:::eval
    J[Robustness Suite<br/>Bootstrap CI & Noise Perturbation]:::eval

    %% Flow
    A --> B
    B --> C
    C -- Training Data --> D
    C -- Test Data --> D
    
    D -- Processed Train --> E
    E --> F
    
    D -- Processed Test --> F
    F --> G
    G --> H
    
    H --> I
    H --> J
```

## Repository structure

```
CE4318_PipeConditions_ML_Pipeline_Project/
│
├── data/
│   ├── raw/
│   │   └── pipe_condition_class_synthetic.csv
│   └── processed/
│
├── notebooks/                  # Original Jupyter notebooks for EDA and initial testing
│   └── Notebook_Pipe_Conditions.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py          # loading, EDA helpers, feature engineering
│   ├── train.py              # ColumnTransformer + imblearn pipeline + XGBoost
│   ├── evaluate.py           # accuracy/F1, bootstrap CIs, noise perturbation
│   ├── visualize.py          # confusion matrix, feature importance
│   └── run_pipeline.py       # orchestrates the full pipeline end-to-end
│
├── output/
│   ├── models/
│   │   └── best_xgb_model.joblib
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── bootstrap_distribution.png
│   │   └── noise_robustness.png
│   └── results/
│       ├── classification_report.txt
        ├── ordinal_evaluation_report.txt  # Added to track QWK and MAE
│       └── bootstrap_metrics.csv
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## What each script does

### `src/data_prep.py`

Loads the raw CSV, runs a short EDA pass (distribution checks, missingness audit, target-class balance), and engineers the `Age × Soil_PH` interaction term. Returns a cleaned feature matrix and target vector ready for the modeling stage. Keeping data preparation isolated from training makes the pipeline easier to debug and reproduce.

### `src/train.py`

Defines and fits the modeling pipeline:

1. A `ColumnTransformer` that routes columns into three preprocessing branches:
   * **Skewed numerics** — median imputation + `StandardScaler`,
   * **Standard numerics** — mean imputation + `StandardScaler`,
   * **Categoricals** — most-frequent imputation + `OneHotEncoder`.
2. An `imblearn` pipeline that chains the `ColumnTransformer`, `SMOTE`, and the `XGBClassifier` together. Wrapping SMOTE inside the pipeline guarantees it is fit only on the training fold during cross-validation and is never applied to held-out data.
3. A stratified train/test split followed by model fitting and serialization to `output/models/best_xgb_model.joblib`.

### `src/evaluate.py`

Runs the full evaluation suite:

* **Standard metrics** — accuracy, macro-F1, per-class precision/recall, full classification report.
* **Bootstrap evaluation** — 100 resampled test folds; reports 95% confidence intervals on accuracy and F1, giving an honest read on the variance of the headline numbers.
* **Noise-perturbation robustness** — Gaussian noise injected at increasing scales (relative to each feature's standard deviation) onto the numerical features. Tracks accuracy degradation as input quality drops, simulating the noisy-sensor and stale-record conditions typical of municipal infrastructure data.

### `src/visualize.py`

Generates and saves diagnostic figures:

* confusion matrix on the test set,
* XGBoost feature importance (gain-based),
* bootstrap metric distribution, and
* noise-robustness curve.

### `src/run_pipeline.py`

Master script. Calls `data_prep` → `train` → `evaluate` → `visualize` in order and writes all artifacts under `output/`. Intended as the single entry point for full reproduction.

## Data overview

The dataset is a synthetic pipe inventory simulating municipal infrastructure records. Each row corresponds to a single pipe segment.

* **Numerical features:** `Age`, `Diameter`, `Slope`, `Depth`, `Length`, `Soil PH`
* **Categorical features:** `Material` (PVC, VCP, RC, …), `Soil Type` (Clay, Sand, Loam, …), `Road Type`
* **Engineered feature:** `Age × Soil_PH` (interaction term capturing accelerated chemical degradation in older pipes exposed to non-neutral soil)
* **Target:** `Condition Rating` — ordinal 1–5

Because the dataset is synthetic, results should be read as a demonstration of the pipeline structure and evaluation methodology, not as field-validated condition predictions.

## Installation

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies:

* numpy
* pandas
* scikit-learn
* imbalanced-learn
* xgboost
* matplotlib
* seaborn
* joblib

## Recommended usage

Run from the repository root.

### 1. Run the full pipeline

```
python src/run_pipeline.py
```

This executes data preparation, training, evaluation, and visualization in sequence and writes all outputs to `output/`.

### 2. Run individual stages (for iterating)

```
python src/data_prep.py
python src/train.py
python src/evaluate.py
python src/visualize.py
```

## Expected outputs

After a complete run:

### Models
* `output/models/best_xgb_model.joblib`

### Figures
* `output/figures/confusion_matrix.png`
* `output/figures/feature_importance.png`
* `output/figures/bootstrap_distribution.png`
* `output/figures/noise_robustness.png`

### Results
* `output/results/classification_report.txt`
* `output/results/bootstrap_metrics.csv`

## Notes

* SMOTE is applied only on training folds via the `imblearn` pipeline. This prevents synthetic samples from contaminating validation or test data — a common but silent source of inflated metrics that is easy to introduce when SMOTE is run as a standalone preprocessing step before splitting.
* The bootstrap evaluation uses 100 resamples by default. Increase the iteration count in `evaluate.py` for tighter confidence intervals at the cost of runtime.
* Noise perturbation is applied only to numerical features and is scaled relative to each feature's standard deviation, so the test is invariant to feature units.
* All scripts use relative paths anchored to the repository root; the directory structure should remain unchanged for reproducibility.

## License

MIT — see `LICENSE`.
