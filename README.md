# Cardiac Arrest Risk Modeling

## Project overview

This repository contains a completed, reproducible workflow for exploratory analysis, statistical modeling, predictive modeling, final evaluation, interpretability, and subgroup robustness checks on a tabular cardiac arrest risk dataset. The project is intended for retrospective data science analysis and demonstration of a clinically aware machine learning workflow; it is **not** intended for direct clinical deployment.

The workflow emphasizes:

- preserving raw source data unchanged;
- storing derived datasets in `data/processed/`;
- keeping reusable preprocessing and feature logic in `src/`;
- recording tabular outputs in `reports/`;
- saving reproducible model artifacts in `models/`; and
- documenting clinical, ethical, and validation limitations before any real-world use.

## Dataset

| File | Description |
| --- | --- |
| `data/CardiacPatientData.csv` | Raw patient-level clinical observations, including demographics, vital signs, laboratory values, lifestyle/history indicators, triage score, and outcome label. Treat this file as read-only. |
| `data/processed/cardiac_patient_processed.csv` | Derived dataset with deterministic engineered clinical features created from the raw CSV. Regenerate this file instead of editing it manually. |

The raw dataset includes:

- **Identifiers and demographics:** `ID`, `Age`, `Gender`
- **Vital signs:** `SBP`, `DBP`, `HR`, `RR`, `BT`, `SpO2`
- **Neurological/triage features:** `GCS`, `TriageScore`
- **Laboratory values:** `Na`, `K`, `Cl`, `Urea`, `Ceratinine`
- **Lifestyle and family history:** `Alcoholic`, `Smoke`, `FHCD`
- **Target:** `Outcome`

> `Ceratinine` appears to be a misspelling of `Creatinine`. Keep the raw column name unchanged in `data/CardiacPatientData.csv`; normalize or document it only in derived analysis outputs.

### Data handling rules

- Do **not** edit, overwrite, rename, or clean `data/CardiacPatientData.csv` in place.
- Place cleaned, transformed, or feature-engineered datasets under `data/processed/`.
- Keep generated report tables under `reports/` and generated figures under `reports/figures/`.
- Keep generated model binaries under `models/`; `models/*.joblib` files are ignored by Git and should be regenerated as needed.

## Repository structure

```text
Cardiac-Arrest-Risk-Modeling/
├── README.md
├── requirements.txt
├── data/
│   ├── CardiacPatientData.csv
│   └── processed/
│       └── cardiac_patient_processed.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── 01_data_quality_assessment.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_baseline_logistic_regression.ipynb
│   ├── 04_statistical_analysis_odds_ratios.ipynb
│   ├── 05_predictive_modeling.ipynb
│   ├── 06_final_model_evaluation.ipynb
│   ├── 07_model_interpretability.ipynb
│   └── 08_bias_fairness_robustness.ipynb
├── reports/
│   ├── data_dictionary.md
│   ├── final_report.md
│   ├── baseline_logistic_regression_metrics.csv
│   ├── final_model_metrics.csv
│   ├── model_comparison.csv
│   ├── odds_ratio_results.csv
│   ├── robustness_checks.csv
│   ├── subgroup_missingness_tests.csv
│   ├── subgroup_performance.csv
│   ├── threshold_analysis.csv
│   └── figures/
├── src/
│   ├── config.py
│   ├── create_processed_dataset.py
│   ├── features.py
│   └── preprocessing.py
└── tests/
    └── .gitkeep
```

## Setup instructions

1. Clone the repository and move into the project directory:

   ```bash
   git clone <repository-url>
   cd Cardiac-Arrest-Risk-Modeling
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Start Jupyter if you plan to run notebooks interactively:

   ```bash
   jupyter notebook
   ```

## How to run the analysis

From the repository root, first regenerate the processed dataset:

```bash
python src/create_processed_dataset.py
```

Then run the notebooks in numeric order from `notebooks/01_data_quality_assessment.ipynb` through `notebooks/08_bias_fairness_robustness.ipynb`. This order ensures that dataset checks, exploratory outputs, baseline metrics, model comparison results, the saved model artifact, final evaluation, interpretability outputs, and robustness tables are produced consistently.

The notebooks can be run interactively in Jupyter, or executed from the command line with a notebook runner such as `nbconvert`:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_quality_assessment.ipynb --inplace
```

Repeat the command for each notebook in the notebook order below.

## Notebook order

| Order | Notebook | Purpose | Key outputs |
| --- | --- | --- | --- |
| 1 | `01_data_quality_assessment.ipynb` | Checks shape, data types, missingness, duplicates, repeated IDs, clinical ranges, outliers, and raw data integrity. | Data-quality summaries and figures under `reports/` and `reports/figures/`. |
| 2 | `02_exploratory_data_analysis.ipynb` | Summarizes cohort characteristics, outcome balance, stratified summaries, distributions, group outcome rates, and correlations. | EDA figures under `reports/figures/`. |
| 3 | `03_baseline_logistic_regression.ipynb` | Builds a leakage-aware baseline logistic regression pipeline. | `reports/baseline_logistic_regression_metrics.csv` and baseline plots. |
| 4 | `04_statistical_analysis_odds_ratios.ipynb` | Estimates unadjusted and adjusted odds ratios for interpretable statistical analysis. | `reports/odds_ratio_results.csv`. |
| 5 | `05_predictive_modeling.ipynb` | Trains and compares candidate machine learning models with leakage-safe splitting and cross-validation. | `reports/model_comparison.csv` and `models/best_model.joblib`. |
| 6 | `06_final_model_evaluation.ipynb` | Evaluates the saved final model on the held-out test set, including threshold analysis, discrimination, calibration, and confusion matrices. | `reports/final_model_metrics.csv`, `reports/threshold_analysis.csv`, and final model figures. |
| 7 | `07_model_interpretability.ipynb` | Explains the final model with coefficients where applicable, permutation importance, and SHAP where compatible. | Interpretability figures under `reports/figures/`. |
| 8 | `08_bias_fairness_robustness.ipynb` | Evaluates subgroup performance, missingness patterns, and robustness to modeling/preprocessing alternatives. | `reports/subgroup_performance.csv`, `reports/subgroup_missingness_tests.csv`, and `reports/robustness_checks.csv`. |

## Generated reports

The repository includes generated tabular reports and narrative documentation:

| File | Description |
| --- | --- |
| `reports/data_dictionary.md` | Variable definitions, assumed encodings, clinical context, and data-quality notes. |
| `reports/final_report.md` | Final narrative report summarizing methods, results, interpretation, limitations, and next steps. |
| `reports/baseline_logistic_regression_metrics.csv` | Held-out performance metrics for the baseline logistic regression model. |
| `reports/odds_ratio_results.csv` | Unadjusted and adjusted odds-ratio results. |
| `reports/model_comparison.csv` | Candidate model comparison results from predictive modeling. |
| `reports/final_model_metrics.csv` | Final model metrics, selected operating threshold, and calibration/discrimination summaries. |
| `reports/threshold_analysis.csv` | Sensitivity, specificity, predictive values, and confusion matrix counts across thresholds. |
| `reports/subgroup_performance.csv` | Model performance by subgroup. |
| `reports/subgroup_missingness_tests.csv` | Missingness comparisons by subgroup. |
| `reports/robustness_checks.csv` | Robustness scenarios and performance comparisons. |

Generated figures are written to `reports/figures/`. Several figure patterns are ignored by Git because they can be regenerated from the notebooks.

## Model artifacts

The primary model artifact is:

| Artifact | Created by | Description |
| --- | --- | --- |
| `models/best_model.joblib` | `notebooks/05_predictive_modeling.ipynb` | Serialized scikit-learn pipeline for the best-performing candidate model. |

`models/*.joblib` files are intentionally ignored by Git. Recreate the final model artifact by running `notebooks/05_predictive_modeling.ipynb`; downstream evaluation and interpretability notebooks load this artifact when it is available.

## License

This project is free to use, copy, modify, and distribute as long as appropriate credit is given to the original author (Shreyans Jain/SJ-Jain-Systems). See the [LICENSE](LICENSE) file for details.

## Important clinical and ethical limitations

- This project is for retrospective analysis and model-development demonstration only; it is not a validated medical device or bedside decision-support tool.
- The dataset provenance, collection setting, inclusion criteria, exclusion criteria, outcome definition, measurement timing, and label quality must be confirmed before clinical interpretation.
- Repeated `ID` values may represent repeated encounters or measurements; splitting must remain leakage-safe at the patient or group level when repeated IDs are present.
- Associations and feature importances are not causal effects and should not be interpreted as treatment recommendations.
- Model performance may vary across demographic or clinical subgroups; subgroup, fairness, and missingness analyses are necessary but not sufficient for safe deployment.
- External validation on independent data is required before any operational use.
- Threshold selection must be based on explicit clinical tradeoffs, workflow capacity, risk tolerance, and prospective validation.
- Sensitive health data must be handled under applicable privacy, security, governance, and institutional review requirements.

## Future enhancements

- Add automated tests for preprocessing, feature engineering, and model training utilities.
- Add a lightweight pipeline runner, such as Make, DVC, Snakemake, or a Python CLI, to execute the full workflow from a clean checkout.
- Expand metadata with confirmed data provenance, cohort criteria, units, measurement timing, and outcome definitions.
- Compare model performance against established clinical scores or triage rules when appropriate.
- Add external and temporal validation datasets.
- Incorporate time-aware or survival modeling if timestamped longitudinal data becomes available.
- Add model-card documentation covering intended use, contraindications, subgroup behavior, monitoring, and maintenance requirements.
- Add continuous integration checks for notebooks, code formatting, and reproducibility.
