# Cardiac Arrest Risk Modeling

This repository contains a tabular clinical dataset for building a cardiac arrest risk modeling workflow. The dataset has been organized under `data/` so analysis code, notebooks, reports, and model artifacts can be added without mixing them with raw data files.

## Dataset

| File | Description |
| --- | --- |
| `data/CardiacPatientData.csv` | Patient-level clinical observations with vital signs, demographics, laboratory values, lifestyle/history indicators, triage score, and outcome label. |

The current CSV includes the following fields:

- **Identifiers and demographics:** `ID`, `Age`, `Gender`
- **Vital signs:** `SBP`, `DBP`, `HR`, `RR`, `BT`, `SpO2`
- **Neurological/triage features:** `GCS`, `TriageScore`
- **Laboratory values:** `Na`, `K`, `Cl`, `Urea`, `Ceratinine`
- **Lifestyle and family history:** `Alcoholic`, `Smoke`, `FHCD`
- **Target:** `Outcome`

> Note: `Ceratinine` appears to be a misspelling of `Creatinine`. Keep the raw column name unchanged in the source data, but document or normalize it in downstream cleaned datasets.

## Project Roadmap

A strong bioinformatics or clinical data analysis project should move from raw data understanding to validated, interpretable, and reproducible modeling. The roadmap below can guide future development of this repository.

### 1. Project setup and reproducibility

- Create a consistent Python or R environment file, such as `requirements.txt`, `environment.yml`, `pyproject.toml`, or `renv.lock`.
- Add a notebook or script structure, for example:
  - `notebooks/` for exploratory analysis
  - `src/` for reusable preprocessing and modeling code
  - `reports/` for generated figures and summaries
  - `models/` for saved model artifacts, if appropriate
- Define data handling rules so raw data remains unchanged and derived datasets are stored separately, such as `data/processed/`.
- Add a reproducible pipeline tool if the project grows, such as Make, Snakemake, Nextflow, DVC, or a lightweight Python CLI.

### 2. Data dictionary and clinical context

- Build a data dictionary describing each variable, unit of measurement, expected range, encoding, and clinical interpretation.
- Confirm binary encodings for columns such as `Gender`, `Alcoholic`, `Smoke`, `FHCD`, and `Outcome`.
- Clarify whether each row is an independent patient, repeated measurement, encounter, or time point, especially because `ID` values may repeat.
- Document inclusion criteria, exclusion criteria, collection setting, and outcome definition.
- Identify whether the task is risk prediction, triage support, retrospective association analysis, or another clinical objective.

### 3. Data quality assessment

- Check missing values, duplicate rows, repeated patient IDs, invalid values, and impossible physiological measurements.
- Validate numeric ranges for vital signs, electrolytes, renal function markers, age, GCS, and triage score.
- Review inconsistent labels, spelling issues, and unit assumptions.
- Decide how to handle repeated observations from the same patient to avoid data leakage between train and test sets.
- Produce a data quality report with summary tables and visualizations.

### 4. Exploratory data analysis

- Summarize cohort characteristics overall and stratified by `Outcome`.
- Visualize distributions of vitals, labs, age, GCS, and triage score.
- Compare outcome rates across clinically relevant groups, such as age bands, SpO2 ranges, GCS categories, smoking status, and family history.
- Examine correlations among predictors and detect multicollinearity.
- Evaluate class balance for the `Outcome` label.

### 5. Preprocessing and feature engineering

- Split data into training, validation, and test sets before fitting transformations.
- Use patient-level splitting if `ID` represents repeated measurements from the same individual.
- Impute missing values using training data only.
- Encode categorical variables consistently.
- Scale or transform numeric variables where needed.
- Consider clinically meaningful derived features, such as:
  - Pulse pressure: `SBP - DBP`
  - Shock index: `HR / SBP`
  - Age bands
  - Hypoxemia indicator from `SpO2`
  - GCS severity category
  - Electrolyte abnormality flags

### 6. Baseline statistical analysis

- Fit simple baseline models before complex models.
- Start with interpretable approaches such as logistic regression and regularized logistic regression.
- Estimate effect sizes with confidence intervals where appropriate.
- Compare unadjusted and adjusted associations between predictors and outcome.
- Report assumptions and limitations clearly.

### 7. Predictive modeling

- Train multiple candidate models, such as:
  - Logistic regression
  - Random forest
  - Gradient boosting models such as XGBoost, LightGBM, or CatBoost
  - Support vector machines or neural networks only if justified by performance and data size
- Use cross-validation that respects patient grouping if repeated IDs exist.
- Tune hyperparameters with a validation set or nested cross-validation.
- Avoid leakage by fitting preprocessing steps inside the cross-validation pipeline.

### 8. Model evaluation

- Report metrics that match the clinical use case, including:
  - AUROC
  - AUPRC
  - Sensitivity/recall
  - Specificity
  - Precision/positive predictive value
  - Negative predictive value
  - F1 score
  - Calibration slope and intercept
  - Brier score
- Choose operating thresholds based on clinical tradeoffs rather than accuracy alone.
- Include confusion matrices at selected thresholds.
- Evaluate calibration plots and decision curve analysis if the model may support clinical decisions.

### 9. Interpretability and biological or clinical insight

- Use model coefficients, permutation importance, SHAP values, or partial dependence plots to explain influential features.
- Compare model findings with known clinical risk factors for cardiac deterioration or arrest.
- Distinguish predictive signal from causal interpretation.
- Highlight clinically plausible patterns, surprising associations, and limitations.

### 10. Bias, fairness, and robustness

- Assess model performance across subgroups, such as age, gender, smoking status, and other clinically relevant categories.
- Check whether missingness or measurement patterns differ across groups.
- Evaluate robustness to outliers, alternative preprocessing choices, and different train/test splits.
- Document potential sources of bias, including single-site collection, selection bias, measurement bias, and label quality.

### 11. Reporting and documentation

- Create a final analysis report with methods, cohort summary, model results, interpretation, limitations, and next steps.
- Include reproducible figures and tables.
- Version model artifacts and processed datasets only when permitted.
- Add clear instructions for rerunning the analysis from a clean checkout.

### 12. Future enhancements

- Add external validation data if available.
- Compare model performance against existing clinical scores.
- Incorporate time-aware modeling if observations are longitudinal.
- Explore survival analysis if time-to-event labels are available.
- Add automated tests for preprocessing functions and model pipelines.
- Create a model card describing intended use, limitations, ethical considerations, and monitoring needs.

## Suggested Initial Repository Structure

```text
Cardiac-Arrest-Risk-Modeling/
├── README.md
├── data/
│   └── CardiacPatientData.csv
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── modeling.py
├── reports/
│   └── figures/
└── tests/
```

Only `README.md` and `data/CardiacPatientData.csv` currently exist. The remaining folders are recommended additions for future work.

## Immediate Next Steps

1. Create a data dictionary for every column.
2. Confirm whether repeated `ID` values represent repeated measurements from the same patient.
3. Run an exploratory data analysis notebook.
4. Build a leakage-safe baseline logistic regression model.
5. Evaluate discrimination, calibration, and clinically meaningful thresholds.
