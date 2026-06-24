# Final Report: Cardiac Arrest Risk Modeling

## 1. Project objective

The objective of this project is to develop a reproducible, clinically interpretable risk-modeling workflow for predicting the binary `Outcome` field in a cardiac arrest risk dataset. The project is framed as a retrospective tabular clinical prediction analysis rather than a deployable clinical decision-support product. The main goals are to:

- characterize the cohort and data quality issues before modeling;
- engineer clinically meaningful predictors from vital signs, demographics, laboratory measurements, history indicators, and triage information;
- compare transparent statistical baselines with machine-learning models using patient-ID-aware splitting;
- select an operating threshold that prioritizes sensitivity while limiting false alarms; and
- document interpretability, fairness, robustness, limitations, and reproducibility considerations.

The current analysis assumes `Outcome = 1` is the positive event class. Before any clinical use, the exact clinical meaning of `Outcome = 1` must be confirmed from the source system or study protocol.

## 2. Dataset description

The raw dataset is stored at `data/CardiacPatientData.csv` and contains 5,906 observations, 20 columns, and 112 unique `ID` values. Repeated identifiers are common: 108 IDs appear more than once, and the largest ID group contains 266 rows. This makes grouped splitting essential so that the same patient or encounter identifier does not appear in both training and test data.

The raw fields include:

| Variable group | Columns | Notes |
| --- | --- | --- |
| Identifier | `ID` | Repeated values suggest multiple observations per patient, encounter, or episode. |
| Demographics | `Age`, `Gender` | `Gender` is encoded as 0/1; source meaning should be confirmed. |
| Vital signs | `SBP`, `DBP`, `HR`, `RR`, `BT`, `SpO2` | Blood pressure, heart rate, respiratory rate, temperature, and oxygen saturation. |
| Neurologic/triage | `GCS`, `TriageScore` | `GCS` should ordinarily be 3-15; `TriageScore` appears to use levels 1-3. |
| Laboratory values | `Na`, `K`, `Cl`, `Urea`, `Ceratinine` | `Ceratinine` appears to be a misspelling of creatinine and is retained as the raw column name. |
| Lifestyle/history | `Alcoholic`, `Smoke`, `FHCD` | Binary indicators with missingness; definitions require source confirmation. |
| Target | `Outcome` | Binary outcome; positive class rate is high at 85.6%. |

A processed dataset is also available at `data/processed/cardiac_patient_processed.csv`; it preserves the raw columns and appends deterministic engineered features.

## 3. Clinical context

Cardiac arrest risk prediction is clinically high stakes because missed events can delay escalation of care, while excessive false alarms can contribute to alarm fatigue and unnecessary interventions. The variables in this dataset are consistent with emergency, inpatient, or acute-care risk assessment: vital signs can reflect hemodynamic instability or respiratory compromise; GCS can reflect neurologic deterioration; electrolyte and renal markers can capture metabolic abnormalities; and triage score may encode clinician-assessed acuity.

The analysis should therefore be interpreted as risk stratification support, not diagnosis. Model outputs are estimated probabilities or risk scores that require clinical validation, calibration review, workflow integration, and prospective testing before use in care.

## 4. Data quality findings

The data quality assessment found several issues that materially affect modeling and interpretation:

- **Repeated IDs and leakage risk:** The dataset contains 112 unique IDs across 5,906 rows, with most IDs repeated. All model evaluation tables therefore use grouped splitting by `ID` when possible.
- **Duplicate rows:** There are 54 exact duplicate rows in the raw data. These may reflect repeated measurements, data export duplication, or duplicated charting and should be reconciled with the source data.
- **Outcome imbalance:** `Outcome = 1` occurs in 5,053 of 5,906 rows, or 85.6%. Accuracy alone is therefore not an adequate metric.
- **Missingness:** Laboratory variables have substantial missingness: `Urea` 56.2%, `Ceratinine` 55.2%, `Cl` 55.0%, and both `Na` and `K` 54.9%. Triage/history variables are also frequently missing: `TriageScore` 32.2%, `Smoke` 29.7%, `Alcoholic` 29.7%, and `FHCD` 27.5%.
- **Physiologic plausibility flags:** Broad screening checks identified 14 rows where `DBP > SBP`, 48 rows with `GCS` outside the standard 3-15 range, and extreme vital-sign values such as `SBP = 444`, `DBP = 999`, `HR = 381`, and `RR = 100`.
- **Unit and definition uncertainty:** Temperature values cluster near 98, suggesting Fahrenheit. `Ceratinine` likely means creatinine, but units and spelling should be confirmed. The direction of `TriageScore` severity also needs confirmation.

Relevant generated or regenerated outputs include `reports/data_dictionary.md`, the data quality notebook outputs in `reports/figures/missing_values_by_column.png`, `reports/figures/clinical_numeric_distributions.png`, `reports/figures/clinical_variable_boxplots.png`, `reports/figures/outcome_distribution.png`, and `reports/figures/numeric_correlation_heatmap.png`. Binary figure files are intentionally ignored by version control and can be regenerated from the notebooks.

## 5. Exploratory data analysis summary

EDA confirmed a high-event-rate cohort with substantial repeated observations by ID and missingness concentrated in laboratory and history/triage variables. Outcome-stratified summaries and plots should be interpreted cautiously because repeated observations may overweight patients with many measurements.

Key EDA findings include:

- **Class balance:** The positive outcome class dominates, so AUROC, AUPRC, sensitivity, specificity, PPV, NPV, calibration, and confusion matrices are all needed to understand performance.
- **Vital signs and scores:** Distributions show clinically plausible central tendencies but include outliers requiring robustness checks.
- **Laboratory data:** Lab measurements are sparse and likely missing-not-at-random, because labs may have been ordered for clinically distinct patients.
- **Risk categories:** Age bands, SpO2 categories, GCS severity, and history indicators provide clinically readable summaries but must not be interpreted causally.
- **Correlations:** Correlation plots help detect redundant predictors, especially among related physiologic measurements.

Generated EDA figures are expected under `reports/figures/`, including `eda_02_outcome_class_balance.png`, `eda_04_distribution_<feature_group>.png`, `eda_05_numeric_boxplots_by_outcome.png`, `eda_06_outcome_rate_by_age_band.png`, `eda_07_outcome_rate_by_spo2_category.png`, `eda_08_outcome_rate_by_gcs_severity.png`, `eda_09_outcome_rates_history_gender.png`, and `eda_10_numeric_predictor_correlation_heatmap.png`.

## 6. Feature engineering

Feature engineering was intentionally limited to deterministic, clinically meaningful transformations so that derived columns can be applied separately within each data split without learning from the full dataset. The processed dataset appends the following features:

| Engineered feature | Definition | Clinical rationale |
| --- | --- | --- |
| `pulse_pressure` | `SBP - DBP` | Captures arterial pressure amplitude and may indicate vascular or hemodynamic changes. |
| `shock_index` | `HR / SBP` | Common instability marker combining heart rate and systolic pressure. |
| `age_band` | `<18`, `18-39`, `40-64`, `65-79`, `80+` | Improves subgroup analysis and interpretability. |
| `hypoxemia_flag` | `SpO2 < 90` | Flags clinically important oxygen desaturation. |
| `gcs_severity` | severe, moderate, mild, unknown | Converts GCS to clinically familiar severity groups. |
| `sodium_abnormal_flag` | `Na` outside 135-145 | Flags electrolyte abnormality. |
| `potassium_abnormal_flag` | `K` outside 3.5-5.0 | Flags electrolyte abnormality. |
| `chloride_abnormal_flag` | `Cl` outside 98-107 | Flags electrolyte abnormality. |
| `urea_abnormal_flag` | `Urea` outside 7-45 | Flags renal/metabolic abnormality. |
| `creatinine_abnormal_flag` | `Ceratinine` outside 60-110 | Flags possible renal abnormality while preserving raw spelling lineage. |

Fitted preprocessing steps such as imputation, one-hot encoding, scaling, and model fitting should be learned on the training split only.

## 7. Baseline statistical analysis

The baseline logistic regression analysis used a `StratifiedGroupKFold by patient ID` split with 4,746 training rows, 1,160 test rows, 90 training IDs, 22 test IDs, and no overlapping train/test IDs. At the default operating threshold, the baseline achieved:

| Metric | Baseline logistic regression |
| --- | ---: |
| AUROC | 0.858 |
| AUPRC | 0.979 |
| Accuracy | 0.822 |
| Sensitivity / recall | 0.898 |
| Specificity | 0.293 |
| PPV | 0.897 |
| NPV | 0.295 |
| F1 | 0.898 |
| Confusion matrix | TN 43, FP 104, FN 103, TP 910 |

The odds-ratio analysis in `reports/odds_ratio_results.csv` provides both unadjusted and adjusted associations. In the adjusted clinical model, several variables remained statistically associated with the positive class, including family history (`FHCD`, OR 26.38, 95% CI 11.73-59.28), SBP per 1 mmHg (OR 1.04, 95% CI 1.03-1.05), age per year (OR 0.97, 95% CI 0.96-0.98), HR per bpm (OR 0.91, 95% CI 0.90-0.92), RR per breath/min (OR 0.91, 95% CI 0.89-0.94), and SpO2 per percentage point (OR 1.14, 95% CI 1.08-1.20). These are associations with the coded outcome, not causal effects, and some coefficient directions may reflect label definition, sampling, repeated measurements, or confounding.

## 8. Predictive modeling methods

### Unit of analysis

The dataset contains 5,906 rows but only **112 unique patient IDs**. Rows per patient range from 1 to 266 (median ~50), and only 4 patients contribute a single row, so the records are repeated measurements rather than independent observations. The **effective sample size is therefore approximately 112 patients, not ~5,900 rows**.

For this reason, all model development and evaluation split and cross-validate at the **patient/ID level** using `StratifiedGroupKFold` / `GroupKFold`: every record for a patient is kept within a single fold, so no patient appears in both training and test. Row-level splitting would let a patient's correlated records straddle the split and produce optimistically biased metrics. Held-out folds contain as few as ~22 patients, which is the main reason the bootstrap confidence intervals reported in `reports/final_model_metrics.csv` are wide.

As a sensitivity analysis, notebook 08 adds a patient-level robustness arm (`reports/patient_level_robustness.csv`) that aggregates the data to one row per patient (maximum `Outcome` label, mean of numeric predictors) and refits the model with cross-validation. Patient-grouped row-level cross-validation gave AUROC ≈ 0.74, while the aggregated patient-level model gave AUROC ≈ 0.90 (5-fold) and ≈ 0.91 (leave-one-patient-out). The discrimination signal persists when each patient counts exactly once, but with only ~112 patients all of these estimates carry substantial uncertainty and should be treated as preliminary rather than confirmatory.

### Model comparison

The predictive modeling workflow compared multiple candidate models using patient-ID-aware splitting and cross-validation:

- Logistic Regression
- Regularized Logistic Regression
- Random Forest
- Gradient Boosting

The comparison table is stored in `reports/model_comparison.csv`. Candidate pipelines used preprocessing appropriate for mixed numeric and categorical predictors and were evaluated using AUROC, AUPRC, F1, sensitivity, specificity, precision, NPV, and confusion matrices.

The held-out model comparison ranked Random Forest highest by test AUROC:

| Rank | Model | Test AUROC | Test AUPRC | Test sensitivity | Test specificity | Test precision | Test F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Random Forest | 0.881 | 0.980 | 0.997 | 0.129 | 0.888 | 0.939 |
| 2 | Gradient Boosting | 0.860 | 0.979 | 0.972 | 0.122 | 0.884 | 0.926 |
| 3 | Regularized Logistic Regression | 0.855 | 0.978 | 0.892 | 0.320 | 0.900 | 0.896 |
| 4 | Logistic Regression | 0.852 | 0.977 | 0.898 | 0.503 | 0.926 | 0.912 |

The selected model was saved in the metrics metadata as `models/best_model.joblib`. Model artifact files are excluded from version control and can be regenerated by rerunning the modeling notebooks.

## 9. Final model evaluation

The final model evaluation is summarized in `reports/final_model_metrics.csv`. On the held-out grouped test set of 1,160 rows with an event rate of 87.3%, the final model achieved:

| Metric | Final model |
| --- | ---: |
| AUROC | 0.881 |
| AUPRC | 0.980 |
| Brier score | 0.083 |
| Calibration intercept | -0.955 |
| Calibration slope | 1.489 |
| Recommended threshold | 0.75 |
| Sensitivity at threshold | 0.906 |
| Specificity at threshold | 0.476 |
| PPV at threshold | 0.923 |
| NPV at threshold | 0.424 |
| F1 at threshold | 0.914 |
| Confusion matrix at threshold | TN 70, FP 77, FN 95, TP 918 |

The final model shows moderate-to-good discrimination and a high AUPRC, the latter inflated by the high positive-class prevalence. These are **point estimates from an effective sample of only ~112 patients (≈22 in the held-out fold)**, so they carry wide confidence intervals (reported in `reports/final_model_metrics.csv`) and should be read as preliminary rather than confirmatory. Calibration metrics indicate that predicted probabilities may require recalibration before clinical deployment. The low-to-moderate NPV also means a low-risk prediction should not be used to rule out risk without clinical oversight.

Generated final evaluation figures include `reports/figures/final_model_roc_curve.png`, `reports/figures/final_model_precision_recall_curve.png`, `reports/figures/final_model_calibration_curve.png`, and `reports/figures/final_model_confusion_matrices.png`.

## 10. Threshold selection

Threshold analysis is stored in `reports/threshold_analysis.csv`. The recommended threshold is **0.75**, selected as the highest threshold maintaining sensitivity of at least 0.90 to reduce false alarms while preserving high case detection.

At the recommended threshold:

- 995 of 1,160 test observations were flagged high risk (85.8%).
- Sensitivity was 0.906, meaning 918 of 1,013 positive cases were detected.
- Specificity was 0.476, meaning 70 of 147 non-events were correctly classified as low risk.
- PPV was 0.923, but NPV was only 0.424.
- There were 95 false negatives and 77 false positives.

The threshold represents a sensitivity-first strategy. If the intended use case is bedside escalation, this may be reasonable; if the use case is resource allocation with limited capacity, a threshold with higher specificity may be preferred. For example, threshold 0.85 increased specificity to 0.850 but reduced sensitivity to 0.829.

## 11. Interpretability and clinical insight

Interpretability analyses are designed to connect model behavior with clinically understandable predictors. The interpretability notebook generates permutation-importance and SHAP-based figures, including:

- `reports/figures/final_model_permutation_importance.png`
- `reports/figures/final_model_shap_importance.png`
- `reports/figures/final_model_shap_summary.png`

Clinically plausible predictors include oxygenation (`SpO2` and hypoxemia flags), hemodynamic measurements (`SBP`, `DBP`, pulse pressure, shock index), heart and respiratory rates, neurologic status (`GCS` and `gcs_severity`), renal/electrolyte markers, age, family history, and triage score. However, importance values describe model reliance for prediction, not causality.

Important interpretation caveats:

- Repeated observations may make the model learn patient- or encounter-specific patterns.
- Missingness may itself be informative if labs and triage/history variables are collected selectively.
- Some odds-ratio directions are counterintuitive for conventional clinical risk and may reflect the coding of `Outcome`, acute treatment effects, selection bias, or confounding.
- If the final estimator is tree-based, feature effects may be nonlinear and interaction-heavy.

## 12. Bias, fairness, and robustness

Fairness and subgroup results are reported in `reports/subgroup_performance.csv` and `reports/subgroup_missingness_tests.csv`. Robustness checks are reported in `reports/robustness_checks.csv`, and the patient-level (one row per patient) sensitivity analysis is in `reports/patient_level_robustness.csv` (see the Unit of analysis subsection in Section 8).

Subgroup analysis showed that performance and missingness varied across clinically relevant groups:

- **Age:** Some age bands had only positive outcomes in the test split, so AUROC/AUPRC were undefined. The `80+` group had 70 observations and all were positive outcomes, causing threshold-dependent metrics to be unstable.
- **Gender:** The subgroup encoded `Gender = 0` had AUROC 0.919 and specificity 0.606, while `Gender = 1` had AUROC 0.651 and specificity 0.104. This suggests potential subgroup performance disparity, although the clinical meaning of the gender encoding must be confirmed.
- **Smoking, alcohol, and family history:** Missing categories often had very high outcome rates and substantial missingness, indicating missing-not-at-random concerns.
- **Triage groups:** The `3 / urgent` group had sensitivity 1.000 but specificity 0.000 at the selected threshold, suggesting the threshold did not separate non-events in that subgroup.
- **Missingness by subgroup:** Multiple predictors had missingness tests with p-values below 0.05 across age, gender, smoking, alcohol, FHCD, and triage groups, supporting the need for missingness-aware robustness analysis.

Robustness checks found that AUROC varied across alternative ID-aware split seeds and preprocessing scenarios. Alternative split seeds produced AUROCs ranging from 0.814 to 0.950, and the primary engineered-feature robustness run produced AUROC 0.800. Raw-features-only modeling achieved AUROC 0.831, suggesting engineered features were helpful for interpretability but not uniformly necessary for discrimination. These results indicate that performance is sensitive to split composition and should be validated on an external cohort.

## 13. Limitations

The main limitations are:

1. **Outcome ambiguity:** The clinical meaning and timing of `Outcome = 1` are not documented in the data files.
2. **Repeated observations and small effective sample:** Rows are not independent. The 5,906 rows correspond to only 112 unique patients (one patient contributes 266 rows), so the **effective sample size is ~112 patients**. Headline metrics rest on this small number of patients — and as few as ~22 in any held-out fold — so estimates are uncertain regardless of the large row count, and all results should be interpreted at the patient level. The patient-level robustness arm in `reports/patient_level_robustness.csv` is a sensitivity check on this point.
3. **Missing-not-at-random predictors:** Laboratory and history/triage variables are missing for large portions of the cohort.
4. **Data quality outliers:** Implausible values such as `DBP = 999` and `GCS = 100` need source-system review.
5. **No external validation:** Current performance is based on internal grouped splits only.
6. **High event prevalence:** AUPRC and PPV are elevated by the high positive-class rate and may not generalize to settings with lower prevalence.
7. **Calibration uncertainty:** Calibration slope/intercept indicate the risk probabilities may need recalibration before clinical interpretation.
8. **Fairness uncertainty:** Subgroup encodings, especially `Gender`, require source confirmation before fairness conclusions can be considered definitive.
9. **Potential temporal leakage:** Without timestamps, the analysis cannot verify that all predictors were measured before the outcome.
10. **No deployment validation:** The project has not assessed clinical workflow, alert burden, clinician response, or patient outcomes.

## 14. Future work

Recommended next steps include:

- Confirm the data provenance, row definition, outcome definition, variable units, and categorical encodings.
- Add timestamp-aware cohort construction to ensure predictors precede outcomes.
- Reconcile repeated IDs and duplicate rows with source records.
- Add formal data quality tables, missingness mechanism analysis, and outlier adjudication.
- Evaluate calibration with recalibration methods such as Platt scaling, isotonic regression, or logistic recalibration on a validation set.
- Perform external validation on data from a different time period, hospital, or source system.
- Evaluate clinically meaningful operating thresholds with stakeholders and quantify alert burden.
- Add decision-curve analysis or net-benefit analysis to compare thresholds.
- Expand fairness analysis once protected-class definitions are confirmed.
- Package the pipeline into a single reproducible command or workflow tool, such as Make, Snakemake, DVC, or a Python CLI.

## 15. Reproducibility instructions

From the repository root, the analysis can be reproduced as follows.

1. Create and activate a Python environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Regenerate the processed dataset:

   ```bash
   python src/create_processed_dataset.py
   ```

4. Run the notebooks in order:

   ```bash
   jupyter notebook
   ```

   Execute:

   - `notebooks/01_data_quality_assessment.ipynb`
   - `notebooks/02_exploratory_data_analysis.ipynb`
   - `notebooks/03_baseline_logistic_regression.ipynb`
   - `notebooks/04_statistical_analysis_odds_ratios.ipynb`
   - `notebooks/05_predictive_modeling.ipynb`
   - `notebooks/06_final_model_evaluation.ipynb`
   - `notebooks/07_model_interpretability.ipynb`
   - `notebooks/08_bias_fairness_robustness.ipynb`

5. Review regenerated tabular outputs in `reports/`:

   - `reports/baseline_logistic_regression_metrics.csv`
   - `reports/odds_ratio_results.csv`
   - `reports/model_comparison.csv`
   - `reports/final_model_metrics.csv`
   - `reports/threshold_analysis.csv`
   - `reports/subgroup_performance.csv`
   - `reports/subgroup_missingness_tests.csv`
   - `reports/robustness_checks.csv`
   - `reports/patient_level_robustness.csv`

6. Review regenerated figures in `reports/figures/`. Binary figure outputs and model artifacts are intentionally excluded from version control by `.gitignore`; regenerate them from the notebooks when needed.

For reproducible model evaluation, maintain patient-ID-aware splitting and ensure all fitted preprocessing steps are trained only on the training data within each split or cross-validation fold.
