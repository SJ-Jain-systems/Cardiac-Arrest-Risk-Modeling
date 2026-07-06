# Model Card: Cardiac Arrest Risk Prediction Model

## 1. Model name

**Cardiac Arrest Risk Prediction Model**

The current selected model is a **Random Forest** classifier saved in the project metadata as `models/best_model.joblib`. The artifact is not version-controlled and should be regenerated from the modeling notebooks before any formal validation or deployment activity.

## 2. Intended use

This model is intended for **retrospective risk stratification research** using tabular clinical data related to cardiac arrest risk. Appropriate uses include:

- evaluating whether routinely available demographic, vital-sign, laboratory, history, and triage variables contain predictive signal for the coded cardiac arrest outcome;
- comparing transparent baseline models against machine-learning approaches under patient-ID-aware train/test splitting;
- identifying candidate clinical predictors for further study and prospective validation;
- supporting quality-improvement or research discussions about model operating thresholds, alert burden, calibration, and subgroup performance.

The model may be considered only as an investigational decision-support component after extensive external validation, prospective evaluation, clinical governance review, and workflow testing.

## 3. Not intended use

This model is **not intended** for:

- standalone diagnosis, triage, treatment selection, discharge decisions, or escalation/de-escalation of care;
- replacing clinician judgment, bedside assessment, established emergency protocols, or institutional early-warning systems;
- use in populations, hospitals, time periods, or clinical workflows that differ from the development data without external validation;
- automated denial or allocation of care resources;
- real-time clinical deployment without timestamp validation, calibration assessment, monitoring, and safety review.

## 4. Dataset description

The raw dataset is stored at `data/CardiacPatientData.csv`. It contains **5,906 observations**, **20 raw columns**, and **112 unique `ID` values**. Most identifiers repeat, with 108 IDs appearing more than once and the largest ID group containing 266 rows. Because repeated identifiers are common, model evaluation uses patient-ID-aware grouped splitting where possible to avoid placing the same ID in both training and test data.

Raw variable groups include:

| Variable group | Columns | Description |
| --- | --- | --- |
| Identifier | `ID` | Repeated patient, encounter, or episode identifier; exact row definition requires source confirmation. |
| Demographics | `Age`, `Gender` | Age and binary-coded gender; gender coding direction is not documented. |
| Vital signs | `SBP`, `DBP`, `HR`, `RR`, `BT`, `SpO2` | Blood pressure, heart rate, respiratory rate, body temperature, and oxygen saturation. |
| Neurologic / triage | `GCS`, `TriageScore` | Glasgow Coma Scale and triage acuity score; severity direction for triage should be confirmed. |
| Laboratory values | `Na`, `K`, `Cl`, `Urea`, `Ceratinine` | Electrolyte and renal/metabolic markers; `Ceratinine` appears to be a source misspelling of creatinine. |
| Lifestyle / history | `Alcoholic`, `Smoke`, `FHCD` | Binary-coded history indicators with substantial missingness; definitions require confirmation. |
| Target | `Outcome` | Binary outcome label. |

Important data quality findings include high positive-class prevalence, repeated IDs, 54 exact duplicate rows, substantial missingness in laboratory and history/triage fields, and physiologically implausible values such as `DBP = 999`, `HR = 381`, and `GCS = 100`. These issues must be reconciled before any clinical use.

## 5. Target variable

The target variable is **`Outcome`**, modeled as a binary classification target. The current analysis assumes:

- `Outcome = 1` is the positive event class;
- `Outcome = 0` is the negative class.

The raw dataset has a high positive-class rate: **5,053 of 5,906 rows (85.6%)** have `Outcome = 1`. The exact clinical definition, timing, and ascertainment process for `Outcome = 1` are not documented in the repository and must be confirmed against the source system or study protocol before clinical interpretation.

## 6. Features used

The model uses the available raw clinical predictors and deterministic engineered features. The `ID` field is used for grouped splitting and should not be used as a direct predictor.

### Raw predictors

- **Demographics:** `Age`, `Gender`
- **Vital signs:** `SBP`, `DBP`, `HR`, `RR`, `BT`, `SpO2`
- **Neurologic / triage:** `GCS`, `TriageScore`
- **Laboratory values:** `Na`, `K`, `Cl`, `Urea`, `Ceratinine`
- **Lifestyle / history:** `Alcoholic`, `Smoke`, `FHCD`

### Engineered predictors

| Engineered feature | Definition |
| --- | --- |
| `pulse_pressure` | `SBP - DBP` |
| `shock_index` | `HR / SBP` |
| `age_band` | Categorizes age as `<18`, `18-39`, `40-64`, `65-79`, or `80+`. |
| `hypoxemia_flag` | Flags `SpO2 < 90`. |
| `gcs_severity` | Categorizes GCS as severe, moderate, mild, or unknown. |
| `sodium_abnormal_flag` | Flags `Na` outside 135-145. |
| `potassium_abnormal_flag` | Flags `K` outside 3.5-5.0. |
| `chloride_abnormal_flag` | Flags `Cl` outside 98-107. |
| `urea_abnormal_flag` | Flags `Urea` outside 7-45. |
| `creatinine_abnormal_flag` | Flags `Ceratinine` outside 60-110. |

Preprocessing for mixed numeric and categorical inputs includes appropriate imputation and encoding learned from the training split only.

## 7. Model type

The final model is selected from five tuned candidates -- logistic regression, regularized (elasticnet) logistic regression, random forest, gradient boosting, and LightGBM with monotonic constraints -- each hyperparameter-searched with patient-grouped `RandomizedSearchCV` (`GroupKFold`, so no patient's rows span a search fold) via `src.modeling.tune_model`. Candidates are ranked by mean cross-validated AUROC (AUPRC as tie-breaker); the winning configuration and its hyperparameters are recorded in `reports/model_comparison.csv` and `models/best_model_metadata.json`.

Before saving, the winning pipeline is wrapped in `sklearn.calibration.CalibratedClassifierCV` (sigmoid/Platt scaling, fit via the same patient-grouped folds) because raw predicted probabilities from an uncalibrated model are not necessarily trustworthy as probabilities -- see the calibration-slope/intercept discussion in Section 8. Platt scaling was chosen over isotonic regression because the effective sample size here is only ~112 patients: isotonic's non-parametric fit is prone to instability at this scale, while Platt's 2-parameter logistic fit is far more stable. `models/best_model.joblib` therefore stores the complete calibrated pipeline, not a bare estimator.

### Monotonic constraints (LightGBM candidate)

The LightGBM candidate is fit with monotonic constraints so it cannot learn a clinically implausible sign flip on features with strong statistical evidence of direction. Constraints are derived mechanically from the adjusted odds-ratio model in `reports/odds_ratio_results.csv` (`src.modeling.derive_monotonic_constraints`): a statistically significant (p < 0.05) positive adjusted coefficient maps to a non-decreasing constraint (+1), a significant negative coefficient maps to non-increasing (-1), and every other numeric feature -- including the 9 not covered by the adjusted model, and any feature whose adjusted coefficient was not statistically significant -- is left **unconstrained (0)** rather than guessed from textbook physiology. The full mapping is saved to `reports/monotonic_constraint_directions.csv`.

| Feature | Constraint | Source | Coefficient | p-value |
| --- | --- | --- | --- | --- |
| `SBP` | +1 | Adjusted clinical model | 0.0428 | 5.0e-17 |
| `HR` | -1 | Adjusted clinical model | -0.0946 | 7.9e-50 |
| `RR` | -1 | Adjusted clinical model | -0.0905 | 7.1e-11 |
| `SpO2` | +1 | Adjusted clinical model | 0.1315 | 5.1e-07 |
| `Age` | -1 | Adjusted clinical model | -0.0300 | 2.7e-05 |
| `GCS` | 0 | Adjusted clinical model (not significant) | -0.0033 | 0.744 |
| `DBP`, `BT`, `Na`, `K`, `Cl`, `Urea`, `Ceratinine`, `pulse_pressure`, `shock_index` | 0 | Not covered by the adjusted model | -- | -- |

**Important caveat:** these directions reflect each feature's statistical association with the *currently coded* `Outcome` label in this dataset -- not confirmed clinical causality. The adjusted model itself found some directions that run counter to naive clinical intuition (e.g. higher HR/RR/Age associated with *lower* odds of the coded `Outcome=1`), and Section 5 notes that the exact clinical meaning and positive-class direction of `Outcome` are not independently confirmed. **If that meaning is later clarified and found to be inverted, every constraint sign above must be flipped**, and the LightGBM candidate retrained accordingly.

## 8. Evaluation metrics

Evaluation used a held-out, patient-ID-aware grouped test split with **1,160 rows**, **22 test IDs**, no overlapping train/test IDs, and a test event rate of **87.3%**.

### Discrimination and calibration

| Metric | Value |
| --- | ---: |
| AUROC | 0.881 |
| AUPRC | 0.980 |
| Brier score | 0.083 |
| Calibration intercept | -0.955 |
| Calibration slope | 1.489 |

### Recommended threshold performance

The recommended threshold is **0.75**, selected as the highest threshold maintaining sensitivity of at least 0.90 to limit false alarms while preserving high case detection.

| Metric at threshold 0.75 | Value |
| --- | ---: |
| Sensitivity / recall | 0.906 |
| Specificity | 0.476 |
| Positive predictive value | 0.923 |
| Negative predictive value | 0.424 |
| F1 score | 0.914 |
| Flagged high risk | 995 of 1,160 rows (85.8%) |
| Confusion matrix | TN 70, FP 77, FN 95, TP 918 |

Because the positive outcome is very common in this dataset, AUPRC and positive predictive value are strongly influenced by prevalence and may not generalize to lower-prevalence care settings.

## 9. Subgroup performance summary

Subgroup performance was evaluated across age bands, gender encodings, smoking status, alcohol status, family history of cardiovascular disease, GCS severity, and triage-score groups. Key findings include:

- **Age:** Some age bands had only positive outcomes in the test split, making AUROC and AUPRC undefined or unstable. The `80+` group had 70 observations and all were positive outcomes.
- **Gender:** The subgroup encoded `Gender = 0` had stronger performance than `Gender = 1` in the available test split. `Gender = 0` had AUROC 0.919 and specificity 0.606, while `Gender = 1` had AUROC 0.651 and specificity 0.104. The clinical meaning of the gender encoding must be confirmed before drawing fairness conclusions.
- **Smoking, alcohol, and family history:** Missing categories often had high outcome rates and substantial missingness, suggesting missing-not-at-random concerns.
- **Triage score:** The `3 / urgent` triage group had sensitivity 1.000 but specificity 0.000 at the selected threshold, indicating poor separation of non-events in that subgroup.
- **Robustness:** Alternative ID-aware split seeds produced materially different AUROC values, indicating that internal performance is sensitive to split composition and requires external validation.

## 10. Clinical interpretation

The model estimates the probability of the coded positive `Outcome` using clinical indicators that are plausible for acute deterioration risk, including oxygenation, hemodynamics, heart and respiratory rates, neurologic status, renal/electrolyte markers, age, history indicators, and triage acuity.

At the recommended threshold, the model prioritizes sensitivity: it detects most positive cases but flags a large proportion of observations as high risk. A high-risk prediction should be interpreted as a prompt for clinical review rather than a diagnosis. A low-risk prediction should be interpreted cautiously because the negative predictive value is limited and there were 95 false negatives in the held-out test split.

Feature importance and model explanations describe predictive associations learned by the model. They do **not** prove causality and should not be interpreted as evidence that changing a single feature will necessarily reduce or increase cardiac arrest risk.

## 11. Limitations

Major limitations include:

1. **Outcome ambiguity:** The clinical definition and timing of `Outcome = 1` are undocumented.
2. **Repeated observations:** Rows are not independent, and repeated IDs may represent multiple time points, encounters, or duplicated charting.
3. **Missingness:** Laboratory and history/triage variables have substantial missingness that may be clinically informative and non-random.
4. **Data quality concerns:** Implausible physiologic values and duplicate rows require source-system review.
5. **No external validation:** Reported results are based on internal grouped splits only.
6. **High event prevalence:** Metrics may not transfer to settings with different base rates.
7. **Calibration uncertainty:** Calibration metrics suggest predicted probabilities may require recalibration before risk communication.
8. **Subgroup uncertainty:** Protected-class definitions and subgroup sizes are insufficient for definitive fairness conclusions.
9. **Temporal uncertainty:** The dataset lacks timestamps needed to verify that predictors preceded outcomes.
10. **No deployment study:** Alert fatigue, clinician response, patient outcomes, and operational safety have not been evaluated.

## 12. Ethical considerations

Use of this model in clinical settings would raise significant ethical obligations:

- ensure predictions support, rather than replace, clinician judgment;
- prevent automation bias and overreliance on model scores;
- communicate uncertainty, calibration limitations, and false-negative risk clearly;
- protect patient privacy and comply with applicable data governance requirements;
- validate performance in the intended clinical population before deployment;
- establish accountability for model updates, failures, and clinical escalation pathways;
- assess whether alerts increase workload, alarm fatigue, or inequitable care delivery.

## 13. Bias and fairness concerns

The current analysis identifies several fairness risks:

- subgroup performance differs across encoded gender categories;
- missingness varies across demographic and clinical subgroups;
- subgroup encodings, especially gender, are not sufficiently documented;
- some subgroup sample sizes are small or contain only one outcome class;
- high event prevalence and repeated observations may amplify patterns from overrepresented patients or encounters;
- model behavior may not generalize to populations with different demographics, disease prevalence, care access, measurement practices, or triage workflows.

Before deployment, fairness evaluation should be repeated with confirmed protected-class definitions, clinically meaningful subgroup definitions, external data, calibrated predictions, and stakeholder-approved thresholds.

## 14. Monitoring recommendations

If this model is ever advanced beyond research, it should be monitored continuously for:

- input data schema changes, unit changes, and categorical coding changes;
- missingness rates and out-of-range physiologic values;
- population drift in demographics, acuity, prevalence, and care setting;
- discrimination metrics such as AUROC and AUPRC;
- calibration performance and threshold-specific sensitivity, specificity, PPV, and NPV;
- subgroup performance and fairness metrics;
- alert volume, false-positive burden, false-negative cases, and clinician override patterns;
- clinical outcomes, unintended consequences, and workflow burden;
- retraining triggers, version control, approval records, and audit logs.

Monitoring should include predefined review intervals, safety thresholds, escalation procedures, and rollback plans.

## 15. Clinical safety warning

> **Warning:** This model should **not** be used as a standalone clinical decision tool. It has not been externally validated, prospectively evaluated, calibrated for deployment, or approved for clinical use. Any output must be reviewed in context by qualified healthcare professionals and should never replace clinical judgment, emergency protocols, or institutional standards of care.
