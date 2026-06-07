# Cardiac Arrest Dataset Data Dictionary

This data dictionary documents the raw variables in `data/CardiacPatientData.csv`. It preserves the source column names exactly as provided, including the raw column name `Ceratinine`, which appears to be a misspelling of **Creatinine**. Do not rename `Ceratinine` in the raw data; handle any cleaned or presentation-layer alias separately.

## General assumptions

- Numeric clinical measurements may contain physiologically implausible values, missing values, or data-entry placeholders; model pipelines should validate and flag these rather than silently correcting them.
- Binary variables are assumed to use `0 = no/absent` and `1 = yes/present` unless project documentation indicates otherwise.
- Clinical units are inferred from common emergency-care conventions and the observed value scale in the CSV; confirm against the data collection protocol before clinical interpretation.
- Expected ranges below describe plausible validation ranges for adult/patient screening and may be broader than normal reference ranges.

## Variables

| Variable name | Likely clinical meaning | Data type | Possible unit | Expected range | Encoding assumptions | Validation notes |
|---|---|---:|---|---|---|---|
| `ID` | Patient or encounter identifier. Repeated IDs may indicate multiple time-stamped observations for the same patient/encounter. | Integer / categorical identifier | Not applicable | Positive integer; uniqueness depends on data design | Treat as an identifier, not a numeric predictor unless intentionally deriving grouping features | Check for missing IDs, duplicated IDs, and whether rows with the same ID represent repeated measures rather than duplicate records. |
| `Age` | Patient age at presentation or event. | Integer or numeric | Years | 0-120 years; adult cardiac-arrest datasets commonly expected to be about 18-100 years | Direct numeric value | Flag ages outside plausible human range, negative values, and ages inconsistent with inclusion criteria. |
| `Gender` | Patient sex or gender category as encoded by the source dataset. | Binary categorical / integer | Not applicable | 0 or 1 if binary-coded | Exact mapping is not documented; likely one code for female and one for male, but do not assume direction without source metadata | Validate only allowed codes are present; document the confirmed code mapping before using in fairness or subgroup analyses. |
| `SBP` | Systolic blood pressure. | Numeric / integer | mmHg | 40-300 mmHg for broad clinical validation | Direct measurement | Very low, very high, zero, or impossible values should be flagged; values outside the plausible range may represent entry errors, device artifacts, or arrest/peri-arrest physiology. |
| `DBP` | Diastolic blood pressure. | Numeric / integer | mmHg | 20-200 mmHg for broad clinical validation | Direct measurement | DBP should generally be less than or equal to SBP; flag DBP greater than SBP, extreme values, and likely placeholders. |
| `HR` | Heart rate or pulse rate. | Numeric / integer | Beats per minute (bpm) | 0-250 bpm for broad emergency validation | Direct measurement | HR of 0 may be clinically meaningful during arrest if explicitly recorded; otherwise flag extremely low/high values and device artifacts. |
| `RR` | Respiratory rate. | Numeric / integer | Breaths per minute | 0-80 breaths/min for broad emergency validation | Direct measurement | RR of 0 may be meaningful in apnea/arrest contexts; flag extreme values and distinguish true zero from missing/placeholder values. |
| `BT` | Body temperature. | Numeric | Likely degrees Fahrenheit based on observed values around 94-100; alternatively could be miscoded if Celsius was expected | 86-108 °F if Fahrenheit; 30-42 °C if Celsius | Direct measurement | Confirm unit before analysis; values around 98 strongly suggest Fahrenheit. If Celsius is expected, these values are invalid. |
| `SpO2` | Peripheral oxygen saturation by pulse oximetry. | Numeric / integer | Percent (%) | 0-100% | Direct measurement | Values must not exceed 100%; low values can be clinically plausible but should be checked for sensor artifact or missing-value coding. |
| `GCS` | Glasgow Coma Scale score. | Integer or numeric score | Score points | 3-15 for standard GCS | Direct score if standard GCS | Standard GCS cannot exceed 15; any values above 15 should be treated as invalid, miscoded, or possibly confused with another variable such as oxygen saturation. |
| `TriageScore` | Emergency triage acuity category or severity score. | Ordinal categorical / integer | Score category | Dataset appears to use 1-3; possible triage systems vary | Ordinal category; direction of severity is not documented | Confirm whether lower or higher scores indicate greater acuity before modeling; validate against allowed categories only. |
| `Na` | Serum sodium concentration. | Numeric | mmol/L or mEq/L | 110-170 mmol/L | Direct lab value | Flag severe outliers and missing labs; sodium may be absent for patients without laboratory results. |
| `K` | Serum potassium concentration. | Numeric | mmol/L or mEq/L | 1.5-9.0 mmol/L | Direct lab value | Flag severe outliers; potassium values are sensitive to hemolysis and timing relative to resuscitation. |
| `Cl` | Serum chloride concentration. | Numeric | mmol/L or mEq/L | 70-140 mmol/L | Direct lab value | Flag severe outliers and missing labs; assess consistency with sodium and bicarbonate/anion-gap variables if later added. |
| `Urea` | Blood urea or blood urea nitrogen measurement, depending on local lab convention. | Numeric | Possibly mg/dL for BUN or mg/dL/mmol/L for urea depending on source | 5-250 mg/dL for BUN-style broad validation; confirm local unit | Direct lab value | The variable name alone does not distinguish urea from BUN; confirm unit and analyte before clinical interpretation. Very high values should be checked but may be plausible in renal failure. |
| `Ceratinine` | Raw column likely intended to mean **Creatinine**, a renal function marker. | Numeric | Likely µmol/L based on observed scale; possibly mg/dL if source uses different convention | 20-1500 µmol/L if µmol/L; 0.2-17 mg/dL if mg/dL | Direct lab value; raw name intentionally retained as `Ceratinine` | Document as a likely misspelling of Creatinine. Do not rename the raw column; if creating a cleaned feature, preserve lineage back to `Ceratinine`. Confirm unit before applying clinical cutoffs. |
| `Alcoholic` | History of alcohol use disorder, alcohol use, or alcohol-related status. | Binary categorical / integer | Not applicable | 0 or 1 | Assumed `0 = no`, `1 = yes` | Confirm definition: current use, history, misuse, or clinician-assigned risk may differ. Validate missing and non-binary values. |
| `Smoke` | Smoking status or history. | Binary categorical / integer | Not applicable | 0 or 1 | Assumed `0 = no`, `1 = yes` | Confirm whether this means current smoking, ever smoking, or tobacco exposure. Validate missing and non-binary values. |
| `FHCD` | Family history of cardiac disease or cardiovascular disease. | Binary categorical / integer | Not applicable | 0 or 1 | Assumed `0 = no`, `1 = yes`; acronym likely means family history of cardiac disease | Confirm acronym meaning with data source. Validate missing and non-binary values. |
| `Outcome` | Target outcome label, likely cardiac arrest outcome, event status, survival status, or mortality depending on study definition. | Binary categorical / integer | Not applicable | 0 or 1 | Assumed binary class label; positive-class meaning is not documented | Confirm whether `1` means arrest, death, survival, or another event before training/evaluation. Check class balance and avoid using post-outcome variables as predictors. |

## Dataset-specific validation priorities

1. Confirm the source-code mappings for `Gender`, `TriageScore`, and `Outcome` before clinical reporting or model interpretation.
2. Retain the raw column name `Ceratinine` in raw ingestion and document any cleaned alias such as `Creatinine` in downstream feature engineering.
3. Validate physiologic consistency among vital signs, especially `DBP <= SBP`, `SpO2 <= 100`, standard `GCS` in the range 3-15, and plausible units for `BT`.
4. Treat laboratory variables (`Na`, `K`, `Cl`, `Urea`, `Ceratinine`) as potentially missing-not-at-random because labs may only be collected for selected patients.
