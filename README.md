# Cardiac Arrest Risk Modeling

[![CI](https://github.com/sj-jain-systems/cardiac-arrest-risk-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/sj-jain-systems/cardiac-arrest-risk-modeling/actions/workflows/ci.yml)

## Project overview

This repository holds a complete, reproducible workflow for a tabular cardiac arrest risk dataset. It walks through exploratory analysis, statistical modeling, predictive modeling, final evaluation, interpretability, and subgroup robustness checks. Think of it as a retrospective data science study and a demonstration of a clinically aware machine learning workflow. It is not built for direct clinical use.

A few ideas run through the whole project.

- The raw source data stays untouched.
- Derived datasets live in `data/processed/`.
- Reusable preprocessing and feature logic lives in `src/`.
- Tabular outputs go to `reports/`.
- Model artifacts go to `models/`.
- The clinical, ethical, and validation limits are written down before anyone leans on the results.

## Dataset

| File | Description |
| --- | --- |
| `data/CardiacPatientData.csv` | Raw patient-level clinical observations. Includes demographics, vital signs, laboratory values, lifestyle and history indicators, a triage score, and the outcome label. Treat this file as read-only. |
| `data/processed/cardiac_patient_processed.csv` | Derived dataset with deterministic engineered clinical features built from the raw CSV. Regenerate this file rather than editing it by hand. |

Here is what the raw dataset covers.

- Identifiers and demographics. `ID`, `Age`, `Gender`
- Vital signs. `SBP`, `DBP`, `HR`, `RR`, `BT`, `SpO2`
- Neurological and triage features. `GCS`, `TriageScore`
- Laboratory values. `Na`, `K`, `Cl`, `Urea`, `Ceratinine`
- Lifestyle and family history. `Alcoholic`, `Smoke`, `FHCD`
- Target. `Outcome`

`Ceratinine` looks like a misspelling of `Creatinine`. Leave the raw column name as it is in `data/CardiacPatientData.csv`, and only normalize or explain it in derived analysis outputs.

### Data handling rules

- Do not edit, overwrite, rename, or clean `data/CardiacPatientData.csv` in place.
- Put cleaned, transformed, or feature-engineered datasets under `data/processed/`.
- Keep generated report tables under `reports/` and generated figures under `reports/figures/`.
- Keep generated model binaries under `models/`. The `models/*.joblib` files are ignored by Git and should be regenerated as needed.

### Data validation

The raw CSV has a schema contract in `src/validation.py`, written with [pandera](https://pandera.readthedocs.io/). It spells out the expected columns, their dtypes, the allowed category codes, and broad clinical plausibility ranges. The point is that bad data fails loudly, with a clear column-level error, instead of turning into a confusing traceback deep inside a notebook or a quietly wrong model. The ranges are wide screening bounds meant to catch data mistakes like bad units, swapped columns, or impossible values, not diagnostic cutoffs. Laboratory and vital columns can be null, because real missingness happens and gets handled later by leakage-safe imputation.

```python
from src.preprocessing import load_raw_dataset
from src.validation import validate_raw_dataframe

df = validate_raw_dataframe(load_raw_dataset())          # raises on any breach
# For unlabelled records meant for inference (no Outcome column):
validate_raw_dataframe(new_records, require_outcome=False)
```

Validation runs lazily by default, so it gathers every violation and reports them together instead of stopping at the first one. It needs `pandera`, which is declared in the project dependencies.

## Repository structure

```text
Cardiac-Arrest-Risk-Modeling/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml            # source of truth for dependencies and tooling config
├── requirements.txt          # pinned mirror of pyproject dependencies
├── .pre-commit-config.yaml   # ruff, black, and nbstripout hooks
├── .github/
│   └── workflows/
│       └── ci.yml            # lint, test, reproduce dataset, execute notebooks
├── data/
│   ├── CardiacPatientData.csv
│   └── processed/            # generated, rebuilt by src/create_processed_dataset.py
├── models/
│   └── best_model_metadata.json   # best_model.joblib is git-ignored, regenerate it
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
│   ├── model_card.md
│   ├── baseline_logistic_regression_metrics.csv
│   ├── data_quality_summary.csv
│   ├── final_model_metrics.csv
│   ├── model_comparison.csv
│   ├── monotonic_constraint_directions.csv
│   ├── odds_ratio_results.csv
│   ├── patient_level_robustness.csv
│   ├── robustness_checks.csv
│   ├── subgroup_missingness_tests.csv
│   ├── subgroup_performance.csv
│   ├── threshold_analysis.csv
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── config.py             # path constants anchored to the project root
│   ├── preprocessing.py      # loading, target split, leakage-safe splitting
│   ├── validation.py         # pandera schema contract for the raw CSV
│   ├── features.py           # deterministic clinical feature engineering
│   ├── modeling.py           # pipelines, CV, tuning, monotonic constraints
│   ├── evaluation.py         # metrics, patient-level bootstrap CIs, plots
│   ├── predict.py            # inference layer and CLI for scoring new records
│   ├── create_processed_dataset.py   # CLI entry point to build the processed CSV
│   └── rag/                  # optional retrieval-augmented generation layer
│       ├── __init__.py
│       ├── config.py
│       ├── loaders.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── vector_store.py
│       ├── retriever.py
│       ├── generator.py
│       ├── pipeline.py
│       ├── cli.py
│       └── test_rag_pipeline.py       # offline RAG tests (stub embedder and mocked client)
└── tests/
    ├── test_data_pipeline.py
    ├── test_modeling.py
    └── test_evaluation.py
```

## Setup instructions

1. Clone the repository and move into the project directory.

   ```bash
   git clone <repository-url>
   cd Cardiac-Arrest-Risk-Modeling
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use this instead.

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies. Versions are pinned for reproducibility, and `pyproject.toml` is the source of truth.

   The recommended path installs the project as an editable package along with the development tooling (`pytest`, `ruff`, `black`, `pre-commit`, `nbstripout`).

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

   If you only want the runtime analysis stack with no dev tooling, use this.

   ```bash
   python -m pip install -e .
   ```

   The same pinned set is also available through `requirements.txt`.

   ```bash
   python -m pip install -r requirements.txt
   ```

4. If you like, turn on the pre-commit hooks so notebooks get committed output-free and code stays formatted.

   ```bash
   pre-commit install
   ```

5. Start Jupyter if you plan to run notebooks interactively.

   ```bash
   jupyter notebook
   ```

## How to run the analysis

### The easy path, reproduce everything with make

From a clean checkout, one command from the repository root reproduces the whole thing.

```bash
make all
```

This runs, in order, `install` (editable package and pinned requirements), `data` (build the processed dataset), `lint` (ruff and black checks), `test` (pytest), and `notebooks` (execute `01` through `08` in order, in place). The individual stages work on their own too, so `make data`, `make lint`, `make test`, and `make notebooks` are all fair game. `make clean` clears out generated datasets, model artifacts, figures, and notebook checkpoints. Run `make` with no target, or read the comment block at the top of the `Makefile`, for the full list.

### The manual path

From the repository root, rebuild the processed dataset first.

```bash
python src/create_processed_dataset.py
```

Then run the notebooks in numeric order, from `notebooks/01_data_quality_assessment.ipynb` through `notebooks/08_bias_fairness_robustness.ipynb`. That order makes sure the dataset checks, exploratory outputs, baseline metrics, model comparison results, the saved model artifact, final evaluation, interpretability outputs, and robustness tables all come out consistent with one another.

You can run the notebooks interactively in Jupyter, or from the command line with a runner like `nbconvert`.

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_quality_assessment.ipynb --inplace
```

Run that command once per notebook, following the order in the table below.

## Notebook order

| Order | Notebook | Purpose | Key outputs |
| --- | --- | --- | --- |
| 1 | `01_data_quality_assessment.ipynb` | Checks shape, data types, missingness, duplicates, repeated IDs, clinical ranges, outliers, and raw data integrity. | Data-quality summaries and figures under `reports/` and `reports/figures/`. |
| 2 | `02_exploratory_data_analysis.ipynb` | Summarizes cohort characteristics, outcome balance, stratified summaries, distributions, group outcome rates, and correlations. | EDA figures under `reports/figures/`. |
| 3 | `03_baseline_logistic_regression.ipynb` | Builds a leakage-aware baseline logistic regression pipeline. | `reports/baseline_logistic_regression_metrics.csv` and baseline plots. |
| 4 | `04_statistical_analysis_odds_ratios.ipynb` | Estimates unadjusted and adjusted odds ratios for interpretable statistical analysis. | `reports/odds_ratio_results.csv`. |
| 5 | `05_predictive_modeling.ipynb` | Trains and compares candidate machine learning models with leakage-safe splitting and cross-validation. | `reports/model_comparison.csv` and `models/best_model.joblib`. |
| 6 | `06_final_model_evaluation.ipynb` | Evaluates the saved final model on the held-out test set, covering threshold analysis, discrimination, calibration, and confusion matrices. | `reports/final_model_metrics.csv`, `reports/threshold_analysis.csv`, and final model figures. |
| 7 | `07_model_interpretability.ipynb` | Explains the final model with coefficients where applicable, permutation importance, and SHAP where compatible. | Interpretability figures under `reports/figures/`. |
| 8 | `08_bias_fairness_robustness.ipynb` | Evaluates subgroup performance, missingness patterns, and robustness to modeling and preprocessing alternatives. | `reports/subgroup_performance.csv`, `reports/subgroup_missingness_tests.csv`, and `reports/robustness_checks.csv`. |

## Headline results

Here is the final held-out performance of the saved model, with 95% confidence intervals from a patient-level bootstrap (1,000 resamples of unique patient IDs, not rows). Sensitivity and specificity are reported at the recommended operating threshold of 0.45. The point estimates and intervals come out of `notebooks/06_final_model_evaluation.ipynb` and land in `reports/final_model_metrics.csv`.

| Metric | Estimate (95% CI) |
| --- | --- |
| AUROC | 0.70 (0.20 to 0.98) |
| AUPRC | 0.94 (0.79 to 1.00) |
| Sensitivity at 0.45 | 0.90 (0.81 to 0.99) |
| Specificity at 0.45 | 0.56 (0.00 to 0.97) |
| Brier score | 0.13 (0.03 to 0.26) |

The held-out test set has 1,080 rows but only 22 unique patients, which is why the intervals are so wide. That is on purpose. Because one patient can show up in hundreds of rows, the bootstrap resamples at the patient level rather than the row level. Resampling by row would treat correlated repeated measures as if they were independent, inflate the effective sample size, and make the model look far more certain than it is. The wide intervals are an honest picture of how few independent patients these estimates actually rest on.

## Prediction and inference

Once the model artifact exists at `models/best_model.joblib`, which `notebooks/05_predictive_modeling.ipynb` produces, `src/predict.py` can score new patient records. Inference goes through the same `src.modeling.build_feature_matrix` function used during training, so the engineered clinical features, the dropped `ID` column, and the categorical casting all match between training and scoring. There is no separate scoring path that could quietly drift away from how the model was fit. The high-risk decision threshold comes from `reports/final_model_metrics.csv` when that file is present, and falls back to the documented default of 0.45.

### Command line

```bash
# Single record from a JSON file, a {column: value} object. Use '-' to read stdin.
python -m src.predict --input patient.json

# Batch scoring, a CSV of records in and a CSV of scores out
python -m src.predict --csv new_patients.csv --out scored.csv

# Override the model artifact or the operating threshold
python -m src.predict --input patient.json --threshold 0.5
```

A single-record call prints JSON with `risk_score` (the calibrated probability of the positive outcome), the `threshold` it used, and the `high_risk` decision. Batch scoring writes those same columns, plus `ID` when you supply it, as CSV.

### From Python

```python
from src.predict import predict_record

predict_record(
    {
        "SBP": 120, "DBP": 80, "HR": 88, "RR": 18, "BT": 98, "SpO2": 97,
        "Age": 66, "Gender": 1, "GCS": 15, "Na": 139, "K": 4.0, "Cl": 105,
        "Urea": 41, "Ceratinine": 91, "Alcoholic": 1, "Smoke": 1,
        "FHCD": 0, "TriageScore": 3,
    }
)
# {'risk_score': 0.83..., 'threshold': 0.45, 'high_risk': True}
```

`predict_dataframe(df)` scores a whole dataframe at once and can take a model you already loaded, so repeated batches do not reload the artifact every time.

A quick reminder before anyone reads too much into a score. These come from a retrospective model that has not been externally validated and was trained on very few unique patients, as the headline results make clear. They are a demonstration, not medical advice and not a decision-support tool. Everything in the limitations section applies here too.

## Generated reports

The repository ships a set of generated tables and written documentation.

| File | Description |
| --- | --- |
| `reports/data_dictionary.md` | Variable definitions, assumed encodings, clinical context, and data-quality notes. |
| `reports/final_report.md` | Final narrative report covering methods, results, interpretation, limitations, and next steps. |
| `reports/model_card.md` | Model card covering intended use, data, metrics, and limitations. |
| `reports/baseline_logistic_regression_metrics.csv` | Held-out performance metrics for the baseline logistic regression model. |
| `reports/data_quality_summary.csv` | Data-quality summary produced by the data-quality assessment notebook. |
| `reports/odds_ratio_results.csv` | Unadjusted and adjusted odds-ratio results. |
| `reports/model_comparison.csv` | Candidate model comparison results from predictive modeling. |
| `reports/monotonic_constraint_directions.csv` | Odds-ratio-derived monotonic constraint directions used by the LightGBM candidate. |
| `reports/final_model_metrics.csv` | Final model metrics with patient-level bootstrap 95% CIs (for example `auroc`, `auroc_lower`, `auroc_upper`), the selected operating threshold, and calibration and discrimination summaries. |
| `reports/threshold_analysis.csv` | Sensitivity, specificity, predictive values, and confusion matrix counts across thresholds. |
| `reports/subgroup_performance.csv` | Model performance by subgroup. |
| `reports/subgroup_missingness_tests.csv` | Missingness comparisons by subgroup. |
| `reports/patient_level_robustness.csv` | Patient-level robustness scenarios. |
| `reports/robustness_checks.csv` | Robustness scenarios and performance comparisons. |

Generated figures land in `reports/figures/`. Several figure patterns are ignored by Git since they can be rebuilt from the notebooks.

## Model artifacts

The main model artifacts are these two.

| Artifact | Created by | Description |
| --- | --- | --- |
| `models/best_model.joblib` | `notebooks/05_predictive_modeling.ipynb` | Serialized scikit-learn pipeline for the best-performing candidate model. |
| `models/best_model_metadata.json` | `notebooks/05_predictive_modeling.ipynb` | Metadata describing the selected model. This one is committed. The `.joblib` binary is git-ignored. |

The `models/*.joblib` files are ignored by Git on purpose. Rebuild the final model artifact by running `notebooks/05_predictive_modeling.ipynb`. The evaluation and interpretability notebooks, along with the `src/predict.py` inference layer, load this artifact whenever it is available.

## RAG and retrieval

The project ships an optional retrieval-augmented generation layer in `src/rag/` that lets you ask questions about the generated reports in plain language. It indexes the Markdown and CSV artifacts under `reports/` and answers questions like *"What is the sensitivity at threshold 0.45?"* or *"How is creatinine abnormality defined?"* with grounded, source-cited answers. It stays completely separate from the analysis pipeline. None of the notebooks or `src/` analysis modules import it, and its dependencies sit in their own optional group.

- Retrieval uses local [sentence-transformers](https://www.sbert.net/) embeddings (`all-MiniLM-L6-v2`) and a lightweight NumPy cosine-similarity index saved under `models/rag_index/`, which is git-ignored. This path is fully offline and needs no API key.
- Generation sends the retrieved excerpts to the Claude API (`claude-opus-4-8`) with a grounding prompt that tells the model to answer only from the provided reports, cite sources by filename, and say *"I don't know based on the reports."* when the context does not cover the question. This path needs `ANTHROPIC_API_KEY`.

### Install

The RAG dependencies are an optional extra, so the core analysis install stays small.

```bash
python -m pip install -e ".[rag]"
```

### Build the index

```bash
python -m src.rag.cli build          # or run make rag-index
```

This downloads the embedding model once and writes `models/rag_index/embeddings.npz` and `models/rag_index/chunks.json`.

### Query

```bash
# Retrieval only, no API key needed. Prints the top-k excerpts with scores.
python -m src.rag.cli query "What is the sensitivity at threshold 0.45?" --retrieve-only

# Full RAG, needs ANTHROPIC_API_KEY. Prints a grounded, cited answer.
export ANTHROPIC_API_KEY=...
python -m src.rag.cli query "How is creatinine abnormality defined?"
# or run make rag-query Q="How is creatinine abnormality defined?"
```

Use `--top-k N` to change how many excerpts get retrieved. The default is 5.

One clinical and ethical note here. RAG answers are grounded summaries of this project's existing reports, not medical advice and not a decision-support tool. The same limits below apply, since an answer can only be as reliable as the analysis underneath it, and only the aggregate reports are indexed. The raw patient CSV is never part of the index.

## License

This project is free to use, copy, modify, and distribute, as long as you give appropriate credit to the original author (Shreyans Jain / SJ-Jain-Systems). See the [LICENSE](LICENSE) file for details.

## Important clinical and ethical limitations

- This project is for retrospective analysis and model-development demonstration only. It is not a validated medical device or a bedside decision-support tool.
- The dataset provenance, collection setting, inclusion and exclusion criteria, outcome definition, measurement timing, and label quality all need to be confirmed before anyone reads the results clinically.
- Repeated `ID` values may stand for repeated encounters or measurements, so splitting has to stay leakage-safe at the patient or group level whenever IDs repeat.
- Associations and feature importances are not causal effects and should not be read as treatment recommendations.
- Model performance can vary across demographic and clinical subgroups. The subgroup, fairness, and missingness analyses are necessary but not enough on their own for safe deployment.
- External validation on independent data is required before any operational use.
- The `src/predict.py` inference layer is a convenience wrapper for scoring, not a sanctioned clinical decision-support interface. The same limits apply to every score it produces.
- Threshold choice has to rest on explicit clinical tradeoffs, workflow capacity, risk tolerance, and prospective validation.
- Sensitive health data has to be handled under the applicable privacy, security, governance, and institutional review rules.

## Future enhancements

- Add external and temporal validation datasets.
- Fill in metadata with confirmed data provenance, cohort criteria, units, measurement timing, and outcome definitions.
- Compare model performance against established clinical scores or triage rules where it makes sense.
- Bring in time-aware or survival modeling if timestamped longitudinal data ever becomes available.
- Add nested cross-validation and a learning-curve or power analysis to show how many patients it would take for the confidence intervals to be genuinely usable.
- Add experiment tracking and a model registry so each run logs its parameters, metrics, and artifact hash instead of overwriting the last one.
- Wrap `src/predict.py` in a small web service such as FastAPI if online scoring is ever needed, behind the right governance.
