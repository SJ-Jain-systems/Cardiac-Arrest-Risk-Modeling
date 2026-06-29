# =============================================================================
# Cardiac Arrest Risk Modeling - reproducible analysis pipeline
#
# Usage:
#   make all        Reproduce everything: install -> data -> lint -> test -> notebooks
#   make install    Install the package (editable) and pinned requirements
#   make data       Build data/processed/cardiac_patient_processed.csv
#   make lint       Run ruff and black (check-only) on src/ and tests/
#   make test       Run the pytest suite
#   make notebooks  Execute notebooks 01-08 in order (in place)
#   make clean      Remove generated data, models, figures, and checkpoints
#
# Optional RAG layer (requires `pip install -e ".[rag]"`; kept out of `all`):
#   make rag-index            Build the retrieval index from reports/
#   make rag-query Q="..."     Answer a question (generation needs ANTHROPIC_API_KEY)
#
# Override the interpreter if needed:  make all PYTHON=python3.11
# =============================================================================

PYTHON ?= python
PIP    := $(PYTHON) -m pip

PROCESSED_CSV := data/processed/cardiac_patient_processed.csv
NB_DIR        := notebooks
NBEXEC        := jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200

.PHONY: all install data lint test notebooks clean \
        nb01 nb02 nb03 nb04 nb05 nb06 nb07 nb08 \
        rag-index rag-query

# Full reproduction from a clean checkout, in strict order.
all:
	$(MAKE) install
	$(MAKE) data
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) notebooks

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) src/create_processed_dataset.py
	@echo "Wrote $(PROCESSED_CSV)"

lint:
	ruff check src/ tests/
	black --check src/ tests/

test:
	pytest tests/ -v

# Execute notebooks in order; each depends on the previous completing.
notebooks: nb08

nb01:
	$(NBEXEC) $(NB_DIR)/01_data_quality_assessment.ipynb
nb02: nb01
	$(NBEXEC) $(NB_DIR)/02_exploratory_data_analysis.ipynb
nb03: nb02
	$(NBEXEC) $(NB_DIR)/03_baseline_logistic_regression.ipynb
nb04: nb03
	$(NBEXEC) $(NB_DIR)/04_statistical_analysis_odds_ratios.ipynb
nb05: nb04
	$(NBEXEC) $(NB_DIR)/05_predictive_modeling.ipynb
nb06: nb05
	$(NBEXEC) $(NB_DIR)/06_final_model_evaluation.ipynb
nb07: nb06
	$(NBEXEC) $(NB_DIR)/07_model_interpretability.ipynb
nb08: nb07
	$(NBEXEC) $(NB_DIR)/08_bias_fairness_robustness.ipynb

# Optional RAG layer. Kept out of `all` so reproduction stays offline and
# key-free. Building the index needs the `rag` extra; querying with generation
# additionally needs ANTHROPIC_API_KEY (use --retrieve-only to skip the LLM).
rag-index:
	$(PYTHON) -m src.rag.cli build

rag-query:
	$(PYTHON) -m src.rag.cli query "$(Q)"

clean:
	rm -rf data/processed/
	rm -f models/*.joblib
	rm -rf reports/figures/*
	find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} +
