"""Project-wide path configuration for cardiac arrest risk modeling.

Raw clinical data is treated as read-only. Use PROCESSED_DATA_DIR for any
cleaned, transformed, or derived datasets instead of writing over RAW_DATA_PATH.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "CardiacPatientData.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
