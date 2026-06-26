"""Central configuration: paths, dataset constants, and experiment defaults."""
from __future__ import annotations
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DP_DIR = os.path.join(DATA_DIR, "dp_data")
KANON_DIR = os.path.join(DATA_DIR, "k_anon_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ADULT_EMB_RT_CSV = os.path.join(DATA_DIR, "adult", "embeddings_runtimes.csv")

# ── Adult dataset column names ─────────────────────────────────────────────────
ADULT_COLUMNS = [
    "age", "workClass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
ADULT_CAT_COLS = [
    "workClass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]
ADULT_NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ADULT_LABEL = "income"

# CIFAR-10 normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

# ── Experiment defaults ────────────────────────────────────────────────────────
DEFAULT_SEED       = 7
MIA_RESAMPLES      = 10
MIA_EVAL_CAP       = 5000
