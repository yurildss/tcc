from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import textwrap
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

try:
    import wfdb
except Exception:  # pragma: no cover - optional dependency
    wfdb = None

try:
    from kan import KAN
except Exception:  # pragma: no cover - optional dependency
    KAN = None

ROOT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = ROOT_DIR / "dataset_split"
BENCHMARK_OUTPUT = ROOT_DIR / "benchmark_resultados"
ARDUINO_OUTPUT = ROOT_DIR / "ArduinoCode"

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("tcc_project")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def as_list(values: Iterable[Any]) -> List[Any]:
    return list(values)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
