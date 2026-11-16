"""Shared dataset validation, feature definitions, and chronological splitting."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
import sklearn

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Steel_industry_data.csv"
TARGET = "CO2(tCO2)"
NUMERIC = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]
CATEGORICAL = ["Day_of_week", "Load_Type"]
FEATURES = NUMERIC + CATEGORICAL
USAGE_ONLY = ["Usage_kWh"]


def load_data(path=DATA_PATH):
    """Read and validate the original CSV, then sort by actual timestamps."""
    raw = pd.read_csv(path)
    expected = {"date", TARGET, "WeekStatus", *FEATURES}
    if set(raw.columns) != expected:
        raise ValueError(f"Unexpected CSV columns: {list(raw.columns)}")
    if raw.isna().any().any() or raw.duplicated().any():
        raise ValueError("Missing values or duplicate rows: investigate before modeling.")
    raw.insert(0, "source_row", np.arange(len(raw)))
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M", errors="raise")
    if raw["date"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")
    if not np.isfinite(raw[NUMERIC + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in numeric columns.")
    return raw.sort_values("date", kind="stable").reset_index(drop=True)


def split_data(df):
    """The first 70% of chronological observations train; last 30% test."""
    cut = len(df) * 7 // 10
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    assert train["date"].max() < test["date"].min()
    assert set(train["source_row"]).isdisjoint(test["source_row"])
    return train, test


def split_manifest(df):
    cut = len(df) * 7 // 10
    manifest = df[["source_row", "date"]].copy()
    manifest["split"] = np.where(np.arange(len(df)) < cut, "train", "test")
    return manifest


def get_splits(path=DATA_PATH, result_dir=None):
    """Require the shared split manifest so all model scripts use identical rows."""
    result_dir = Path(result_dir) if result_dir else ROOT / "results"
    df = load_data(path)
    manifest_path = result_dir / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run 'python src/data_preprocessing.py' first.")
    expected = split_manifest(df)
    saved = pd.read_csv(manifest_path, parse_dates=["date"])
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    return split_data(df)


def make_preprocessor(scale_numeric=True):
    """Fit only inside each model's training pipeline (never on test rows)."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("numeric", numeric_step, NUMERIC),
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore",
                                       sparse_output=False), CATEGORICAL),
    ])


def main(data_path=DATA_PATH, output_root=ROOT):
    output_root = Path(output_root)
    result_dir = output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    train, test = split_data(df)
    split_manifest(df).to_csv(result_dir / "split_manifest.csv", index=False)

    # Keep the original IQR-based audit artifact, without deleting any rows.
    q1, q3 = train[TARGET].quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flagged = df.loc[(df[TARGET] < low) | (df[TARGET] > high),
                     ["source_row", "date", "Usage_kWh", TARGET]]
    flagged.to_csv(result_dir / "flagged_target_rows_retained.csv", index=False)
    summary = {
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "n_rows": len(df), "original_columns": 11,
        "n_train": len(train), "n_test": len(test),
        "train_start": str(train.date.min()), "train_end": str(train.date.max()),
        "test_start": str(test.date.min()), "test_end": str(test.date.max()),
        "rows_removed": 0, "missing_values": 0, "duplicate_rows": 0,
        "expanded_features": FEATURES, "target": TARGET,
        "target_units": "Native dataset values; source metadata conflicts with column label.",
        "target_unique_values": sorted(df[TARGET].unique().tolist()),
        "target_zero_rows": int((df[TARGET] == 0).sum()),
        "target_negative_rows": int((df[TARGET] < 0).sum()),
        "excluded": {"date": "Timestamp sorting", "source_row": "Row alignment",
                     "WeekStatus": "Redundant with Day_of_week", TARGET: "Target"},
        "flagged_target_rows_retained": int(len(flagged)),
        "target_iqr_thresholds_training": [float(low), float(high)],
        "timestamp_policy": "Parse original timestamps literally and sort chronologically.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (result_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Split saved: {len(train):,} train / {len(test):,} test observations")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.data, args.output)
