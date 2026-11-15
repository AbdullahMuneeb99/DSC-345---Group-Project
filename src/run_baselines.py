"""Reproducible baseline contribution for DSC 345. Run from any directory."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "CO2(tCO2)"
NUMERIC = ["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh",
           "Leading_Current_Reactive_Power_kVarh", "Lagging_Current_Power_Factor",
           "Leading_Current_Power_Factor", "NSM"]
CATEGORICAL = ["Day_of_week", "Load_Type"]
FEATURES = NUMERIC + CATEGORICAL
LABELS = {"mean_reference": "Training mean", "proportional_baseline": "Proportional baseline",
          "linear_usage": "Usage-only linear", "linear_expanded": "Expanded linear"}


def load_data(path):
    raw = pd.read_csv(path)
    required = ["date", TARGET, "WeekStatus"] + FEATURES
    if set(raw.columns) != set(required):
        raise ValueError(f"Expected the original 11-column UCI CSV; got {list(raw.columns)}")
    if raw.isna().any().any():
        raise ValueError("Missing values found. Agree on treatment before comparing models.")
    if raw.duplicated().any():
        raise ValueError("Duplicate rows found. Agree on treatment before splitting.")
    raw.insert(0, "source_row", np.arange(len(raw)))
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M", errors="raise")
    if raw["date"].duplicated().any():
        raise ValueError("Repeated timestamps found; review before a chronological split.")
    if not np.isfinite(raw[NUMERIC + [TARGET]].to_numpy()).all():
        raise ValueError("Non-finite numeric observations found.")
    # Use the literal timestamps. The original CSV puts 00:00 after 23:45
    # on each date; do not silently shift dates or retain that file ordering.
    return raw.sort_values("date", kind="stable").reset_index(drop=True)


def model_specs():
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore",
                                       sparse_output=False), CATEGORICAL),
    ])
    return {
        "mean_reference": (["Usage_kWh"], DummyRegressor(strategy="mean")),
        # This coefficient is calibrated ONLY on training data. It is not an
        # externally verified grid emission factor.
        "proportional_baseline": (["Usage_kWh"], LinearRegression(fit_intercept=False)),
        "linear_usage": (["Usage_kWh"], LinearRegression(fit_intercept=True)),
        "linear_expanded": (FEATURES, Pipeline([
            ("preprocessing", preprocessing),
            ("regression", LinearRegression(fit_intercept=True)),
        ])),
    }


def scores(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def main(data_path=None, output_dir=None):
    path = Path(data_path) if data_path else ROOT / "data/Steel_industry_data.csv"
    out = Path(output_dir) if output_dir else ROOT
    result_dir, figure_dir = out / "results", out / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    cut = len(df) * 7 // 10
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    assert train.date.max() < test.date.min()
    assert not set(train.source_row) & set(test.source_row)
    assert len(train) + len(test) == len(df)
    split = df[["source_row", "date"]].copy()
    split["split"] = np.where(np.arange(len(df)) < cut, "train", "test")
    split.to_csv(result_dir / "split_manifest.csv", index=False)
    metrics, cv_rows, fitted = [], [], {}
    predictions = test[["source_row", "date", TARGET]].rename(columns={TARGET: "actual"}).copy()
    # Three expanding chronological validation folds, entirely inside the 70%.
    # No model or hyperparameter is chosen using the final test labels.
    cv = TimeSeriesSplit(n_splits=3)
    for name, (features, estimator) in model_specs().items():
        for fold, (fit_idx, val_idx) in enumerate(cv.split(train), start=1):
            fit, val = train.iloc[fit_idx], train.iloc[val_idx]
            assert fit.date.max() < val.date.min() < test.date.min()
            model = clone(estimator).fit(fit[features], fit[TARGET])
            cv_rows.append({"model": name, "fold": fold, "n_train": len(fit),
                            "n_validation": len(val), "train_end": str(fit.date.max()),
                            "validation_start": str(val.date.min()),
                            "validation_end": str(val.date.max()),
                            **scores(val[TARGET], model.predict(val[features]))})
        model = clone(estimator).fit(train[features], train[TARGET])
        fitted[name] = model
        for partition, frame in [("train", train), ("test", test)]:
            prediction = model.predict(frame[features])
            metrics.append({"model": name, "partition": partition, "n": len(frame),
                            **scores(frame[TARGET], prediction),
                            "negative_predictions": int((prediction < 0).sum())})
            if partition == "test":
                predictions[name] = prediction
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(result_dir / "metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(result_dir / "chronological_validation.csv", index=False)
    predictions.to_csv(result_dir / "test_predictions.csv", index=False)
    df[NUMERIC + [TARGET]].describe().to_csv(result_dir / "descriptive_statistics.csv")
    counts = df[TARGET].value_counts().sort_index().rename_axis(TARGET).reset_index(name="count")
    counts.to_csv(result_dir / "target_distribution.csv", index=False)
    # Flag unusual values using TRAINING quartiles; keep them, since high loads
    # can be real observations and dropping them would change the comparison.
    q1, q3 = train[TARGET].quantile([.25, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flagged = df.loc[(df[TARGET] < low) | (df[TARGET] > high), ["source_row", "date", "Usage_kWh", TARGET]]
    flagged.to_csv(result_dir / "flagged_target_rows_retained.csv", index=False)
    factor = float(fitted["proportional_baseline"].coef_[0])
    slope = float(fitted["linear_usage"].coef_[0])
    intercept = float(fitted["linear_usage"].intercept_)
    expanded = fitted["linear_expanded"]
    coeff = pd.DataFrame({"transformed_feature": expanded["preprocessing"].get_feature_names_out(),
                          "coefficient": expanded["regression"].coef_})
    coeff.to_csv(result_dir / "expanded_coefficients.csv", index=False)
    audit = {
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": len(df), "original_columns": 11, "missing_values": 0,
        "duplicate_rows": 0, "rows_removed": 0, "n_train": len(train), "n_test": len(test),
        "train_start": str(train.date.min()), "train_end": str(train.date.max()),
        "test_start": str(test.date.min()), "test_end": str(test.date.max()),
        "timestamp_policy": "Parse literally day/month/year; stable chronological sort; no day shifts.",
        "target": TARGET, "target_units": "Native dataset values; metadata conflicts between tCO2 name and ppm description.",
        "target_unique_values": counts[TARGET].tolist(), "target_zero_rows": int((df[TARGET] == 0).sum()),
        "target_negative_rows": int((df[TARGET] < 0).sum()),
        "flagged_target_rows_retained": len(flagged), "target_iqr_thresholds_training": [float(low), float(high)],
        "expanded_features": FEATURES, "excluded": {"date": "Split only", TARGET: "Target",
            "WeekStatus": "Redundant with Day_of_week", "source_row": "Alignment key only"},
        "proportional_factor": factor, "usage_slope": slope, "usage_intercept": intercept,
        "expanded_intercept": float(expanded["regression"].intercept_),
        "validation": "TimeSeriesSplit(n_splits=3, gap=0) within training only; no tuning; final 30% unchanged.",
        "prediction_policy": "Raw predictions; no clipping or rounding.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (result_dir / "run_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    names = ["proportional_baseline", "linear_usage", "linear_expanded"]
    test_metrics = metric_df[metric_df.partition == "test"].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        values = test_metrics.loc[names, metric]
        bars = ax.bar(["Proportional", "Usage linear", "Expanded linear"], values, color=["#929292", "#505050", "#171717"])
        ax.bar_label(bars, fmt="%.6f", padding=4, fontsize=10)
        ax.set_ylim(0, values.max() * 1.25)
        ax.set_ylabel(f"{metric} (native CO2 target units)")
        ax.set_title(f"Test {metric}: lower is better")
    fig.savefig(figure_dir / "test_errors.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for ax, name in zip(axes, names):
        ax.scatter(predictions.actual, predictions[name], s=7, alpha=.12, color="black", rasterized=True)
        lo = min(0., float(predictions[name].min())); hi = max(predictions.actual.max(), predictions[name].max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")
        ax.set(xlabel="Actual CO2 target", ylabel="Predicted CO2 target", title=LABELS[name], xlim=(lo, hi), ylim=(lo, hi))
    fig.savefig(figure_dir / "actual_vs_predicted.png", dpi=180); plt.close(fig)
    daily = predictions.set_index("date")[["actual"] + names].resample("D").mean()
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.plot(daily.index, daily.actual, label="Actual", color="black", linewidth=2)
    ax.plot(daily.index, daily.linear_usage, label="Usage-only linear", color="#888888", linestyle="--")
    ax.plot(daily.index, daily.linear_expanded, label="Expanded linear", color="#444444", linestyle=":")
    ax.set(title="Daily mean predictions during the test period", ylabel="Native CO2 target units", xlabel="2018 date")
    ax.legend(frameon=False); fig.savefig(figure_dir / "test_daily_means.png", dpi=180); plt.close(fig)
    print(metric_df[metric_df.partition == "test"].to_string(index=False))
    print(f"Training-calibrated proportional factor: {factor:.10f}")
    print(f"Usage-only formula: {slope:.10f} * Usage_kWh + ({intercept:.10f})")
    return audit, metric_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/Steel_industry_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.data, args.output)
