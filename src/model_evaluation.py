
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from data_preprocessing import ROOT, TARGET


def scores(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def fit_and_evaluate(name, estimator, features, train, test):
    """3 training-only time folds, then final fit and train/test metrics."""
    validation = []
    cv = TimeSeriesSplit(n_splits=3)
    for fold, (fit_idx, val_idx) in enumerate(cv.split(train), start=1):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        fitted = clone(estimator).fit(fit[features], fit[TARGET])
        validation.append({
            "model": name, "fold": fold, "n_train": len(fit),
            "n_validation": len(val), "train_end": str(fit.date.max()),
            "validation_start": str(val.date.min()),
            "validation_end": str(val.date.max()),
            **scores(val[TARGET], fitted.predict(val[features])),
        })

    fitted = clone(estimator).fit(train[features], train[TARGET])
    metrics = []
    for partition, frame in (("train", train), ("test", test)):
        predicted = fitted.predict(frame[features])
        metrics.append({
            "model": name, "partition": partition, "n": len(frame),
            **scores(frame[TARGET], predicted),
            "negative_predictions": int((predicted < 0).sum()),
        })
    return fitted, metrics, validation, fitted.predict(test[features])


def save_group(group, metrics, validation, predictions, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(result_dir / f"metrics_{group}.csv", index=False)
    pd.DataFrame(validation).to_csv(result_dir / f"validation_{group}.csv", index=False)
    predictions.to_csv(result_dir / f"test_predictions_{group}.csv", index=False)


def combine_results(output_root=ROOT):
    output_root = Path(output_root)
    result_dir = output_root / "results"
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    groups = ["baselines", "random_forest", "gradient_boosting"]
    missing = [g for g in groups if not (result_dir / f"metrics_{g}.csv").exists()
               or not (result_dir / f"test_predictions_{g}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Run model scripts first; missing groups: {missing}")

    metric_tables, cv_tables, prediction_tables = [], [], []
    for group in groups:
        metric_tables.append(pd.read_csv(result_dir / f"metrics_{group}.csv"))
        val_path = result_dir / f"validation_{group}.csv"
        if val_path.exists():
            cv_tables.append(pd.read_csv(val_path))
        prediction_tables.append(pd.read_csv(result_dir / f"test_predictions_{group}.csv"))

    base = prediction_tables[0]
    for other in prediction_tables[1:]:
        # Match by original record ID, not accidentally by row position.
        if not base["source_row"].equals(other["source_row"]):
            raise ValueError("Prediction files do not use the same test rows/order.")
        if not base["date"].equals(other["date"]):
            raise ValueError("Prediction files have different dates.")
        if not np.allclose(base["actual"], other["actual"]):
            raise ValueError("Prediction files disagree about actual CO2 values.")
        extra = other.drop(columns=["source_row", "date", "actual"])
        duplicates = set(base.columns) & set(extra.columns)
        if duplicates:
            raise ValueError(f"Duplicate model columns: {duplicates}")
        base = pd.concat([base, extra], axis=1)

    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics.to_csv(result_dir / "metrics.csv", index=False)
    base.to_csv(result_dir / "test_predictions.csv", index=False)
    if cv_tables:
        pd.concat(cv_tables, ignore_index=True).to_csv(
            result_dir / "chronological_validation.csv", index=False
        )

    test_metrics = metrics.loc[metrics.partition == "test"].copy()
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(test_metrics.model, test_metrics.RMSE)
    ax.set(title="Test RMSE by regression model", ylabel="RMSE (reported CO2 units)")
    ax.tick_params(axis="x", rotation=40)
    fig.savefig(figure_dir / "test_errors.png", dpi=160)
    plt.close(fig)

    # Compare representative models, keeping the original filename.
    names = [n for n in ["proportional_baseline", "linear_usage", "linear_expanded",
                          "random_forest_expanded", "gradient_boosting_expanded"]
             if n in base.columns]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), layout="constrained")
    for ax, name in zip(np.atleast_1d(axes), names):
        ax.scatter(base.actual, base[name], s=5, alpha=0.12)
        upper = max(float(base.actual.max()), float(base[name].max()))
        ax.plot([0, upper], [0, upper], "k--", linewidth=1)
        ax.set(title=name, xlabel="Actual", ylabel="Predicted")
    fig.savefig(figure_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    daily_names = [n for n in ["actual", "linear_usage", "linear_expanded",
                                "random_forest_expanded", "gradient_boosting_expanded"]
                   if n in base.columns]
    daily = base.assign(date=pd.to_datetime(base.date)).set_index("date")[daily_names].resample("D").mean()
    ax = daily.plot(figsize=(11, 5), title="Daily mean CO2 predictions on test observations")
    ax.set_ylabel("Reported CO2 units")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "test_daily_means.png", dpi=160)
    plt.close(ax.figure)
    print(test_metrics[["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    combine_results(args.output)
