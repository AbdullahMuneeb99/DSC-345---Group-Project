"""Train the four original reference/linear models using shared project helpers."""
from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from data_preprocessing import (ROOT, DATA_PATH, TARGET, FEATURES, USAGE_ONLY,
                                get_splits, make_preprocessor)
from model_evaluation import fit_and_evaluate, save_group


def model_specs():
    return {
        "mean_reference": (USAGE_ONLY, DummyRegressor(strategy="mean")),
        "proportional_baseline": (USAGE_ONLY, LinearRegression(fit_intercept=False)),
        "linear_usage": (USAGE_ONLY, LinearRegression(fit_intercept=True)),
        "linear_expanded": (FEATURES, Pipeline([
            ("preprocessing", make_preprocessor(scale_numeric=True)),
            ("regression", LinearRegression(fit_intercept=True)),
        ])),
    }


def main(data_path=DATA_PATH, output_root=ROOT):
    output_root = Path(output_root)
    result_dir = output_root / "results"
    train, test = get_splits(data_path, result_dir)
    predictions = test[["source_row", "date", TARGET]].rename(columns={TARGET: "actual"}).copy()
    all_metrics, all_cv, models = [], [], {}
    for name, (features, estimator) in model_specs().items():
        fitted, metrics, cv, predicted = fit_and_evaluate(name, estimator, features, train, test)
        models[name] = fitted
        all_metrics.extend(metrics)
        all_cv.extend(cv)
        predictions[name] = predicted
    save_group("baselines", all_metrics, all_cv, predictions, result_dir)

    expanded = models["linear_expanded"]
    pd.DataFrame({
        "transformed_feature": expanded["preprocessing"].get_feature_names_out(),
        "coefficient": expanded["regression"].coef_,
    }).to_csv(result_dir / "expanded_coefficients.csv", index=False)

    summary = json.loads((result_dir / "preprocessing_summary.json").read_text(encoding="utf-8"))
    summary.update({
        "proportional_factor": float(models["proportional_baseline"].coef_[0]),
        "usage_slope": float(models["linear_usage"].coef_[0]),
        "usage_intercept": float(models["linear_usage"].intercept_),
        "expanded_intercept": float(expanded["regression"].intercept_),
        "models": list(models),
        "validation": "Three expanding chronological folds inside training; held-out test unchanged.",
        "prediction_policy": "Raw predictions; no clipping or rounding.",
    })
    (result_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(all_metrics).query("partition == 'test'")["model MAE RMSE R2".split()].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.data, args.output)
