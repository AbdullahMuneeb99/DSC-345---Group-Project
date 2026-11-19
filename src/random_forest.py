
from joblib import dump
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import ROOT, DATA_PATH, TARGET, FEATURES, USAGE_ONLY, get_splits, make_preprocessor
from model_evaluation import fit_and_evaluate, save_group


def main():
    result_dir = ROOT / "results"
    model_dir = result_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train, test = get_splits(DATA_PATH, result_dir)
    predictions = test[["source_row", "date", TARGET]].rename(columns={TARGET: "actual"}).copy()
    all_metrics, all_cv = [], []

    for suffix, features in (("usage", USAGE_ONLY), ("expanded", FEATURES)):
        name = f"random_forest_{suffix}"
        estimator = RandomForestRegressor(
            n_estimators=100, max_depth=14, random_state=123, n_jobs=-1
        )
        if suffix == "expanded":
            estimator = Pipeline([
                ("preprocessing", make_preprocessor(scale_numeric=False)),
                ("regression", estimator),
            ])
        fitted, metrics, cv, predicted = fit_and_evaluate(name, estimator, features, train, test)
        dump(fitted, model_dir / f"{name}.joblib")
        all_metrics.extend(metrics)
        all_cv.extend(cv)
        predictions[name] = predicted

    save_group("random_forest", all_metrics, all_cv, predictions, result_dir)
    print(pd.DataFrame(all_metrics).query("partition == 'test'")["model MAE RMSE R2".split()].to_string(index=False))


if __name__ == "__main__":
    main()
