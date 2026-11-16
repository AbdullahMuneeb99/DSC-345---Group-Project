"""Post-hoc permutation importance on a fixed sample of held-out data."""
from joblib import load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from data_preprocessing import ROOT, DATA_PATH, FEATURES, TARGET, get_splits


def main():
    _, test = get_splits(DATA_PATH, ROOT / "results")
    # A fixed sample makes permutation importance less expensive to compute.
    test_sample = test.sample(n=min(2000, len(test)), random_state=123)
    rows = []
    for name in ("random_forest_expanded", "gradient_boosting_expanded"):
        path = ROOT / "results" / "models" / f"{name}.joblib"
        model = load(path)  # Load only our own team's trusted model files.
        importance = permutation_importance(
            model, test_sample[FEATURES], test_sample[TARGET],
            scoring="neg_mean_squared_error", n_repeats=3, random_state=123
        )
        for feature, mean, std in zip(FEATURES, importance.importances_mean,
                                       importance.importances_std):
            rows.append({"model": name, "feature": feature,
                         "importance_mean": mean, "importance_std": std})

    importance_df = pd.DataFrame(rows)
    importance_df.to_csv(ROOT / "results" / "feature_importance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, (name, group) in zip(axes, importance_df.groupby("model", sort=False)):
        group = group.sort_values("importance_mean")
        ax.barh(group.feature, group.importance_mean)
        ax.set(title=name, xlabel="Increase in MSE after feature shuffling")
    fig.savefig(ROOT / "figures" / "feature_importance.png", dpi=160)
    plt.close(fig)
    print("Saved results/feature_importance.csv and figures/feature_importance.png")


if __name__ == "__main__":
    main()
