

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import ROOT, DATA_PATH, NUMERIC, TARGET, load_data


def main():
    df = load_data(DATA_PATH)
    figure_dir, result_dir = ROOT / "figures", ROOT / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    df[NUMERIC + [TARGET]].describe().to_csv(result_dir / "descriptive_statistics.csv")
    df[TARGET].value_counts().sort_index().rename_axis(TARGET).reset_index(
        name="count"
    ).to_csv(result_dir / "target_distribution.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(df["Usage_kWh"], bins=40)
    axes[0].set(title="Electricity use distribution", xlabel="Usage (kWh)", ylabel="Records")
    axes[1].hist(df[TARGET], bins=15)
    axes[1].set(title="Reported CO2 distribution", xlabel="Reported CO2", ylabel="Records")
    fig.savefig(figure_dir / "distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(df["Usage_kWh"], df[TARGET], s=5, alpha=0.12)
    ax.set(title="Electricity use vs reported CO2", xlabel="Usage (kWh)", ylabel="Reported CO2")
    fig.savefig(figure_dir / "usage_vs_co2.png", dpi=160)
    plt.close(fig)

    corr = df[NUMERIC + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.savefig(figure_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    df["month"] = df.date.dt.month
    monthly = df.groupby("month", as_index=False).agg(
        total_co2=(TARGET, "sum"), total_kwh=("Usage_kWh", "sum")
    )
    monthly.to_csv(result_dir / "monthly_totals.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.bar(monthly.month, monthly.total_co2)
    ax.set(title="Monthly reported CO2 totals in 2018", xlabel="Month", ylabel="Total reported CO2")
    ax.set_xticks(range(1, 13))
    fig.savefig(figure_dir / "monthly_co2.png", dpi=160)
    plt.close(fig)

    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                  5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn",
                  10: "Autumn", 11: "Autumn"}
    df["season"] = df.month.map(season_map)
    seasonal = df.groupby("season", as_index=False).agg(
        total_co2=(TARGET, "sum"), total_kwh=("Usage_kWh", "sum")
    )
    seasonal.to_csv(result_dir / "seasonal_totals.csv", index=False)

    load = df.groupby("Load_Type", as_index=False)["Usage_kWh"].mean()
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.bar(load.Load_Type, load.Usage_kWh)
    ax.set(title="Average electricity use by load category", ylabel="Mean kWh")
    fig.savefig(figure_dir / "usage_by_load_type.png", dpi=160)
    plt.close(fig)
    print("EDA artifacts saved in figures/ and results/.")


if __name__ == "__main__":
    main()
