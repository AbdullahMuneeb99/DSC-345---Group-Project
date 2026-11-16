
# Predicting Electricity-Related CO₂ Emissions in a Steel Manufacturing Facility

**DSC 345 — Machine Learning Group Project**

## Project Overview

This project uses the **Steel Industry Energy Consumption Dataset** from the UCI Machine Learning Repository, containing electricity consumption and operational measurements from DAEWOO Steel Co. Ltd. in South Korea. The dataset consists of 35,040 observations recorded at 15-minute intervals throughout 2018.

Our goal is to apply supervised machine learning regression to estimate electricity-related CO₂ emissions and analyze the relationship between industrial energy consumption and reported emissions.

## Methodology

We preprocess the dataset by checking data quality, preparing numerical and categorical features, and performing a chronological **70% training / 30% testing split**.

We train and compare several regression approaches, including:
- Proportional Baseline
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

Models are evaluated using **MAE, RMSE, and R²**, along with time-series cross-validation. We also perform exploratory data analysis (EDA), feature importance analysis, and seasonal emissions analysis to interpret our findings.

## How to Run

**Requirements:** Python 3.12 and the dependencies listed in `requirements.txt`.

### 1. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Project

```bash
python main.py
```

The program automatically executes data preprocessing, exploratory data analysis, model training, evaluation, and feature importance analysis.

Generated results and visualizations are saved in the `results/` and `figures/` directories.

## Data Source

V E, S., Shin, C., & Cho, Y. (2021). *Steel Industry Energy Consumption* [Dataset]. UCI Machine Learning Repository.

**DOI:** https://doi.org/10.24432/C52G8C

**Dataset License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The original dataset is preserved without modification. Processed results and visualizations are stored separately.
