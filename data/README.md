# Data source and attribution

V E, S., Shin, C., & Cho, Y. (2021). *Steel Industry Energy Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C

Source: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ . Retain the attribution when redistributing the data.

`Steel_industry_data.csv` was retrieved November 12, 2026 and is included unmodified. It has 35,040 observations and 11 columns. The project uses `CO2(tCO2)` as the target. The column name suggests tCO2, while UCI's description lists ppm; results therefore retain the native target scale without a physical-unit conversion.

SHA-256:

```text
9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc
```

Derived splits and predictions are stored in `../results/`; the source CSV is never overwritten by the analysis script.
