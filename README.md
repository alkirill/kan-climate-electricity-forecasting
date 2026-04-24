# KAN Forecasting on Climate and Electricity Data

This repository contains selected materials for a study of Kolmogorov-Arnold Networks (KAN) in forecasting tasks.

The repository is focused on two directions:

- climate time series forecasting;
- electricity forecasting based on Russian regional market data.

## Contents

### Climate

- `itmo_kan_timeseries.ipynb` — main notebook for climate forecasting experiments with KAN and Hybrid KAN.

### Electricity

- `russian_electricity_eda.ipynb` — exploratory analysis of the electricity dataset.
- `russian_electricity_real_kan.ipynb` — leakage-safe electricity forecasting experiment.
- `russian_electricity_stronger_kan.ipynb` — strengthened KAN setup with improved electricity forecasting results.

### Paper

- `paper_kan/main.tex` — LaTeX draft of the paper.
- `paper_kan/references.bib` — bibliography.

## Main Findings

### Climate case

- The best model in the climate experiments was `Hybrid KAN`.
- KAN-based models were useful both for forecasting and for interpretation through learned phi-functions.

### Electricity case

- In the baseline leakage-safe setup, boosting-based models were stronger than the initial KAN formulations.
- After strengthening the setup, the best KAN-family model became much more competitive.
- The final electricity results show that KAN can be useful as an interpretable nonlinear forecasting approach, even when it is not the absolute best baseline overall.

## Notes

- The repository does not include the full raw electricity data dump.
- Large generated artifacts and local service files are excluded from version control.

## Dependencies

Main environment:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `jupyter`

Install from:

```bash
pip install -r requirements.txt
```
