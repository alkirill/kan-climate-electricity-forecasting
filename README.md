# KAN Forecasting on Climate and Electricity Data

This repository contains the main code and selected research artifacts for a study on the application of Kolmogorov-Arnold Networks (KAN) to forecasting tasks.

The work includes two case studies:

- climate forecasting on multivariate time series data;
- short-term electricity price forecasting on a leakage-safe regional panel dataset for the Russian electricity market.

## Repository Scope

This GitHub version is intended to contain the main code, notebooks, and paper/presentation sources.
It is not intended to store the full raw electricity data dump or all generated intermediate artifacts.

## Main Files

### Core electricity experiments

- `russian_electricity_real_kan.py` — leakage-safe baseline electricity forecasting experiment with KAN and Hybrid KAN.
- `russian_electricity_stronger_kan.py` — strengthened KAN setup with delta target, region embedding, and residual learning.
- `russian_electricity_panel_analysis.py` — panel construction and exploratory analysis utilities.

### Notebooks

- `russian_electricity_eda.ipynb` — EDA and initial analysis of electricity data.
- `russian_electricity_real_kan.ipynb` — notebook wrapper for the leakage-safe KAN experiment.
- `russian_electricity_stronger_kan.ipynb` — notebook wrapper for the strengthened KAN experiment.
- `itmo_kan_timeseries.ipynb` — climate forecasting experiments in the KAN time-series setup.
- `diploma_kan_energy_forecasting.ipynb` — additional electricity forecasting notebook.
- `diploma_kan_ett_forecasting.ipynb` — additional ETT forecasting notebook.

### Paper and presentation

- `paper_icdm_kan/main.tex` — conference-style LaTeX draft of the paper.
- `paper_icdm_kan/references.bib` — bibliography for the paper.
- `generate_energy_slides.py` — script for generating additional electricity slides.
- `generate_final_presentation.py` — concise presentation generator.
- `generate_extended_presentation.py` — extended presentation generator preserving the original slide deck and appending the new electricity block.

## Main Results

### Climate case

- The best model in the climate case was `Hybrid KAN`.
- KAN-based models were useful not only for prediction quality, but also for interpretation through learned phi-functions.

### Electricity case

- In the baseline leakage-safe setup, the strongest overall model was `HistGradientBoosting`.
- After strengthening the KAN formulation, the best KAN-family model became `HybridKANEmbedDelta`.
- The strengthened KAN setup substantially reduced the gap to the best boosting baseline.

## Dependencies

The main Python environment includes:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `jupyter`
- `Pillow`
- `PyMuPDF`

See `requirements.txt` for a simple install list.

## Suggested Repository Structure

Public GitHub repository:

- source code and notebooks;
- paper sources;
- slide generation scripts;
- selected lightweight result figures and metric tables;
- no large raw market dumps unless you explicitly want to publish them.

## What Is Better Not to Upload

Unless you intentionally want an open-data repository, it is better not to upload:

- the full `Данные электроэнергии/` raw folder;
- generated caches such as `__pycache__/`;
- bulky presentation render assets;
- large intermediate CSV files and duplicated exported plots;
- local PDFs that are not necessary for reproducing the code.

## Reproducibility

The code was developed in Jupyter notebooks and Python scripts.
The electricity experiments write metrics, prediction tables, and interpretability plots to `analysis_outputs/`.

If raw data are not published, the repository can still be made useful by:

- keeping the processing scripts;
- documenting the expected input file structure;
- including selected final result tables and figures;
- describing how the leakage-safe panel is constructed.

## How to Run

Example:

```bash
python russian_electricity_real_kan.py
python russian_electricity_stronger_kan.py
```

For notebooks:

```bash
jupyter lab
```

## Publication Notes

This repository is suitable as:

- a thesis companion repository;
- a code supplement for a conference submission;
- a portfolio project showing KAN experiments on real forecasting tasks.
