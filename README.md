# CATS Time Series Anomaly Analysis

## Overview

This project explores the Controlled Anomalies Time Series (CATS) dataset and performs both exploratory data analysis and deeper statistical analysis for anomaly detection.

## Getting Started

### 1. Clone the repository

```
git clone https://github.com/yourusername/cats-anomaly-analysis.git
cd cats-anomaly-analysis
```

### 2. Install uv (if not installed)

Follow the official instructions to install `uv`:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, use PowerShell as documented in the uv docs.

### 3. Create and sync the environment

```
uv sync
```

This will create a virtual environment (e.g. `.venv`) and install dependencies from `pyproject.toml` (such as `pandas`, `matplotlib`, `numpy`, `scikit-learn`, `scipy`, `huggingface_hub`).

### 4. Activate the environment

```
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\activate         # Windows
```

### 5. Download the dataset

`exploration.py` will automatically download `data.csv` from the Hugging Face dataset
`patrickfleith/controlled-anomalies-time-series-dataset` using `huggingface_hub` if `data_final.csv` is not present in the project root.  
Ensure you have an internet connection for the first run.

## How to Run

### Exploratory analysis

```
uv run python exploration.py
```

This will:

- Download the dataset (if needed) into `data_final.csv`.
- Produce basic counts, category distributions, and several visualizations saved under `visualizations/`.

### Statistical analysis

```
uv run python statistical_analysis.py
```

This will:

- Load `data.csv` (make sure it exists or copy/rename `data_final.csv` accordingly).
- Run segment extraction, hypothesis tests, PCA, mutual information, correlation-change analysis, and more.
- Save numeric outputs under `visualizations/stat_results/` and plots under `visualizations/`.

## Project Structure

- `exploration.py` – Data download, cleaning, EDA, and core visualizations.
- `statistical_analysis.py` – Advanced statistical analysis, feature engineering, and anomaly-focused plots.
- `visualizations/` – Output directory for generated figures.
- `visualizations/stat_results/` – Numeric summaries and CSV outputs from the statistical analysis.

## Notes

- Adjust data paths in the scripts if you use a different filename or location for the dataset.
- All commands assume you are in the project root directory.
