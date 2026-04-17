# Welcome to Nazuna's documentation!

Nazuna provides utilities for analyzing time-series forecasting models.  
GitHub repo: [https://github.com/CookieBox26/nazuna](https://github.com/CookieBox26/nazuna)

!!! info

    The dataset under `nazuna/datasets/jma/` was obtained from [Japan Meteorological Agency (JMA) website](https://www.data.jma.go.jp/stats/etrn/index.php) and processed by the author.

## Installation

This package is not yet registered on PyPI. Please install from the GitHub repo.

### With uv from a cloned GitHub repo
```bash
git clone https://github.com/CookieBox26/nazuna.git
cd nazuna
git checkout <revision>  # check out a specific revision (optional)

# If you want to install the CUDA 12.6 version of PyTorch:
uv sync --extra torch-cu126
uv sync --extra torch-cu126 --extra test  # if you want to test

# If you want to install the CPU-only version of PyTorch:
uv sync --extra torch-cpu
uv sync --extra torch-cpu --extra test  # if you want to test
```

??? note "With pip from a GitHub URL"

    ```bash
    pip install git+https://github.com/CookieBox26/nazuna.git  # main branch HEAD
    pip install git+https://github.com/CookieBox26/nazuna.git@<revision>  # specific revision

    # If you want to install the CPU-only version of PyTorch:
    pip install "nazuna[torch-cpu] @ git+https://github.com/CookieBox26/nazuna.git"
    ```

## Usage
Run tasks defined in a TOML config file:
```bash
uv run python -m nazuna ./out/traffic_eval_sa/config.toml
```
For details on how to write the TOML config file, see [About Config File](config.md).

## Running Examples
Nazuna includes example configurations that use bundled JMA weather data. You can run them with:
```bash
# For debugging Evaluate and Train
uv run python -m nazuna.examples jma_daily_eval_sa
uv run python -m nazuna.examples jma_daily_train_savd

# Examples for JMA weather data (3 months)
uv run python -m nazuna.examples jma_hourly_3m_eval_sa
uv run python -m nazuna.examples jma_hourly_3m_train_savd
uv run python -m nazuna.examples jma_hourly_3m_train_dlinear

# Examples for JMA weather data (36 months)
uv run python -m nazuna.examples jma_hourly_36m_eval_sa
```
