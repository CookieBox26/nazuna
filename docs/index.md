# Welcome to Nazuna's documentation!

Nazuna provides utilities for analyzing time-series forecasting models.  
GitHub repo: [https://github.com/CookieBox26/nazuna](https://github.com/CookieBox26/nazuna)

## Installation

This package is not yet registered on PyPI. Please install from the GitHub repo.

!!! warning

    Nazuna requires `torch`, but it does **not** install it automatically. Please install either the CPU or CUDA version of PyTorch by yourself before using Nazuna.

### With pip from a GitHub URL (uv works the same way)
```bash
pip install git+https://github.com/CookieBox26/nazuna.git  # main branch HEAD
pip install git+https://github.com/CookieBox26/nazuna.git@<revision>  # specific revision

# If you want to install the CPU-only version of PyTorch:
pip install "nazuna[torch-cpu] @ git+https://github.com/CookieBox26/nazuna.git"
```

### With uv from a cloned GitHub repo
```bash
git clone https://github.com/CookieBox26/nazuna.git
cd nazuna
git checkout <revision>  # check out a specific revision (optional)
uv sync

# If you want to install the CPU-only version of PyTorch:
uv sync --extra torch-cpu
uv sync --extra torch-cpu --extra test  # if you want to test
```

## Usage
Run tasks defined in a TOML config file:
```bash
python -m nazuna ./out/traffic_eval_sa/config.toml
```
For details on how to write the TOML config file, see [About Config File](config.md).

## Running Examples
Nazuna includes example configurations that use bundled JMA weather data. You can run them with:
```bash
# For debugging Evaluate and Train
python -m nazuna.examples jma_daily_eval_sa
python -m nazuna.examples jma_daily_train_savd

# Examples for JMA weather data (3 months)
python -m nazuna.examples jma_hourly_3m_eval_sa
python -m nazuna.examples jma_hourly_3m_train_savd
python -m nazuna.examples jma_hourly_3m_train_dlinear

# Examples for JMA weather data (24 months)
python -m nazuna.examples jma_hourly_24m_eval_sa
```
