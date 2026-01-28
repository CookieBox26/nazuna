# Welcome to Nazuna's documentation!

Nazuna provides utilities for analyzing time-series forecasting models.

### Installation

!!! warning

    Nazuna requires `torch`, but it does **not** install it automatically. Please install either the CPU or CUDA version of PyTorch by yourself before using Nazuna.

This package is not yet registered on PyPI. Please install from the [GitHub repository](https://github.com/CookieBox26/nazuna).

```bash
pip install git+https://github.com/CookieBox26/nazuna.git  # main HEAD
pip install git+https://github.com/CookieBox26/nazuna.git@<revision>  # specific revision
```

### Usage

Run tasks defined in a TOML config file:

```bash
python -m nazuna ./my_config.toml
```

### Running Examples

Nazuna includes example configurations that use bundled JMA weather data. You can run them with `python -m nazuna.examples`:

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

### Features

- Scaling coefficients can be changed dynamically.
