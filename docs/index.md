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

### Running Examples

```bash
# Evaluate SimpleAverage model on JMA weather data:
python -m nazuna.examples eval_sa_jma_daily
python -m nazuna.examples eval_sa_jma_hourly_3m
python -m nazuna.examples eval_sa_jma_hourly_24m

# Train SimpleAverageVariableDecay model on JMA weather data:
python -m nazuna.examples train_savd_jma_daily
python -m nazuna.examples train_savd_jma_hourly_3m

# Train SimpleAverageVariableDecayChannelwise model on JMA weather data:
python -m nazuna.examples train_savdc_jma_daily

# Train DLinear model on JMA weather data:
python -m nazuna.examples train_dlinear_jma_hourly_3m
```

### Features

- Scaling coefficients can be changed dynamically.
