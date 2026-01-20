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

### Running an Example

Run the following command to evaluate JMA weather data:

```bash
python -m nazuna.examples.eval_jma_hourly
```

### Features

- Scaling coefficients can be changed dynamically.
- This package includes sample datasets. Use `nazuna.datasets.get_path()` to get the path to a sample dataset.
