# Welcome to Nazuna's documentation!

Nazuna provides utilities for analyzing time-series forecasting models.  
GitHub repo: [https://github.com/CookieBox26/nazuna](https://github.com/CookieBox26/nazuna)

!!! info

    The dataset under `nazuna/datasets/jma/` was obtained from [Japan Meteorological Agency (JMA) website](https://www.data.jma.go.jp/stats/etrn/index.php) and processed by the author.


## Installation
This package is not yet registered on PyPI. Please install from the GitHub repo.

Installing with uv from a cloned GitHub repository is recommended.
```bash
git clone https://github.com/CookieBox26/nazuna.git
cd nazuna

# If you want to install the CUDA 12.6 version of PyTorch:
uv sync --extra torch-cu126
uv sync --extra torch-cu126 --extra dev  # if you want to test

# If you want to install the CPU-only version of PyTorch:
uv sync --extra torch-cpu
uv sync --extra torch-cpu --extra dev  # if you want to test
```

??? info "Installation with pip from a GitHub URL"

    If you prefer installing with pip from a GitHub URL, use the following:
    ```bash
    pip install git+https://github.com/CookieBox26/nazuna.git  # main branch HEAD
    pip install git+https://github.com/CookieBox26/nazuna.git@<revision>  # specific revision

    # If you want to install the CUDA 12.6 version of PyTorch:
    pip install "nazuna[torch-cu126] @ git+https://github.com/CookieBox26/nazuna.git"

    # If you want to install the CPU-only version of PyTorch:
    pip install "nazuna[torch-cpu] @ git+https://github.com/CookieBox26/nazuna.git"
    ```


## Usage
Run tasks defined in a TOML config file:
```bash
# Run example configurations that use bundled JMA weather data
uv run nazuna --example jma_train_dlinear
uv run nazuna --example jma_optuna_dlinear

# Run tasks defined in a TOML config file:
uv run nazuna config.toml
```
For details on how to write the TOML config file, see [How to Run](how_to_run.md).


## Reference

These are the reference documents. Supplementary notes in Japanese are available [here](notes_ja.md).

- [Task Runners](reference_task_runners.md) &ndash; Overview of task execution utilities.
- [Models](reference_models.md) &ndash; List of forecasting models.
- [Others](reference_others.md) &ndash; Miscellaneous classes and functions.
