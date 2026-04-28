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

# Choose according to your environment's GPU
uv sync --extra torch-cu130  # CUDA 13.0
uv sync --extra torch-cu126  # CUDA 12.6
uv sync --extra torch-cpu  # CPU
```

??? info "Installation with pip from a GitHub URL"

    If you prefer installing with pip from a GitHub URL, use the following:
    ```bash

    # Choose according to your environment's GPU
    pip install "nazuna[torch-cu130] @ git+https://github.com/CookieBox26/nazuna.git"
    pip install "nazuna[torch-cu126] @ git+https://github.com/CookieBox26/nazuna.git"
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

- [Workflow](reference_workflow.md) &ndash; Workflow definition (the main entry point for running a nazuna pipeline).
- [Task Runners](reference_task_runners.md) &ndash; Individual tasks executed within a workflow.
- [Models](reference_models.md) &ndash; List of forecasting models.
- [Others](reference_others.md) &ndash; Miscellaneous classes and functions.
