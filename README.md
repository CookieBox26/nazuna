# Nazuna

Nazuna provides utilities for analyzing time-series forecasting models.  
For detailed installation and usage instructions, see the documentation:  
https://nazuna.readthedocs.io/en/latest/  

> [!NOTE]
> The dataset under `nazuna/datasets/jma/` was obtained from the following Japan Meteorological Agency (JMA) pages and formatted by the author.  
> ["気象庁ホーム > 各種データ・資料 > 過去の気象データ検索 > 日ごとの値"](https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no=51&block_no=47636&year=2025&month=12&day=&view=)  
> ["気象庁ホーム > 各種データ・資料 > 過去の気象データ検索 > １時間ごとの値"](https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no=51&block_no=47636&year=2025&month=12&day=1&view=)


### Repository Structure

This repository mainly consists of the following files:

```sh
./
├─ pyproject.toml
│
├─ nazuna/
│  ├─ data_manager.py  # Time-series data management class
│  ├─ batch_sampler.py  # Batch sampler
│  ├─ criteria.py  # Loss functions for training and evaluation
│  ├─ scaler.py  # Scaler (used by models)
│  ├─ models/  # Time-series forecasting models (some examples)
│  │  ├─ base.py
│  │  ├─ simple_average.py
│  │  ├─ autoformer.py
│  │  └─ dlinear.py
│  ├─ utils/
│  │  └─ report.py  # Generates reports from task results
│  ├─ task_runner.py  # Task runner that orchestrates the above modules
│  ├─ datasets/  # Sample datasets
│  └─ examples/  # Example configurations
│
├─ tests/
└─ docs/
```


### Running Nazuna
```sh
# Run example configurations
uv run nazuna --example jma_daily_train_dlinear

# Run tasks defined in a TOML config file:
uv run nazuna config.toml
```


### Development Guide (for Developers)

```sh
uv sync --extra torch-cpu --extra test --extra docs  # CPU
uv sync --extra torch-cu126 --extra test --extra docs  # CUDA 12.6

# make some changes to the code in ./nazuna/
# implement tests in ./tests/

# lint check
uv run ruff check

# run tests locally
uv run pytest
uv run pytest -m ""  # run all tests, including slow ones

# update documentation in ./docs/
uv run mkdocs serve --livereload  # preview documentation locally

# commit changes
```
