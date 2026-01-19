# Nazuna

Nazuna provides utilities for analyzing time-series forecasting models.  
https://nazuna.readthedocs.io/en/latest/  

> [!IMPORTANT]
>  Nazuna requires `torch`, but it does **not** install it automatically. Please install either the CPU or CUDA version of PyTorch by yourself before using Nazuna.


> [!NOTE]
> The dataset under `nazuna/datasets/jma/` was obtained from the following Japan Meteorological Agency (JMA) pages and formatted by the author.  
> ["気象庁ホーム > 各種データ・資料 > 過去の気象データ検索 > 日ごとの値"](https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no=51&block_no=47636&year=2025&month=12&day=&view=)  
> ["気象庁ホーム > 各種データ・資料 > 過去の気象データ検索 > １時間ごとの値"](https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no=51&block_no=47636&year=2025&month=12&day=1&view=)

### Repository Structure

This repository can be installed as a Python package.

```sh
./
├─ pyproject.toml
│
├─ nazuna/
│  ├─ data_manager.py  # Time-series data management class
│  ├─ criteria.py  # Loss functions for training and evaluation
│  ├─ batch_sampler.py  # Batch sampler
│  ├─ scaler.py  # Scaler (used by models)
│  ├─ models/  # Time-series forecasting models
│  │  ├─ base.py
│  │  ├─ dlinear.py
│  │  └─ patchtst.py
│  ├─ reports/  # Reporting functions
│  │
│  ├─ task_runner.py  # Task runner that orchestrates the above modules
│  │
│  ├─ datasets/  # Sample datasets
│  │  └─ jma/
│  └─ examples/  # Usage examples
│
├─ tests/
└─ docs/
```

### Development Guide (for Developers)

```bash
pip install -e '.[test,docs]'  #  install the package in editable mode
# make some changes to the code in ./nazuna/
# implement tests in ./tests/
ruff check  # lint check
pytest  # run tests locally
# update documentation in ./docs/
mkdocs serve --livereload  # preview documentation locally
# update the version in ./pyproject.toml
# commit changes
```
