# Nazuna

Nazuna provides utilities for analyzing time-series forecasting models.  
For detailed installation and usage instructions, see the documentation:  
https://nazuna.readthedocs.io/en/latest/  

> [!NOTE]
> The dataset under `nazuna/datasets/jma/` was obtained from [Japan Meteorological Agency (JMA) website](https://www.data.jma.go.jp/stats/etrn/index.php) and processed by the author.

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
│  │  ├─ _base.py
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

### Installation
```sh
git clone https://github.com/CookieBox26/nazuna.git
cd nazuna

# If you want to install the CUDA 12.6 version of PyTorch:
uv sync --extra torch-cu126
uv sync --extra torch-cu126 --extra test  # if you want to test

# If you want to install the CPU-only version of PyTorch:
uv sync --extra torch-cpu
uv sync --extra torch-cpu --extra test  # if you want to test
```

### Usage
```sh
# Run example configurations that use bundled JMA weather data
uv run nazuna --example jma_train_dlinear
uv run nazuna --example jma_optuna_dlinear

# Run tasks defined in a TOML config file:
uv run nazuna config.toml
```

### Development Guide (for Developers)
```sh
uv sync --extra torch-cu126 --extra test --extra docs  # CUDA 12.6
uv sync --extra torch-cpu --extra test --extra docs  # CPU

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

### References
The linked page is my notes in Japanese.
- [Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. Advances in Neural Information Processing Systems (NeurIPS 2021), vol. 34, 2021.](https://cookiebox26.github.io/cookipedia/articles/haixu_wu_et_al_2021.html)
- [Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. Are transformers effective for time series forecasting? Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2023), vol. 37, pp. 11121-11128, 2023.](https://cookiebox26.github.io/cookipedia/articles/ailing_zeng_et_al_2023.html)
- [Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. International Conference on Learning Representations (ICLR), 2023.](https://cookiebox26.github.io/cookipedia/articles/yuqi_nie_et_al_2023.html)
- [Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. Proceedings of the 12th International Conference on Learning Representations (ICLR 2024), 2024.](https://cookiebox26.github.io/cookipedia/articles/yong_liu_et_al_2024.html)
