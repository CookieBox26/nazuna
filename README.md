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
│  ├─ batch_samplers.py  # Batch sampler
│  ├─ criteria.py  # Loss functions for training and evaluation
│  ├─ models/  # Time-series forecasting models (some examples)
│  │  ├─ common/  # Common modules across models
│  │  ├─ _base.py
│  │  ├─ simple_average.py
│  │  ├─ autoformer.py
│  │  └─ dlinear.py
│  ├─ analysis/  # Analysis utilities
│  ├─ task_runners.py  # Task runner that orchestrates the above modules
│  ├─ workflow.py  # Executes a sequence of tasks
│  ├─ report.py  # Generates reports
│  ├─ datasets/  # Sample datasets
│  ├─ definitions/  # Bundled definitions
│  └─ examples/  # Example configurations
│
├─ tests/
└─ docs/
```

### Installation
```sh
git clone https://github.com/CookieBox26/nazuna.git
cd nazuna

# Choose according to your environment's GPU
uv sync --extra torch-cu130  # CUDA 13.0
uv sync --extra torch-cu126  # CUDA 12.6
uv sync --extra torch-cpu  # CPU
```

### Usage
```sh
# Run example configurations that use bundled JMA weather data
uv run nazuna --example _example_train_dlinear
uv run nazuna --example _example_optuna_dlinear
uv run nazuna --example jma11_baselines
uv run nazuna --example jma11_linears

# Run tasks defined in a TOML config file:
uv run nazuna config.toml

# Skip all tasks and regenerate the report only (force replot graphs)
uv run nazuna config.toml -s 0-99 -f
```

### Development Guide (for Developers)
```sh
# Choose according to your environment's GPU
uv sync --extra torch-cu130 --extra dev --extra docs  # CUDA 13.0
uv sync --extra torch-cu126 --extra dev --extra docs  # CUDA 12.6
uv sync --extra torch-cpu --extra dev --extra docs  # CPU

# make some changes to the code in ./nazuna/
# implement tests in ./tests/

# lint check
uv run ruff check

# run tests locally
uv run pytest
# uv run pytest -m ""  # run all tests, including slow ones
# uv run pytest -vv  # show individual test names

# update documentation in ./docs/
uv run mkdocs serve --livereload  # preview documentation locally

# developers are encouraged to install pre-commit
# however, do not install it if `timeout` is unavailable,
# as it is used for documentation validation
# (available on linux and git bash, but not on macos)
uv run pre-commit install  # first time only
uv run pre-commit run --all-files  # run manually

# commit changes
```

### References
- [Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. Advances in Neural Information Processing Systems (NeurIPS 2021), vol. 34, 2021.](https://github.com/thuml/Autoformer)
- [Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. Are transformers effective for time series forecasting? Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2023), vol. 37, pp. 11121-11128, 2023.](https://github.com/cure-lab/LTSF-Linear)
- [Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. International Conference on Learning Representations (ICLR), 2023.](https://github.com/yuqinie98/PatchTST)
- [Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. Proceedings of the 12th International Conference on Learning Representations (ICLR 2024), 2024.](https://github.com/thuml/iTransformer)
- [Yu-Hsiang Chen, Hsiao-Hua Chang, Chia-Wen Chen, Si-An Chen, Hsiang-Fu Yu, Cho-Jui Hsieh. Gateformer: Advancing Multivariate Time Series Forecasting through Temporal and Variate-Wise Attention with Gated Representations. arXiv preprint, 2025.](https://github.com/nyuolab/Gateformer)
- [Juncheng Liu, Chenghao Liu, Gerald Woo, Yiwei Wang, Bryan Hooi, Caiming Xiong, Doyen Sahoo. UniTST: Effectively Modeling Inter-Series and Intra-Series Dependencies for Multivariate Time Series Forecasting. Transactions on Machine Learning Research (TMLR), 2025.](https://arxiv.org/abs/2406.04975)
