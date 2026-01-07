# Nazuna

Nazuna provides utility functions for time-series data analysis.

> [!IMPORTANT]
>  Nazuna requires `torch`, but it does **not** install it automatically. Please install either the CPU or CUDA version of PyTorch by yourself before using Nazuna.


> [!NOTE]
> The test dataset `tests/data/jma-daily_2025.csv` was retrieved and processed by the author from the Japan Meteorological Agency (JMA) website: ["気象庁ホーム > 各種データ・資料 > 過去の気象データ検索 > 日ごとの値"](https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no=51&block_no=47636&year=2025&month=12&day=&view=).

### Development Guide (for Developers)

```
pip install -e '.[test,docs]'  #  install the package in editable mode
# make some changes to the code in ./nazuna/
# implement tests in ./tests/
pytest  # run tests locally
# update documentation in ./docs/
mkdocs serve  # preview documentation locally
# update the version in ./pyproject.toml
# commit changes
```

### Package Structure

```
./
├─ nazuna/
│  ├─ data_manager.py  # TBD
│  ├─ batch_sampler.py
│  └─ models/  # TBD
│      ├─ base.py
│      └─ patchtst.py
├─ tests/
├─ docs/  # TBD
└─ pyproject.toml
```