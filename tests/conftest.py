from nazuna.data_manager import TimeSeriesDataManager
from nazuna.datasets import get_dataset_path
import pytest
import torch
import sys


def pytest_configure(config):
    # macOS has a stricter path length limit than Linux/Windows
    if sys.platform == "darwin":
        config.option.basetemp = "/tmp/pytest"


@pytest.fixture(scope='session')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='session')
def get_data_manager():
    def _get_data_manager(conf_=None):
        conf_data = {
            'path': get_dataset_path('jma', 'weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv'),
            'colname_timestamp': 'date', 'seq_len': 7 * 4, 'pred_len': 7,
            'white_list': ['temp_avg_nagoya', 'temp_avg_fukuoka'],
        }
        if conf_:
            conf_data.update(conf_)
        return TimeSeriesDataManager(**conf_data)
    return _get_data_manager
