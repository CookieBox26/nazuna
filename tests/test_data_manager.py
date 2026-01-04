from nazuna.data_manager import TimeSeriesDataManager
from nazuna import import_class
import math


def test_data_manager():
    conf_data = {
        'path': 'tests/data/jma-daily_2025.csv',
        'colname_timestamp': 'date', 'seq_len': 7 * 4, 'pred_len': 7,
    }
    dm = TimeSeriesDataManager(**conf_data)
    data_loader_train = dm.get_data_loader(
        data_range=(0.0, 0.8),
        batch_sampler_cls=import_class('nazuna.batch_sampler.BatchSampler'),
        batch_sampler_kwargs={'batch_size': 16},
    )
    for i_batch, batch in enumerate(data_loader_train):
        pass
    assert (i_batch + 1) == math.ceil(int((365 - 34) * 0.8) / 16)
