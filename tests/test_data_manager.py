from nazuna.data_manager import TimeSeriesDataManager
from nazuna import import_class
import math


def test_data_manager():
    conf_data = {
        'path': 'tests/data/jma-daily_2025.csv',
        'colname_timestamp': 'date', 'seq_len': 7 * 4, 'pred_len': 7,
    }
    dm = TimeSeriesDataManager(**conf_data)
    assert len(dm.df) == 365

    # offset=0
    data_loader_train = dm.get_data_loader(
        data_range=(0.0, 0.8),
        batch_sampler_cls=import_class('nazuna.batch_sampler.BatchSampler'),
        batch_sampler_kwargs={'batch_size': 16},
        offset=0, rolling_window=28,
    )
    for i_batch, batch in enumerate(data_loader_train):
        pass
    assert math.ceil(int((365 - 34) * 0.8) / 16) == 17
    assert (i_batch + 1) == 17

    # offset=28
    data_loader_train = dm.get_data_loader(
        data_range=(0.0, 0.8),
        batch_sampler_cls=import_class('nazuna.batch_sampler.BatchSampler'),
        batch_sampler_kwargs={'batch_size': 16},
        offset=28, rolling_window=28,
    )
    print()
    for i_batch, batch in enumerate(data_loader_train):
        i_q = 2  # median
        i_col = 2  # temp_avg_nagoya
        for i_sample in [0, 4]:
            print(
                batch.tsta[i_sample][27],  # prediction origin
                ' {:6.3f}'.format(batch.quantiles_full[i_sample, i_q, i_col]),
                ' {:6.3f}'.format(batch.quantiles_cum[i_sample, i_q, i_col]),
                ' {:6.3f}'.format(batch.quantiles_rolling[i_sample, i_q, i_col]),
            )
    assert math.ceil((int((365 - 34) * 0.8) - 28) / 16) == 15
    assert (i_batch + 1) == 15
