from nazuna import load_class
import math
import pytest


@pytest.fixture(scope='module')
def dm(get_data_manager):
    dm_ = get_data_manager({
        'seq_len': 7 * 4, 'pred_len': 7,
        'white_list': ['temp_avg_nagoya', 'temp_avg_fukuoka'],
    })
    assert len(dm_.df) == 365
    return dm_


def test_data_loader(dm, device):
    data_loader_train = dm.get_data_loader(
        data_range=(0.0, 0.8),
        batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
        batch_sampler_params={'batch_size': 16},
        offset=0, rolling_window=28, device=device,
    )

    n_sample_expected = 264
    assert int((365 - 34) * 0.8) == n_sample_expected
    assert data_loader_train.batch_sampler.n_sample == n_sample_expected

    n_batch_expected = 17
    assert math.ceil(264 / 16) == n_batch_expected
    assert data_loader_train.batch_sampler.n_batch == n_batch_expected
    assert 16 * 16 + 8 == 264

    for i_batch, batch in enumerate(data_loader_train):
        batch_size_actual = 16 if (i_batch != n_batch_expected - 1) else 8
        assert batch.tsta.shape == (batch_size_actual, 7 * 4)
        assert batch.tste.shape == (batch_size_actual, 7 * 4)
        assert batch.data.shape == (batch_size_actual, 7 * 4, 2)
        assert batch.tsta_future.shape == (batch_size_actual, 7)
        assert batch.tste_future.shape == (batch_size_actual, 7)
        assert batch.data_future.shape == (batch_size_actual, 7, 2)
        assert batch.quantiles['full'].shape == (batch_size_actual, 3, 1, 2)
        assert batch.quantiles['cum'].shape == (batch_size_actual, 3, 1, 2)
        assert batch.quantiles['rolling'].shape == (batch_size_actual, 3, 1, 2)

def test_data_loader_offset(dm, device):
    data_loader_train = dm.get_data_loader(
        data_range=(0.0, 0.8),
        batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
        batch_sampler_params={'batch_size': 16},
        offset=28, rolling_window=28, device=device,
    )

    n_batch_expected = 15
    assert math.ceil((int((365 - 34) * 0.8) - 28) / 16) == n_batch_expected
    assert data_loader_train.batch_sampler.n_batch == n_batch_expected

    print()
    for i_batch, batch in enumerate(data_loader_train):
        i_q = 2  # median
        i_col = 0  # temp_avg_nagoya
        for i_sample in [0, 4]:
            print(
                batch.tsta[i_sample][27],  # prediction origin
                ' {:6.3f}'.format(batch.quantiles['full'][i_sample, i_q, 0, i_col]),
                ' {:6.3f}'.format(batch.quantiles['cum'][i_sample, i_q, 0, i_col]),
                ' {:6.3f}'.format(batch.quantiles['rolling'][i_sample, i_q, 0, i_col]),
            )
