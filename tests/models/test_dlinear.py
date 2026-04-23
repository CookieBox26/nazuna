from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.dlinear import DLinear, DLinearChannelwise, NLinear
from nazuna.criteria import MSE
import torch


def test_forward(device, dummy_data):
    model = DLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        kernel_size=3,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = dummy_data((1, 4, 3))
    output, _ = model(batch)
    assert list(output.size()) == [1, 2, 3]


def test_get_loss(device, dummy_data):
    model = DLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        kernel_size=3,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0


def test_nlinear_forward(device, dummy_data):
    model = NLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = dummy_data((1, 4, 3))
    output, info = model(batch)
    assert list(output.size()) == [1, 2, 3]
    assert info == {}


def test_nlinear_get_loss(device, dummy_data):
    model = NLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0


def test_channelwise_forward(device, dummy_data):
    model = DLinearChannelwise.create(
        device=device,
        seq_len=4,
        pred_len=2,
        n_channel=3,
        kernel_size=3,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = dummy_data((1, 4, 3))
    output, info = model(batch)
    assert list(output.size()) == [1, 2, 3]
    assert 'seasonal' in info
    assert 'trend' in info


def test_channelwise_get_loss(device, dummy_data):
    model = DLinearChannelwise.create(
        device=device,
        seq_len=4,
        pred_len=2,
        n_channel=3,
        kernel_size=3,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0
