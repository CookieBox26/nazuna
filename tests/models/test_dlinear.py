from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.dlinear import DLinear, DLinearChannelwise
from nazuna.criteria import MSE
import torch


def test_forward(device):
    model = DLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        kernel_size=3,
        bias=True,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
    output, _ = model(batch)
    assert list(output.size()) == [1, 2, 3]


def test_get_loss(device):
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
        data=torch.tensor([[
            [10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.],
            [40., 40., 40.],
        ]], device=device),
        tsta_future=None,
        tste_future=None,
        data_future=torch.tensor([[
            [50., 50., 50.],
            [60., 60., 60.],
        ]], device=device),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)


def test_channelwise_forward(device):
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
    batch = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
    output, info = model(batch)
    assert list(output.size()) == [1, 2, 3]
    assert 'seasonal' in info
    assert 'trend' in info


def test_channelwise_get_loss(device):
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
        data=torch.tensor([[
            [10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.],
            [40., 40., 40.],
        ]], device=device),
        tsta_future=None,
        tste_future=None,
        data_future=torch.tensor([[
            [50., 50., 50.],
            [60., 60., 60.],
        ]], device=device),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
