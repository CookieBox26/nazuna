from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.dlinear import DLinear
import torch
import torch.nn as nn


def test_forward(device):
    model = DLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        kernel_size=3,
        bias=True,
        quantile_mode='full',
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
        quantile_mode='full',
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
        quantiles_full=torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device),
        quantiles_cum=None,
        quantiles_rolling=None,
    )
    criterion = nn.MSELoss()
    loss = model.get_loss(batch, criterion)
