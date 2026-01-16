from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.patchtst import PatchTSTIqr
import torch
import torch.nn as nn


def test_forward(device):
    model = PatchTSTIqr.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode='full',
    )
    batch = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=device)
    output = model(batch)
    assert list(output.size()) == [1, 4, 3]


def test_get_loss(device):
    model = PatchTSTIqr.create(
        device=device,
        seq_len=16,
        pred_len=4,
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
            [10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.],
            [40., 40., 40.],
            [10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.],
            [40., 40., 40.],
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
            [70., 70., 70.],
            [80., 80., 80.],
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
    loss, _ = model.get_loss(batch, criterion)
    assert loss.dim() == 0
