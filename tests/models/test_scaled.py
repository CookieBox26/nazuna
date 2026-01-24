from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.base import BaseModelWithScaler
import torch
import torch.nn as nn


class DummyScaledModel(BaseModelWithScaler):
    """A simple model for testing BaseScaledModel."""

    def _setup(self, seq_len: int, pred_len: int, quantile_mode: str) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self._setup_scaler(quantile_mode)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.linear(x)  # [B, C, H]
        x = x.permute(0, 2, 1)  # [B, H, C]
        return x


class DummyScaledModelWithTuple(BaseModelWithScaler):
    """A model that returns a tuple from forward()."""

    def _setup(self, seq_len: int, pred_len: int, quantile_mode: str) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self._setup_scaler(quantile_mode)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.linear(x)  # [B, C, H]
        x = x.permute(0, 2, 1)  # [B, H, C]
        return x, {'info': 'test'}


def test_forward(device):
    model = DummyScaledModel.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode='full',
    )
    batch = torch.randn(2, 16, 3, device=device)
    output = model(batch)
    assert list(output.size()) == [2, 4, 3]


def test_forward_with_tuple(device):
    model = DummyScaledModelWithTuple.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode='full',
    )
    batch = torch.randn(2, 16, 3, device=device)
    output, info = model(batch)
    assert list(output.size()) == [2, 4, 3]
    assert info == {'info': 'test'}


def test_get_loss(device):
    model = DummyScaledModel.create(
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
    loss = model.get_loss(batch, criterion)


def test_get_loss_with_tuple(device):
    model = DummyScaledModelWithTuple.create(
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
    loss = model.get_loss(batch, criterion)
