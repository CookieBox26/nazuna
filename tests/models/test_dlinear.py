from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.dlinear import (
    DLinear, DLinearChannelwise, NLinear, NLinearChannelwise,
    NLinearChannelCross,
)
from nazuna.criteria import MSE
import torch
from tests.utils import create_from_doc


def test_doc(device):
    _ = create_from_doc(DLinear, device)
    _ = create_from_doc(DLinearChannelwise, device)
    _ = create_from_doc(NLinear, device)
    _ = create_from_doc(NLinearChannelwise, device)
    _ = create_from_doc(NLinearChannelCross, device)


def test_forward(device, dummy_data):
    model = DLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        kernel_size=3,
        bias=True,
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
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0


def test_channelwise_forward(device, dummy_data):
    model = DLinearChannelwise.create(
        device=device,
        seq_len=4,
        pred_len=2,
        c_in=3,
        kernel_size=3,
        bias=True,
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
        c_in=3,
        kernel_size=3,
        bias=True,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0


def test_nlinear_forward(device, dummy_data):
    model = NLinear.create(
        device=device,
        seq_len=4,
        pred_len=2,
        bias=True,
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
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0


def test_nlinear_channelwise_forward(device, dummy_data):
    model = NLinearChannelwise.create(
        device=device,
        seq_len=4,
        pred_len=2,
        c_in=3,
        bias=True,
    )
    batch = dummy_data((1, 4, 3))
    output, info = model(batch)
    assert list(output.size()) == [1, 2, 3]
    assert info == {}


def test_nlinear_channelwise_get_loss(device, dummy_data):
    model = NLinearChannelwise.create(
        device=device,
        seq_len=4,
        pred_len=2,
        c_in=3,
        bias=True,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0


def test_nlinear_channelcross_get_loss(device, dummy_data):
    model = NLinearChannelCross.create(
        device=device,
        seq_len=4,
        pred_len=2,
        c_in=3,
        bias=True,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 4, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 2, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=2)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0
