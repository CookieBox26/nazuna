from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.autoformer import Autoformer, DiffAutoformer
from nazuna.criteria import MSE
import numpy as np
import torch


def test_forward(device):
    model = Autoformer.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
        c_in=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
        decomp_kernel=3,
    )
    x = torch.tensor([[
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
    x_mark_enc = torch.zeros(1, 16, 4, device=device)
    x_mark_dec = torch.zeros(1, 16 // 2 + 4, 4, device=device)
    output, _ = model((x, x_mark_enc, x_mark_dec))
    assert list(output.size()) == [1, 4, 3]


def test_get_loss(device):
    model = Autoformer.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
        c_in=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
        decomp_kernel=3,
    )
    tsta = np.array([[np.datetime64('2025-01-01') + np.timedelta64(i, 'D') for i in range(16)]])
    tsta_future = np.array(
        [[np.datetime64('2025-01-17') + np.timedelta64(i, 'D') for i in range(4)]]
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=tsta,
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
        tsta_future=tsta_future,
        tste_future=None,
        data_future=torch.tensor([[
            [50., 50., 50.],
            [60., 60., 60.],
            [70., 70., 70.],
            [80., 80., 80.],
        ]], device=device),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.shape == ()
    assert loss.batch_mean.item() >= 0


def test_diff_autoformer_forward(device):
    model = DiffAutoformer.create(
        device=device,
        seq_len=17,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
        c_in=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
        decomp_kernel=3,
    )
    x = torch.tensor([[
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
        [50., 50., 50.],
    ]], device=device)
    x_mark_enc = torch.zeros(1, 17, 4, device=device)
    # DiffAutoformer differences the input: effective encoder length is 16.
    x_mark_dec = torch.zeros(1, 16 // 2 + 4, 4, device=device)
    output, _ = model((x, x_mark_enc, x_mark_dec))
    assert list(output.size()) == [1, 4, 3]


def test_diff_autoformer_get_loss(device):
    model = DiffAutoformer.create(
        device=device,
        seq_len=17,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
        c_in=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
        decomp_kernel=3,
    )
    tsta = np.array([[np.datetime64('2025-01-01') + np.timedelta64(i, 'D') for i in range(17)]])
    tsta_future = np.array(
        [[np.datetime64('2025-01-18') + np.timedelta64(i, 'D') for i in range(4)]]
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=tsta,
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
            [50., 50., 50.],
        ]], device=device),
        tsta_future=tsta_future,
        tste_future=None,
        data_future=torch.tensor([[
            [60., 60., 60.],
            [70., 70., 70.],
            [80., 80., 80.],
            [90., 90., 90.],
        ]], device=device),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.shape == ()
    assert loss.batch_mean.item() >= 0
