from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.autoformer import (
    AutoCorrelationLayer,
    Autoformer, DiffAutoformer,
)
from nazuna.criteria import MSE
import numpy as np
import torch
import pytest
from tests.utils import set_training


@pytest.mark.parametrize('independent_heads, mean_corr', [
    (
        False,
        torch.tensor([
            [1., 2., 3., 4., 5., 6.],  # batch 0
            [26., 25., 24., 23., 22., 21.],  # batch 1
        ]),
    ),
    (
        True,
        torch.tensor([
            [  # batch 0
                [1., 2., 3., 4., 5., 6.],  # head 0
                [21., 22., 23., 24., 25., 26.],  # head 1
            ],
            [  # batch 1
                [4., 5., 6., 1., 2., 3.],  # head 0
                [21., 22., 23., 24., 25., 26.],  # head 1
            ],
        ]),
    ),
])
def test_ac_lagged_aggregation_approx(independent_heads, mean_corr):
    d_model = 8
    n_heads = 2
    ac = AutoCorrelationLayer(
        d_model=d_model, n_heads=n_heads,
        independent_heads=independent_heads,
    )
    b = mean_corr.shape[0]
    topk = 3
    v = torch.ones(b, ac.n_heads, 6, ac.d_head)
    ac._lagged_aggregation_approx(b, mean_corr, topk, v)


@pytest.mark.parametrize('independent_heads', [False, True])
@pytest.mark.parametrize('training', [True, False])
def test_forward(device, dummy_data, training, independent_heads):
    model = Autoformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        d_model=8, n_heads=2, d_ff=32, decomp_kernel=5,
        independent_heads=independent_heads,
    )
    set_training(model, training)
    x = dummy_data((1, 16, 3))
    x_mark_enc = torch.zeros(1, 16, 4, device=device)
    x_mark_dec = torch.zeros(1, 8 + 4, 4, device=device)
    output, _ = model((x, x_mark_enc, x_mark_dec))
    assert output.shape == (1, 4, 3)


@pytest.mark.parametrize('independent_heads', [False, True])
@pytest.mark.parametrize('training', [True, False])
def test_get_loss(device, dummy_data, training, independent_heads):
    model = Autoformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        d_model=8, n_heads=2, d_ff=32, decomp_kernel=5,
        independent_heads=independent_heads,
    )
    set_training(model, training)
    tsta = np.array([[
        np.datetime64('2025-01-01') + np.timedelta64(i, 'D')
        for i in range(16)
    ]])
    tsta_future = np.array([[
        np.datetime64('2025-01-17') + np.timedelta64(i, 'D')
        for i in range(4)
    ]])
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=tsta, tste=None, data=dummy_data((1, 16, 3)),
        tsta_future=tsta_future, tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.], [10., 10., 10.], [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0


@pytest.mark.parametrize('independent_heads', [False, True])
@pytest.mark.parametrize('training', [True, False])
def test_diff_autoformer_forward(device, dummy_data, training, independent_heads):
    model = DiffAutoformer.create(
        device=device, seq_len=17, pred_len=4, c_in=3,
        d_model=8, n_heads=2, d_ff=32, decomp_kernel=5,
        independent_heads=independent_heads,
    )
    set_training(model, training)
    x = dummy_data((1, 17, 3))
    x_mark_enc = torch.zeros(1, 17, 4, device=device)
    x_mark_dec = torch.zeros(1, 8 + 4, 4, device=device)
    output, _ = model((x, x_mark_enc, x_mark_dec))
    assert output.shape == (1, 4, 3)


@pytest.mark.parametrize('independent_heads', [False, True])
@pytest.mark.parametrize('training', [True, False])
def test_diff_autoformer_get_loss(device, dummy_data, training, independent_heads):
    model = DiffAutoformer.create(
        device=device, seq_len=17, pred_len=4, c_in=3,
        d_model=8, n_heads=2, d_ff=32, decomp_kernel=5,
        independent_heads=independent_heads,
    )
    set_training(model, training)
    tsta = np.array([[
        np.datetime64('2025-01-01') + np.timedelta64(i, 'D')
        for i in range(17)
    ]])
    tsta_future = np.array([[
        np.datetime64('2025-01-18') + np.timedelta64(i, 'D')
        for i in range(4)
    ]])
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=tsta, tste=None, data=dummy_data((1, 17, 3)),
        tsta_future=tsta_future, tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.], [10., 10., 10.], [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0
