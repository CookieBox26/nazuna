from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.autoformer import AutoCorrelationLayer, Autoformer
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
@pytest.mark.parametrize('prep_type', ['none', 'diff'])
def test_forward(device, dummy_data, training, independent_heads, prep_type):
    model = Autoformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        d_model=8, n_heads=2, d_ff=32, decomp_kernel=5,
        independent_heads=independent_heads,
        prep_type=prep_type,
    )
    set_training(model, training)

    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    x = dummy_data((1, seq_len_input, 3))
    x_mark_enc = torch.zeros(1, seq_len_input, 4, device=device)
    x_mark_dec = torch.zeros(1, 8 + 4, 4, device=device)
    output, _ = model((x, x_mark_enc, x_mark_dec))
    assert output.shape == (1, 4, 3)


@pytest.mark.parametrize('independent_heads', [False, True])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('prep_type', ['none', 'diff'])
def test_get_loss(device, dummy_data, training, independent_heads, prep_type):
    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    tsta = np.array([[
        np.datetime64('2025-01-01') + np.timedelta64(i, 'D')
        for i in range(seq_len_input)
    ]])
    tsta_future = np.array([[
        np.datetime64('2025-01-17') + np.timedelta64(i, 'D')
        for i in range(4)
    ]])
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=tsta, tste=None, data=dummy_data((1, seq_len_input, 3)),
        tsta_future=tsta_future, tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.], [10., 10., 10.], [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)

    model = Autoformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        d_model=8, n_heads=2, d_ff=32, decomp_kernel=5,
        independent_heads=independent_heads,
        prep_type=prep_type,
    )
    set_training(model, training)
    if not training:
        model.scaler.q1s = torch.tensor([[[0., 0., 0.]]], device=device)
        model.scaler.q2s = torch.tensor([[[10., 10., 10.]]], device=device)
        model.scaler.q3s = torch.tensor([[[20., 20., 20.]]], device=device)

    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0
