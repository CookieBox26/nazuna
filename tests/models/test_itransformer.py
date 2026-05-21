from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.itransformer import iTransformer
from nazuna.criteria import MSE
import numpy as np
import torch
import pytest


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_forward(device, dummy_data, prep_type, revin_affine):
    model = iTransformer.create(
        device=device,
        seq_len=16,
        pred_len=4,
        c_in=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
        revin_affine=revin_affine,
        prep_type=prep_type,
    )
    x = dummy_data((1, 16, 3))
    x_mark = torch.zeros(1, 16, 4, device=device)
    output, _ = model((x, x_mark))
    assert list(output.size()) == [1, 4, 3]


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_get_loss(device, dummy_data, prep_type, revin_affine):
    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    model = iTransformer.create(
        device=device,
        seq_len=16,
        pred_len=4,
        c_in=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
        revin_affine=revin_affine,
        prep_type=prep_type,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=np.array([[
            np.datetime64('2025-01-01') + np.timedelta64(i, 'D')
            for i in range(seq_len_input)
        ]]),
        tste=None,
        data=dummy_data((1, seq_len_input, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0
