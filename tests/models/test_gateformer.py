from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.gateformer import Gateformer
from nazuna.criteria import MSE
import numpy as np
import torch
import pytest


variate_kwargs = {
    'd_model_variate': 16, 'n_heads_variate': 2, 'd_ff_variate': 32,
    'e_layers_variate': 1, 'res_attention_variate': True,
}


@pytest.mark.parametrize('extra_kwargs, use_x_mark', [
    ({}, False),
    (variate_kwargs, False),
    ({'use_time_features': True}, True),
])
@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_forward(device, dummy_data, prep_type, revin_affine, extra_kwargs, use_x_mark):
    model = Gateformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=8, stride=8,
        d_model=32, n_heads=4, d_ff=64, e_layers=2,
        revin_affine=revin_affine,
        prep_type=prep_type,
        **extra_kwargs,
    )
    x = dummy_data((1, 16, 3))
    x_mark = torch.zeros(1, 16, 4, device=device) if use_x_mark else None
    output, _ = model((x, x_mark))
    assert list(output.size()) == [1, 4, 3]


@pytest.mark.parametrize('use_time_features', [False, True])
@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_get_loss(device, dummy_data, prep_type, revin_affine, use_time_features):
    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    model = Gateformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=8, stride=8,
        d_model=32, n_heads=4, d_ff=64, e_layers=2,
        revin_affine=revin_affine,
        prep_type=prep_type,
        use_time_features=use_time_features,
    )
    tsta = None
    if use_time_features:
        tsta = np.array([[
            np.datetime64('2025-01-01') + np.timedelta64(i, 'D')
            for i in range(seq_len_input)
        ]])
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=tsta,
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
