from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.patchtst import PatchTST, PositionalEncoding
from nazuna.criteria import MSE
import torch
import pytest


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
def test_forward(device, dummy_data, prep_type):
    model = PatchTST.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        prep_type=prep_type,
    )
    batch = dummy_data((1, 16, 3))
    output, _ = model(batch)
    assert list(output.size()) == [1, 4, 3]


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
def test_get_loss(device, dummy_data, prep_type):
    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    model = PatchTST.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        prep_type=prep_type,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, seq_len_input, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0


def test_positional_encoding_uniform_init():
    d_model = 32
    max_len = 10
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len)
    pe = pe_module.pe  # [max_len, d_model]
    assert pe.shape == (max_len, d_model)
    assert torch.all(pe >= -0.02)
    assert torch.all(pe <= 0.02)
