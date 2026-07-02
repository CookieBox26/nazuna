from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.crossformer import Crossformer
from nazuna.criteria import MSE
import pytest


@pytest.mark.parametrize('norm_first', [True, False])
@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_forward(device, dummy_data, prep_type, revin_affine, norm_first):
    model = Crossformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=6, d_model=32, n_heads=4, d_ff=64, e_layers=3,
        norm_first=norm_first,
        revin_affine=revin_affine,
        prep_type=prep_type,
    )
    batch = dummy_data((1, 16, 3))
    output, _ = model(batch)
    assert output.shape == (1, 4, 3)


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_get_loss(device, dummy_data, prep_type, revin_affine):
    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    model = Crossformer.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=6, d_model=32, n_heads=4, d_ff=64, e_layers=3,
        revin_affine=revin_affine,
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
    assert loss.each_sample.mean().item() > 0.0
