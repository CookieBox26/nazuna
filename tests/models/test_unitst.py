from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.unitst import UniTSTLike
from nazuna.task_runners import TrainTaskRunner
from nazuna.criteria import MSE
import pytest


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_forward(device, dummy_data, prep_type, revin_affine):
    model = UniTSTLike.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=8, stride=8,
        d_model=32, n_heads=4, d_ff=64, e_layers=2, n_dispatchers=4,
        revin_affine=revin_affine,
        prep_type=prep_type,
    )
    x = dummy_data((1, 16, 3))
    output, _ = model(x)
    assert list(output.size()) == [1, 4, 3]


@pytest.mark.parametrize('prep_type', ['none', 'diff'])
@pytest.mark.parametrize('revin_affine', [True, False])
def test_get_loss(device, dummy_data, prep_type, revin_affine):
    seq_len_input = 16 + (0 if (prep_type == 'none') else 1)
    model = UniTSTLike.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=8, stride=8,
        d_model=32, n_heads=4, d_ff=64, e_layers=2, n_dispatchers=4,
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


def _optimized_param_ids(optimizer):
    return {id(p) for g in optimizer.param_groups for p in g['params']}


def test_set_optimizers(device):
    model = UniTSTLike.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=8, stride=8,
        d_model=32, n_heads=4, d_ff=64, e_layers=2, n_dispatchers=4,
    )
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    emb_expected = {id(p) for p in model.patch_proj.parameters()} | {id(model.pos_enc)}

    adam = {'cls_path': 'torch.optim.Adam', 'params': {'lr': 0.01}}
    groups = TrainTaskRunner._OptimizerGroups({
        'emb': {'optimizer': adam},
        'body': {'optimizer': adam},
    })
    model.set_optimizers(groups)
    emb_ids = _optimized_param_ids(groups.groups['emb'].optimizer)
    body_ids = _optimized_param_ids(groups.groups['body'].optimizer)
    assert emb_ids == emb_expected
    assert emb_ids.isdisjoint(body_ids)
    assert emb_ids | body_ids == trainable

    # A single 'model' group falls back to optimizing all parameters together.
    groups = TrainTaskRunner._OptimizerGroups({'model': {'optimizer': adam}})
    model.set_optimizers(groups)
    assert _optimized_param_ids(groups.groups['model'].optimizer) == trainable
