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
    expected = {
        'emb': {id(p) for p in model.patch_proj.parameters()},
        'pos': {id(model.pos_enc)},
        'dispatcher': {id(model.dispatcher)},
        'out': {id(p) for p in model.out_proj.parameters()},
    }

    adam = {'cls_path': 'torch.optim.Adam', 'params': {'lr': 0.01}}
    config = {name: {'optimizer': adam} for name in
              ['emb', 'pos', 'dispatcher', 'out', 'body']}
    groups = TrainTaskRunner._OptimizerGroups(config)
    model.set_optimizers(groups)
    ids = {name: _optimized_param_ids(g.optimizer) for name, g in groups.groups.items()}
    for name, expected_ids in expected.items():
        assert ids[name] == expected_ids
    assert set.union(*ids.values()) == trainable
    for a in ids:
        for b in ids:
            if a != b:
                assert ids[a].isdisjoint(ids[b])

    # Without a dispatcher, the dispatcher group is dropped instead of created.
    model = UniTSTLike.create(
        device=device, seq_len=16, pred_len=4, c_in=3,
        patch_len=8, stride=8,
        d_model=32, n_heads=4, d_ff=64, e_layers=2, use_dispatcher=False,
    )
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    groups = TrainTaskRunner._OptimizerGroups(config)
    model.set_optimizers(groups)
    assert 'dispatcher' not in groups.groups
    ids = {name: _optimized_param_ids(g.optimizer) for name, g in groups.groups.items()}
    assert set.union(*ids.values()) == trainable

    # A single 'model' group falls back to optimizing all parameters together.
    groups = TrainTaskRunner._OptimizerGroups({'model': {'optimizer': adam}})
    model.set_optimizers(groups)
    assert _optimized_param_ids(groups.groups['model'].optimizer) == trainable
