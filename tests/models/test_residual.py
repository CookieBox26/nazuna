from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.residual import (
    ResidualModel, ResidualModel2, ResidualModel3, ResidualDeseasonModel,
    ResidualRegularizedModel,
)
from nazuna.criteria import MSE
import torch


def test_forward(device, dummy_data):
    model = ResidualModel.create(
        device=device,
        seq_len=96,
        pred_len=24,
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 96, 'pred_len': 24, 'period_len': 24, 'decay_rate': 1.0},
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 96, 'pred_len': 24, 'kernel_size': 25, 'bias': True,
        },
    )
    batch = dummy_data((2, 96, 3))
    output, _ = model(batch)
    assert list(output.size()) == [2, 24, 3]


def test_get_loss(device, dummy_data):
    model = ResidualModel.create(
        device=device,
        seq_len=16,
        pred_len=4,
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 16, 'pred_len': 4, 'period_len': 4, 'decay_rate': 1.0},
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 16, 'pred_len': 4, 'kernel_size': 25, 'bias': True,
        },
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 16, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0


def test_forward_naive_diff(device, dummy_data):
    # The naive sub-model uses prep_type="diff", so its forward consumes
    # seq_len = ResidualModel.seq_len - 1 after differencing.
    model = ResidualModel.create(
        device=device,
        seq_len=17,
        pred_len=4,
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={
            'seq_len': 16, 'pred_len': 4, 'period_len': 4, 'prep_type': 'diff',
        },
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 17, 'pred_len': 4, 'kernel_size': 25, 'bias': True,
        },
    )
    x = dummy_data((1, 17, 3))
    output, _ = model(x)
    assert list(output.size()) == [1, 4, 3]


def test_forward_neural_revin(device, dummy_data):
    # A neural sub-model with use_revin has RevIN applied inside forward.
    model = ResidualModel.create(
        device=device,
        seq_len=16,
        pred_len=4,
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 16, 'pred_len': 4, 'period_len': 4},
        neural_model_cls_path='nazuna.models.unitst.UniTSTLike',
        neural_model_params={
            'seq_len': 16, 'pred_len': 4, 'c_in': 3,
            'patch_len': 8, 'stride': 8, 'd_model': 32, 'n_heads': 4,
            'd_ff': 64, 'e_layers': 2, 'n_dispatchers': 4, 'use_revin': True,
        },
    )
    x = dummy_data((1, 16, 3))
    out_on, _ = model(x)
    model.neural_model.use_revin = False
    out_off, _ = model(x)
    assert list(out_on.size()) == [1, 4, 3]
    assert not torch.allclose(out_on, out_off)


def test_deseason_forward(device, dummy_data):
    model = ResidualDeseasonModel.create(
        device=device,
        seq_len=16,
        pred_len=4,
        period_len=4,
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 16, 'pred_len': 4, 'period_len': 4},
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 16, 'pred_len': 4, 'kernel_size': 25, 'bias': True,
        },
    )
    x = dummy_data((1, 16, 3))
    output, _ = model(x)
    assert list(output.size()) == [1, 4, 3]

    # A perfectly periodic input (period 4) has its seasonal mean equal to the
    # input itself, so the deseasonalized residual fed to the neural model is 0.
    residual_in = x - model._seasonal_reconstruction(x)
    assert torch.allclose(residual_in, torch.zeros_like(residual_in), atol=1e-5)


def test_residual_regularized_get_loss_and_backward(device, dummy_data):
    model = ResidualRegularizedModel.create(
        device=device,
        seq_len=16,
        pred_len=4,
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 16, 'pred_len': 4, 'period_len': 4},
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 16, 'pred_len': 4, 'kernel_size': 25, 'bias': True,
        },
        reg_pred=1.0,
        reg_coef=1.0,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 16, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss_and_backward(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0
    # The penalty (neural correction magnitude + weight) is added on top of the
    # plain criterion loss, and the weight `a` takes part in the gradient.
    assert loss.grad_target.item() > loss.each_sample.mean().item()
    assert model.a.grad is not None


def test_residual_model2_forward(device, dummy_data):
    n_channel = 3
    model = ResidualModel2.create(
        device=device,
        n_channel=n_channel,
        seq_len=96,
        pred_len=24,
        naive_model_cls_path=(
            'nazuna.models.simple_average.SimpleAverage'
        ),
        naive_model_params={
            'seq_len': 96, 'pred_len': 24,
            'period_len': 24, 'decay_rate': 1.0,
        },
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 96, 'pred_len': 24,
            'kernel_size': 25, 'bias': True,
        },
    )
    assert list(model.w_naive.size()) == [n_channel]

    batch = dummy_data((2, 96, 3))
    output, _ = model(batch)
    assert list(output.size()) == [2, 24, n_channel]

    # w_naive=1 -> output == naive_out
    with torch.no_grad():
        model.w_naive.fill_(1.0)
        out_w1, _ = model(batch)
        naive_out = model.naive_model(batch)
        if isinstance(naive_out, tuple):
            naive_out = naive_out[0]
        assert torch.allclose(out_w1, naive_out)

    # w_naive=0 -> output == neural_out
    with torch.no_grad():
        model.w_naive.fill_(0.0)
        out_w0, _ = model(batch)
        neural_out = model.neural_model(batch)
        if isinstance(neural_out, tuple):
            neural_out = neural_out[0]
        assert torch.allclose(out_w0, neural_out)


def _make_batch_for_residual3(device, dummy_data, n_channel=3):
    return TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 16, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={'qtile_full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )


def _create_residual_model3(device, n_channel=3):
    return ResidualModel3.create(
        device=device,
        n_channel=n_channel,
        seq_len=16,
        pred_len=4,
        naive_model_cls_path=(
            'nazuna.models.simple_average.SimpleAverage'
        ),
        naive_model_params={
            'seq_len': 16, 'pred_len': 4,
            'period_len': 4, 'decay_rate': 1.0,
        },
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 16, 'pred_len': 4,
            'kernel_size': 25, 'bias': True,
        },
    )


def test_residual_model3_forward(device, dummy_data):
    n_channel = 3
    model = _create_residual_model3(device, n_channel)
    batch = dummy_data((2, 16, 3))
    output, info = model(batch)
    assert list(output.size()) == [2, 4, n_channel]
    assert 'naive' in info


def test_residual_model3_predict(device, dummy_data):
    n_channel = 3
    model = _create_residual_model3(device, n_channel)
    batch = _make_batch_for_residual3(device, dummy_data, n_channel)
    output, info = model.predict(batch)
    assert list(output.size()) == [1, 4, n_channel]


def test_residual_model3_get_loss_and_backward(device, dummy_data):
    n_channel = 3
    model = _create_residual_model3(device, n_channel)
    batch = _make_batch_for_residual3(device, dummy_data, n_channel)
    criterion = MSE.create(device, n_channel=n_channel, pred_len=4)
    loss = model.get_loss_and_backward(batch, criterion)
    assert loss.each_sample.mean().item() > 0.0
    assert model.w_naive.grad is not None
