from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.residual import ResidualModel, ResidualModel2, ResidualModel3
from nazuna.criteria import MSE
import torch


def test_forward(device):
    model = ResidualModel.create(
        device=device,
        seq_len=96,
        pred_len=24,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 96, 'pred_len': 24, 'period_len': 24, 'decay_rate': 1.0},
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 96, 'pred_len': 24, 'kernel_size': 25, 'bias': True,
            'quantile_mode_train': 'full', 'quantile_mode_eval': 'saved',
        },
    )
    batch = torch.randn(2, 96, 3, device=device)
    output, _ = model(batch)
    assert list(output.size()) == [2, 24, 3]


def test_get_loss(device):
    model = ResidualModel.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
        naive_model_cls_path='nazuna.models.simple_average.SimpleAverage',
        naive_model_params={'seq_len': 16, 'pred_len': 4, 'period_len': 4, 'decay_rate': 1.0},
        neural_model_cls_path='nazuna.models.dlinear.DLinear',
        neural_model_params={
            'seq_len': 16, 'pred_len': 4, 'kernel_size': 25, 'bias': True,
            'quantile_mode_train': 'full', 'quantile_mode_eval': 'saved',
        },
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
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
        tsta_future=None,
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


def test_residual_model2_forward(device):
    n_channel = 3
    model = ResidualModel2.create(
        device=device,
        n_channel=n_channel,
        seq_len=96,
        pred_len=24,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
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
            'quantile_mode_train': 'full',
            'quantile_mode_eval': 'saved',
        },
    )
    assert list(model.w_naive.size()) == [n_channel]

    batch = torch.randn(2, 96, n_channel, device=device)
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


def _make_batch_for_residual3(device, n_channel=3):
    return TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
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
        tsta_future=None,
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


def _create_residual_model3(device, n_channel=3):
    return ResidualModel3.create(
        device=device,
        n_channel=n_channel,
        seq_len=16,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
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
            'quantile_mode_train': 'full',
            'quantile_mode_eval': 'saved',
        },
    )


def test_residual_model3_forward(device):
    n_channel = 3
    model = _create_residual_model3(device, n_channel)
    batch = torch.randn(2, 16, n_channel, device=device)
    output, info = model(batch)
    assert list(output.size()) == [2, 4, n_channel]
    assert 'naive' in info


def test_residual_model3_predict(device):
    n_channel = 3
    model = _create_residual_model3(device, n_channel)
    batch = _make_batch_for_residual3(device, n_channel)
    output, info = model.predict(batch)
    assert list(output.size()) == [1, 4, n_channel]


def test_residual_model3_get_loss_and_backward(device):
    n_channel = 3
    model = _create_residual_model3(device, n_channel)
    batch = _make_batch_for_residual3(device, n_channel)
    criterion = MSE.create(device, n_channel=n_channel, pred_len=4)
    loss = model.get_loss_and_backward(batch, criterion)
    assert loss.batch_mean.item() >= 0
    assert model.w_naive.grad is not None
