from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.residual import ResidualModel
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
