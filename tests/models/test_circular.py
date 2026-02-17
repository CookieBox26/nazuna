from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.circular import Circular
from nazuna.criteria import MSE
import torch


def test_forward(device):
    model = Circular.create(
        device=device,
        seq_len=4,
        pred_len=4,
        n_channel=3,
        periods=list(range(2, 25)),
        quantile_mode_train='full',
        quantile_mode_eval='full',
    )
    tste_future = torch.tensor([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
    ], dtype=torch.float32, device=device)
    output, _ = model(tste_future)
    assert list(output.size()) == [2, 4, 3]


def _make_quantiles(device):
    # q1=10, q2=50, q3=90 for each of 3 channels
    # shape: [batch=1, 1, n_channel=3]
    q1 = torch.tensor([[[10., 10., 10.]]], device=device)
    q2 = torch.tensor([[[50., 50., 50.]]], device=device)
    q3 = torch.tensor([[[90., 90., 90.]]], device=device)
    # quantiles[mode] shape: [batch, 3, 1, n_channel]
    return {'full': torch.stack([q1, q2, q3], dim=1)}


def test_get_loss(device):
    model = Circular.create(
        device=device,
        seq_len=4,
        pred_len=4,
        n_channel=3,
        periods=list(range(2, 25)),
        quantile_mode_train='full',
        quantile_mode_eval='full',
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=torch.tensor(
            [[0, 1, 2, 3]], dtype=torch.float32, device=device
        ),
        data=torch.tensor([[
            [10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.],
            [40., 40., 40.],
        ]], device=device),
        tsta_future=None,
        tste_future=torch.tensor(
            [[4, 5, 6, 7]], dtype=torch.float32, device=device
        ),
        data_future=torch.tensor([[
            [50., 50., 50.],
            [60., 60., 60.],
            [70., 70., 70.],
            [80., 80., 80.],
        ]], device=device),
        quantiles=_make_quantiles(device),
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
