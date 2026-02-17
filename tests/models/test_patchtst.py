from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.patchtst import PatchTST, DiffPatchTST, PositionalEncoding
from nazuna.criteria import MSE
import torch


def test_forward(device):
    model = PatchTST.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = torch.tensor([[
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
    ]], device=device)
    output, _ = model(batch)
    assert list(output.size()) == [1, 4, 3]


def test_get_loss(device):
    model = PatchTST.create(
        device=device,
        seq_len=16,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
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


def test_diff_patchtst_forward(device):
    # seq_len=17: diff produces 16 steps, matching patch_len=16
    model = DiffPatchTST.create(
        device=device,
        seq_len=17,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
    )
    batch = torch.tensor([[
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
        [50., 50., 50.],
    ]], device=device)
    output, _ = model(batch)
    assert list(output.size()) == [1, 4, 3]


def test_diff_patchtst_get_loss(device):
    model = DiffPatchTST.create(
        device=device,
        seq_len=17,
        pred_len=4,
        quantile_mode_train='full',
        quantile_mode_eval='saved',
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
            [50., 50., 50.],
        ]], device=device),
        tsta_future=None,
        tste_future=None,
        data_future=torch.tensor([[
            [60., 60., 60.],
            [70., 70., 70.],
            [80., 80., 80.],
            [90., 90., 90.],
        ]], device=device),
        quantiles={'full': torch.tensor([[
            [0., 0., 0.],
            [10., 10., 10.],
            [20., 20., 20.],
        ]], device=device)},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)


def test_positional_encoding_normalization():
    """Test that PositionalEncoding applies mean-subtraction and std-scaling."""
    d_model = 32
    max_len = 10
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len)
    pe = pe_module.pe  # [max_len, d_model]

    assert pe.shape == (max_len, d_model)

    # After normalization, mean should be approximately 0
    assert torch.abs(pe.mean()) < 1e-5

    # After normalization (pe / (std * 10)), the std should be approximately 0.1
    assert torch.abs(pe.std() - 0.1) < 1e-2
