from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.patchtst import PatchTST, DiffPatchTST, PositionalEncoding
from nazuna.criteria import MSE
import torch


def test_forward(device, dummy_data):
    model = PatchTST.create(
        device=device,
        seq_len=16,
        pred_len=4,
        c_in=3,
    )
    batch = dummy_data((1, 16, 3))
    output, _ = model(batch)
    assert list(output.size()) == [1, 4, 3]


def test_get_loss(device, dummy_data):
    model = PatchTST.create(
        device=device,
        seq_len=16,
        pred_len=4,
        c_in=3,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 16, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=torch.tensor([[
            [50., 50., 50.],
            [60., 60., 60.],
            [70., 70., 70.],
            [80., 80., 80.],
        ]], device=device),
        quantiles=None,
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)


def test_diff_patchtst_forward(device):
    # seq_len=17: diff produces 16 steps, matching patch_len=16
    model = DiffPatchTST.create(
        device=device,
        seq_len=17,
        pred_len=4,
        c_in=3,
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
        c_in=3,
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
        quantiles=None,
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)


def test_positional_encoding_zero_init():
    d_model = 32
    max_len = 10
    pe_module = PositionalEncoding(d_model=d_model, max_len=max_len)
    pe = pe_module.pe  # [max_len, d_model]

    assert pe.shape == (max_len, d_model)
    assert torch.all(pe == 0)
