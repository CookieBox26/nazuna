from nazuna import load_class
import torch


def test_mse_loss(device):
    batch_0 = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=device)
    batch_1 = torch.tensor([[
        [11., 12., 11.],
        [22., 24., 22.],
        [33., 36., 33.],
        [44., 48., 44.],
    ]], device=device)

    criterion = load_class('nazuna.criteria.MSELoss').create(device, n_channel=3, pred_len=4)
    loss = criterion(batch_0, batch_1)

    # ( 1^2 + 2^2 + 3^2 + 4^2 ) / 4 = 30 / 4 = 7.5
    # ( 2^2 + 4^2 + 6^2 + 8^2 ) / 4 = 120 / 4 = 30
    expected = torch.tensor([[7.5, 30., 7.5]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)

    # (7.5 + 30 + 7.5) / 3 = 15
    expected = torch.tensor([15.], device=device)
    assert torch.allclose(loss.each_sample, expected)
    assert torch.allclose(loss.batch_mean, expected[0])

    criterion = load_class('nazuna.criteria.MSELoss').create(device, n_channel=3, pred_len=2)
    loss = criterion(batch_0, batch_1)

    # ( 1^2 + 2^2 ) / 2 = 5 / 2 = 2.5
    # ( 2^2 + 4^2 ) / 2 = 20 / 2 = 10
    expected = torch.tensor([[2.5, 10., 2.5]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)


def test_mse_loss_tolerance(device):
    batch_0 = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=device)
    batch_1 = torch.tensor([[
        [11., 12., 11.],
        [22., 24., 22.],
        [33., 36., 33.],
        [44., 48., 44.],
    ]], device=device)

    # tolerance=1.5: errors below 1.5 are treated as zero
    # channel 0: diffs = [1, 2, 3, 4] -> [0, 2, 3, 4] (1 < 1.5)
    # channel 1: diffs = [2, 4, 6, 8] -> [2, 4, 6, 8] (all >= 1.5)
    # channel 2: diffs = [1, 2, 3, 4] -> [0, 2, 3, 4] (1 < 1.5)
    # channel 0: (0^2 + 2^2 + 3^2 + 4^2) / 4 = 29 / 4 = 7.25
    # channel 1: (2^2 + 4^2 + 6^2 + 8^2) / 4 = 120 / 4 = 30
    # channel 2: (0^2 + 2^2 + 3^2 + 4^2) / 4 = 29 / 4 = 7.25
    criterion = load_class('nazuna.criteria.MSELoss').create(
        device, n_channel=3, pred_len=4, tolerance=1.5,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[7.25, 30., 7.25]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)

    # tolerance=0: default behavior (same as no tolerance)
    criterion = load_class('nazuna.criteria.MSELoss').create(
        device, n_channel=3, pred_len=4, tolerance=0,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[7.5, 30., 7.5]], device=device)  # Same as original test
    assert torch.allclose(loss.each_sample_channel, expected)


def test_mae_loss_tolerance(device):
    batch_0 = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=device)
    batch_1 = torch.tensor([[
        [11., 12., 11.],
        [22., 24., 22.],
        [33., 36., 33.],
        [44., 48., 44.],
    ]], device=device)

    # tolerance=1.5: errors below 1.5 are treated as zero
    # channel 0: diffs = [1, 2, 3, 4] -> [0, 2, 3, 4] (1 < 1.5)
    # channel 1: diffs = [2, 4, 6, 8] -> [2, 4, 6, 8] (all >= 1.5)
    # channel 2: diffs = [1, 2, 3, 4] -> [0, 2, 3, 4] (1 < 1.5)
    # channel 0: (0 + 2 + 3 + 4) / 4 = 9 / 4 = 2.25
    # channel 1: (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
    # channel 2: (0 + 2 + 3 + 4) / 4 = 9 / 4 = 2.25
    criterion = load_class('nazuna.criteria.MAELoss').create(
        device, n_channel=3, pred_len=4, tolerance=1.5,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[2.25, 5., 2.25]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)

    # tolerance=0: default behavior (same as no tolerance)
    # channel 0: (1 + 2 + 3 + 4) / 4 = 10 / 4 = 2.5
    # channel 1: (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
    # channel 2: (1 + 2 + 3 + 4) / 4 = 10 / 4 = 2.5
    criterion = load_class('nazuna.criteria.MAELoss').create(
        device, n_channel=3, pred_len=4, tolerance=0,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[2.5, 5., 2.5]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)
