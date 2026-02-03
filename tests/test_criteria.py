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

    criterion = load_class('nazuna.criteria.MSE').create(device, n_channel=3, pred_len=4)
    loss = criterion(batch_0, batch_1)

    # ( 1^2 + 2^2 + 3^2 + 4^2 ) / 4 = 30 / 4 = 7.5
    # ( 2^2 + 4^2 + 6^2 + 8^2 ) / 4 = 120 / 4 = 30
    expected = torch.tensor([[7.5, 30., 7.5]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)

    # (7.5 + 30 + 7.5) / 3 = 15
    expected = torch.tensor([15.], device=device)
    assert torch.allclose(loss.each_sample, expected)
    assert torch.allclose(loss.batch_mean, expected[0])

    criterion = load_class('nazuna.criteria.MSE').create(device, n_channel=3, pred_len=2)
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
    criterion = load_class('nazuna.criteria.MSE').create(
        device, n_channel=3, pred_len=4, tolerance=1.5,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[7.25, 30., 7.25]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)

    # tolerance=0: default behavior (same as no tolerance)
    criterion = load_class('nazuna.criteria.MSE').create(
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
    criterion = load_class('nazuna.criteria.MAE').create(
        device, n_channel=3, pred_len=4, tolerance=1.5,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[2.25, 5., 2.25]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)

    # tolerance=0: default behavior (same as no tolerance)
    # channel 0: (1 + 2 + 3 + 4) / 4 = 10 / 4 = 2.5
    # channel 1: (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5
    # channel 2: (1 + 2 + 3 + 4) / 4 = 10 / 4 = 2.5
    criterion = load_class('nazuna.criteria.MAE').create(
        device, n_channel=3, pred_len=4, tolerance=0,
    )
    loss = criterion(batch_0, batch_1)
    expected = torch.tensor([[2.5, 5., 2.5]], device=device)
    assert torch.allclose(loss.each_sample_channel, expected)


def test_improvement_rate(device):
    true = torch.tensor([[
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=device)
    baseline = torch.tensor([[
        [12., 14., 12.],
        [24., 28., 24.],
        [36., 42., 36.],
        [48., 56., 48.],
    ]], device=device)
    pred = torch.tensor([[
        [11., 12., 11.],
        [22., 24., 22.],
        [33., 36., 33.],
        [44., 48., 44.],
    ]], device=device)

    criterion = load_class('nazuna.criteria.ImprovementRate').create(
        device, n_channel=3, pred_len=4, error_type="mse",
    )
    result = criterion(baseline, pred, true)

    # baseline errors (MSE per channel):
    # channel 0: (2^2 + 4^2 + 6^2 + 8^2) / 4 = 120 / 4 = 30
    # channel 1: (4^2 + 8^2 + 12^2 + 16^2) / 4 = 480 / 4 = 120
    # channel 2: (2^2 + 4^2 + 6^2 + 8^2) / 4 = 120 / 4 = 30
    #
    # pred errors (MSE per channel):
    # channel 0: (1^2 + 2^2 + 3^2 + 4^2) / 4 = 30 / 4 = 7.5
    # channel 1: (2^2 + 4^2 + 6^2 + 8^2) / 4 = 120 / 4 = 30
    # channel 2: (1^2 + 2^2 + 3^2 + 4^2) / 4 = 30 / 4 = 7.5
    #
    # improvement rate per channel = 1 - pred_error / baseline_error
    # channel 0: 1 - 7.5 / 30 = 1 - 0.25 = 0.75
    # channel 1: 1 - 30 / 120 = 1 - 0.25 = 0.75
    # channel 2: 1 - 7.5 / 30 = 1 - 0.25 = 0.75
    expected_each_sample_channel = torch.tensor([[0.75, 0.75, 0.75]], device=device)
    assert torch.allclose(result.each_sample_channel, expected_each_sample_channel)

    # average over channels: (0.75 + 0.75 + 0.75) / 3 = 0.75
    expected_each_sample = torch.tensor([0.75], device=device)
    assert torch.allclose(result.each_sample, expected_each_sample)
    assert torch.allclose(result.batch_mean, expected_each_sample[0])

    # check info contains intermediate errors
    assert "error_baseline" in result.info
    assert "error_pred" in result.info
    expected_baseline_error = torch.tensor([[30., 120., 30.]], device=device)
    expected_pred_error = torch.tensor([[7.5, 30., 7.5]], device=device)
    assert torch.allclose(result.info["error_baseline"], expected_baseline_error)
    assert torch.allclose(result.info["error_pred"], expected_pred_error)


def test_improvement_rate_mae(device):
    true = torch.tensor([[
        [10., 10.],
        [20., 20.],
    ]], device=device)
    baseline = torch.tensor([[
        [14., 18.],
        [28., 36.],
    ]], device=device)
    pred = torch.tensor([[
        [12., 14.],
        [24., 28.],
    ]], device=device)

    criterion = load_class('nazuna.criteria.ImprovementRate').create(
        device, n_channel=2, pred_len=2, error_type="mae",
    )
    result = criterion(baseline, pred, true)

    # baseline errors (MAE per channel):
    # channel 0: (4 + 8) / 2 = 6
    # channel 1: (8 + 16) / 2 = 12
    #
    # pred errors (MAE per channel):
    # channel 0: (2 + 4) / 2 = 3
    # channel 1: (4 + 8) / 2 = 6
    #
    # improvement rate per channel = 1 - pred_error / baseline_error
    # channel 0: 1 - 3 / 6 = 0.5
    # channel 1: 1 - 6 / 12 = 0.5
    expected_each_sample_channel = torch.tensor([[0.5, 0.5]], device=device)
    assert torch.allclose(result.each_sample_channel, expected_each_sample_channel)

    # average over channels: 0.5
    expected_each_sample = torch.tensor([0.5], device=device)
    assert torch.allclose(result.each_sample, expected_each_sample)


def test_improvement_rate_negative(device):
    """Test case where pred is worse than baseline (negative improvement)."""
    true = torch.tensor([[
        [10., 10.],
        [20., 20.],
    ]], device=device)
    baseline = torch.tensor([[
        [11., 11.],
        [21., 21.],
    ]], device=device)
    pred = torch.tensor([[
        [12., 12.],
        [22., 22.],
    ]], device=device)

    criterion = load_class('nazuna.criteria.ImprovementRate').create(
        device, n_channel=2, pred_len=2, error_type="mse",
    )
    result = criterion(baseline, pred, true)

    # baseline errors (MSE per channel):
    # channel 0: (1^2 + 1^2) / 2 = 1
    # channel 1: (1^2 + 1^2) / 2 = 1
    #
    # pred errors (MSE per channel):
    # channel 0: (2^2 + 2^2) / 2 = 4
    # channel 1: (2^2 + 2^2) / 2 = 4
    #
    # improvement rate = 1 - 4 / 1 = -3 (negative = worse than baseline)
    expected_each_sample_channel = torch.tensor([[-3., -3.]], device=device)
    assert torch.allclose(result.each_sample_channel, expected_each_sample_channel)
