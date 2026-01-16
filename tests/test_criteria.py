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
    error, errors_each_sample, errors_each_roads = criterion(batch_0, batch_1)

    # ( 1^2 + 2^2 + 3^2 + 4^2 ) / 4 = 30 / 4 = 7.5
    # ( 2^2 + 4^2 + 6^2 + 8^2 ) / 4 = 120 / 4 = 30
    expected = torch.tensor([[7.5, 30., 7.5]], device=device)
    assert torch.allclose(errors_each_roads, expected)

    # (7.5 + 30 + 7.5) / 3 = 15
    expected = torch.tensor([15.], device=device)
    assert torch.allclose(errors_each_sample, expected)
    assert torch.allclose(error, expected[0])

    criterion = load_class('nazuna.criteria.MSELoss').create(device, n_channel=3, pred_len=2)
    error, errors_each_sample, errors_each_roads = criterion(batch_0, batch_1)

    # ( 1^2 + 2^2 ) / 2 = 5 / 2 = 2.5
    # ( 2^2 + 4^2 ) / 2 = 20 / 2 = 10
    expected = torch.tensor([[2.5, 10., 2.5]], device=device)
    assert torch.allclose(errors_each_roads, expected)
