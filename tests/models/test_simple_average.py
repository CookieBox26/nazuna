from nazuna.models.simple_average import SimpleAverage
import torch


def test_simple_average():
    model = SimpleAverage(
        seq_len=4,
        pred_len=2,
        period_len=2,
    )
    batch = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
    output, _ = model(batch)
    assert list(output.size()) == [1, 2, 3]
    expected = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)
    assert torch.allclose(output, expected)