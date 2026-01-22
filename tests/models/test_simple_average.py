from nazuna.models.simple_average import (
    SimpleAverage, SimpleAverageVariableDecay,
)
import torch


def test_simple_average(device):
    model = SimpleAverage.create(
        device=device,
        seq_len=4,
        pred_len=2,
        period_len=2,
    )

    x = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
    expected = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)

    output, _ = model(x)
    assert torch.allclose(output, expected)


def test_simple_average_variable_decay(device):
    model = SimpleAverageVariableDecay.create(
        device=device,
        seq_len=4,
        pred_len=2,
        period_len=2,
    )
    params = dict(model.named_parameters())
    assert 'decay_rate' in params
    assert params['decay_rate'].requires_grad is True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    x = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
    true = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)

    # Test that backward computes a gradient.
    output, _ = model(x)
    loss = ((output - true) ** 2).mean()
    loss.backward()
    assert model.decay_rate.grad is not None

    # Test that optimizer.step() updates the value.
    before = model.decay_rate.detach().clone()
    optimizer.step()
    after = model.decay_rate.detach()
    assert not torch.allclose(before, after)
